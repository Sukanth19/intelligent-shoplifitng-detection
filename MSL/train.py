"""
Train MSL on the VideoSwin features (train split only) and evaluate on the test split.

    python train.py --dataset ucf
    python train.py --dataset sht

Training follows the MSL paper (AAAI 2022) in simplified form:
  * Multi-sequence learning: for every video, the K-snippet sequence with the
    highest mean score represents the video. Anomalous videos must score that
    sequence high, normal videos low (BCE + hinge ranking between the two).
  * Self-training: K shrinks over stages (32 -> 16 -> 8 -> 4). From stage 2 on,
    the previous stage's snippet scores on anomalous training videos are used
    as pseudo labels, which refines the snippet-level predictions.
  * Video-level head trained with the video labels; at test time the snippet
    scores are corrected with the video probability.

Model selection uses a held-out 10% of the training videos (video-level AUC),
so the test set is only used for the final report.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.datasets.vad_features import VideoFeatureDataset, pad_collate
from src.evaluation.vad_eval import evaluate, predict_features
from src.models import MSLModel

ROOT = Path(__file__).resolve().parent


def max_sequence_score(scores: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Highest mean score over any k consecutive valid snippets. scores, mask: (B, T)."""
    k = min(k, scores.shape[1])
    s = (scores * mask).unsqueeze(1)
    m = mask.float().unsqueeze(1)
    win_sum = F.avg_pool1d(s, k, stride=1).squeeze(1) * k
    win_cnt = F.avg_pool1d(m, k, stride=1).squeeze(1) * k
    win_mean = win_sum / win_cnt.clamp(min=1)
    return win_mean.masked_fill(win_cnt < 0.5, -1.0).max(dim=1).values.clamp(1e-6, 1 - 1e-6)


def msl_loss(out, labels, mask, k, pseudo=None, args=None):
    snippet = out['snippet_scores']
    labels_f = labels.float()
    seq = max_sequence_score(snippet, mask, k)

    loss_seq = F.binary_cross_entropy(seq, labels_f)
    loss_video = F.cross_entropy(out['video_logits'], labels)

    # Ranking: every anomalous video's top sequence above every normal one's
    a, n = seq[labels == 1], seq[labels == 0]
    loss_rank = F.relu(args.margin - a[:, None] + n[None, :]).mean() if len(a) and len(n) else seq.sum() * 0

    # Temporal smoothness and sparsity on anomalous videos
    valid_pairs = (mask[:, 1:] & mask[:, :-1]).float()
    diff = (snippet[:, 1:] - snippet[:, :-1]) ** 2
    loss_smooth = (diff * valid_pairs)[labels == 1].sum() / valid_pairs[labels == 1].sum().clamp(min=1)
    loss_sparse = (snippet * mask)[labels == 1].sum() / mask[labels == 1].sum().clamp(min=1)

    loss = loss_seq + loss_video + loss_rank + args.lambda_smooth * loss_smooth + args.lambda_sparse * loss_sparse

    # Self-training: snippet pseudo labels from the previous stage
    if pseudo is not None:
        w = mask.float()
        target = pseudo.clamp(0, 1)
        loss_pseudo = (F.binary_cross_entropy(snippet.clamp(1e-6, 1 - 1e-6), target, reduction='none') * w).sum() / w.sum()
        loss = loss + args.lambda_pseudo * loss_pseudo

    return loss


def balanced_batches(labels, batch_size, rng):
    """Each batch: half normal, half anomalous videos. One epoch ~= one pass over the anomalies."""
    normal = [i for i, y in enumerate(labels) if y == 0]
    anomaly = [i for i, y in enumerate(labels) if y == 1]
    rng.shuffle(normal)
    rng.shuffle(anomaly)
    half = batch_size // 2
    for b in range(len(anomaly) // half):
        batch = anomaly[b * half:(b + 1) * half] + [normal[(b * half + j) % len(normal)] for j in range(half)]
        rng.shuffle(batch)
        yield batch


@torch.no_grad()
def make_pseudo_labels(model, dataset, indices, device, max_len):
    """Snippet scores of the current model; normal videos get all-zero targets."""
    model.eval()
    pseudo = {}
    for i in indices:
        sample = pad_collate([dataset[i]], max_len)
        if sample['label'][0] == 0:
            pseudo[i] = torch.zeros(sample['features'].shape[1])
            continue
        scores = predict_features(model, sample['features'][0], device)['corrected']
        # Min-max normalise within the video so the strongest sequence becomes a target of ~1
        scores = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-6)
        pseudo[i] = torch.from_numpy(scores).float()
    return pseudo


@torch.no_grad()
def val_auc(model, dataset, indices, device, k):
    model.eval()
    scores = []
    for i in indices:
        s = dataset[i]
        out = model(s['features'].unsqueeze(0).to(device))
        mask = torch.ones_like(out['snippet_scores'], dtype=torch.bool)
        scores.append(float(max_sequence_score(out['snippet_scores'], mask, k)))
    return roc_auc_score(dataset.labels[indices], scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ucf', choices=['ucf', 'sht'])
    p.add_argument('--epochs', type=int, default=40, help='total epochs, split evenly over the stages')
    p.add_argument('--stages', default='32,16,8,4', help='sequence length K per self-training stage')
    p.add_argument('--batch_size', type=int, default=None, help='default 32 (UCF) / 16 (SHT, only 63 anomalous videos)')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--max_len', type=int, default=200, help='long training videos are pooled to this many snippets')
    p.add_argument('--margin', type=float, default=0.5)
    p.add_argument('--lambda_smooth', type=float, default=0.1)
    p.add_argument('--lambda_sparse', type=float, default=0.01)
    p.add_argument('--lambda_pseudo', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--select', default='last_stage', choices=['last_stage', 'best'],
                   help="checkpoint selection on validation: within the final self-training stage "
                        "(validation only has video labels, so 'best' tends to pick coarse early stages)")
    p.add_argument('--out', default=None, help='checkpoint path (default checkpoints/msl_<dataset>_best.pt)')
    args = p.parse_args()
    if args.batch_size is None:
        args.batch_size = 32 if args.dataset == 'ucf' else 16

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_path = Path(args.out) if args.out else ROOT / 'checkpoints' / f'msl_{args.dataset}_best.pt'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.dataset} features...')
    train_set = VideoFeatureDataset(args.dataset, 'train')
    test_set = VideoFeatureDataset(args.dataset, 'test')
    print(f'  train: {len(train_set)} videos ({train_set.labels.sum()} anomalous) | '
          f'test: {len(test_set)} videos ({test_set.labels.sum()} anomalous) | device: {device}')

    # Stratified 90/10 train/val split of the TRAIN videos
    idx_n = [i for i, y in enumerate(train_set.labels) if y == 0]
    idx_a = [i for i, y in enumerate(train_set.labels) if y == 1]
    rng.shuffle(idx_n)
    rng.shuffle(idx_a)
    n_val_n, n_val_a = max(1, len(idx_n) // 10), max(1, len(idx_a) // 10)
    val_idx = idx_n[:n_val_n] + idx_a[:n_val_a]
    fit_idx = idx_n[n_val_n:] + idx_a[n_val_a:]

    model_cfg = dict(input_dim=1024, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                     num_heads=args.num_heads, ff_dim=args.hidden_dim * 4, dropout=args.dropout,
                     sequence_length=32, sequence_stride=16)
    model = MSLModel(**model_cfg).to(device)
    print(f'  model: {sum(p.numel() for p in model.parameters()):,} parameters')

    stages = [int(k) for k in args.stages.split(',')]
    epochs_per_stage = max(1, args.epochs // len(stages))
    best = {'val_auc': -1}
    pseudo = None
    history = []
    t0 = time.time()

    for stage, k in enumerate(stages):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_per_stage)
        for epoch in range(epochs_per_stage):
            model.train()
            losses = []
            fit_labels = train_set.labels[fit_idx]
            for batch_pos in balanced_batches(fit_labels, args.batch_size, rng):
                batch_idx = [fit_idx[j] for j in batch_pos]
                batch = pad_collate([train_set[i] for i in batch_idx], args.max_len)
                feats = batch['features'].to(device)
                mask = batch['mask'].to(device)
                labels = batch['label'].to(device)
                batch_pseudo = None
                if pseudo is not None:
                    batch_pseudo = torch.zeros_like(mask, dtype=torch.float)
                    for r, i in enumerate(batch_idx):
                        batch_pseudo[r, :len(pseudo[i])] = pseudo[i].to(device)

                out = model(feats, mask)
                loss = msl_loss(out, labels, mask, k, batch_pseudo, args)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()

            v_auc = val_auc(model, train_set, val_idx, device, k)
            history.append({'stage': stage + 1, 'k': k, 'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'val_auc': v_auc})
            print(f'stage {stage + 1} (K={k:2d}) epoch {epoch + 1:2d}/{epochs_per_stage}  '
                  f'loss {np.mean(losses):.4f}  val video-AUC {v_auc:.4f}  [{time.time() - t0:.0f}s]')
            selectable = args.select == 'best' or stage == len(stages) - 1
            if selectable and v_auc > best['val_auc']:
                best = {'val_auc': v_auc, 'stage': stage + 1, 'k': k, 'epoch': epoch + 1}
                torch.save({'model_state_dict': model.state_dict(), 'model_config': model_cfg,
                            'dataset': args.dataset, 'train_args': vars(args), 'best': best}, out_path)

        if stage < len(stages) - 1:
            pseudo = make_pseudo_labels(model, train_set, fit_idx, device, args.max_len)

    print(f'\nBest on validation: {best}  -> {out_path}')
    ckpt = torch.load(out_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Alarm threshold: 99th percentile of snippet scores on the held-out NORMAL training
    # videos, i.e. ~1% of normal footage would trigger an alarm.
    val_normal = np.concatenate([predict_features(model, train_set[i]['features'], device)['corrected']
                                 for i in val_idx if train_set.labels[i] == 0])
    ckpt['threshold'] = float(np.quantile(val_normal, 0.99))
    print(f'Alarm threshold (99th pct of validation normal snippets): {ckpt["threshold"]:.3f}')

    cats =['Shoplifting', 'Stealing', 'Robbery', 'Burglary'] if args.dataset == 'ucf' else []
    results = evaluate(model, test_set, device, cats)
    results.update({'best': best, 'train_time_s': round(time.time() - t0, 1), 'history': history,
                    'train_args': vars(args)})
    ckpt['test_results'] = {k: v for k, v in results.items() if k != 'history'}
    torch.save(ckpt, out_path)

    res_path = ROOT / 'results' / f'msl_{args.dataset}_results.json'
    res_path.parent.mkdir(exist_ok=True)
    res_path.write_text(json.dumps(results, indent=2))
    print('\nTEST (frame-level, never used for training or model selection):')
    for key in ('snippet', 'corrected'):
        print(f'  {key:9s}: ' + '  '.join(f'{m} {v * 100:.2f}' for m, v in results[key].items()))
    print(f'  video-level AUC {results["video_auc"] * 100:.2f}   NaN videos: {results["nan_videos"]}')
    print(f'Saved {res_path}')


if __name__ == '__main__':
    main()
