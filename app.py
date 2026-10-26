"""
Shoplifting / anomaly detection web app.

Two real pipelines, both running the trained MSL model (MSL/checkpoints/msl_<ds>_best.pt):
  * Upload a video  -> Video Swin-B features (16-frame snippets) -> MSL -> per-snippet
                       anomaly scores, suspicious intervals and an annotated output video.
  * Dataset explorer -> MSL on the provided UCF-Crime / ShanghaiTech test features,
                       compared against the frame-level ground truth.

Run:  .venv/bin/python app.py   ->  http://localhost:5000
"""

import json
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'MSL'))

from src.datasets.vad_features import FRAMES_PER_SNIPPET, VideoFeatureDataset  # noqa: E402
from src.evaluation.vad_eval import predict_features  # noqa: E402
from src.models import MSLModel  # noqa: E402
from overlay import find_intervals, render_annotated_video  # noqa: E402

UPLOAD_DIR = ROOT / 'uploads'
RESULTS_DIR = ROOT / 'static' / 'results'
CHECKPOINT_DIR = ROOT / 'MSL' / 'checkpoints'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FALLBACK_THRESHOLD = 0.5

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

gpu_lock = threading.Lock()
jobs = {}
_models, _datasets, _dataset_preds = {}, {}, {}
_extractor = None


# --------------------------------------------------------------------------- model / data

def get_model(dataset='ucf'):
    if dataset not in _models:
        path = CHECKPOINT_DIR / f'msl_{dataset}_best.pt'
        if not path.exists():
            abort(503, f'No trained model at {path.relative_to(ROOT)}. Run: cd MSL && python train.py --dataset {dataset}')
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model = MSLModel(**ckpt['model_config']).to(DEVICE).eval()
        model.load_state_dict(ckpt['model_state_dict'])
        _models[dataset] = (model, ckpt)
    return _models[dataset]


def threshold_for(dataset):
    """Alarm threshold calibrated at training time (99th pct of held-out normal snippet scores)."""
    return float(get_model(dataset)[1].get('threshold', FALLBACK_THRESHOLD))


def get_extractor():
    global _extractor
    if _extractor is None:
        from src.videoswin.swin3d_extractor import Swin3DFeatureExtractor
        _extractor = Swin3DFeatureExtractor(device=DEVICE)
    return _extractor


def get_test_set(dataset):
    if dataset not in _datasets:
        _datasets[dataset] = VideoFeatureDataset(dataset, 'test')
    return _datasets[dataset]


def get_dataset_predictions(dataset):
    """Scores every test video once and caches them (a few seconds on GPU)."""
    if dataset not in _dataset_preds:
        model, _ = get_model(dataset)
        ds = get_test_set(dataset)
        with gpu_lock:
            _dataset_preds[dataset] = [predict_features(model, ds[i]['features'], DEVICE) for i in range(len(ds))]
    return _dataset_preds[dataset]


def snippet_gt(frame_gt):
    """Frame-level GT -> fraction of anomalous frames per 16-frame snippet."""
    return frame_gt.reshape(-1, FRAMES_PER_SNIPPET).mean(axis=1)


def clean(x):
    return [round(float(v), 4) for v in x]


# --------------------------------------------------------------------------- pages

@app.route('/')
def index():
    return render_template('app.html')


@app.route('/api/model')
def model_info():
    info = {}
    for ds in ('ucf', 'sht'):
        if not (CHECKPOINT_DIR / f'msl_{ds}_best.pt').exists():
            continue
        model, ckpt = get_model(ds)
        res_file = ROOT / 'MSL' / 'results' / f'msl_{ds}_results.json'
        history = json.loads(res_file.read_text()).get('history', []) if res_file.exists() else []
        info[ds] = {
            'config': ckpt['model_config'],
            'parameters': sum(p.numel() for p in model.parameters()),
            'best': ckpt.get('best'),
            'test_results': ckpt.get('test_results'),
            'threshold': threshold_for(ds),
            'history': history,
        }
    default = info.get('ucf', {}).get('threshold', FALLBACK_THRESHOLD)
    return jsonify({'device': DEVICE, 'models': info, 'threshold': default})


# --------------------------------------------------------------------------- dataset explorer

@app.route('/api/dataset/<dataset>/videos')
def dataset_videos(dataset):
    if dataset not in ('ucf', 'sht'):
        abort(404)
    ds = get_test_set(dataset)
    preds = get_dataset_predictions(dataset)
    videos = [{
        'name': ds.names[i],
        'category': ds.categories[i],
        'label': int(ds.labels[i]),
        'snippets': int(ds.lengths[i]),
        'duration_s': round(float(ds.lengths[i]) * FRAMES_PER_SNIPPET / 30, 1),
        'video_score': round(preds[i]['video'], 4),
        'max_score': round(float(preds[i]['corrected'].max()), 4),
    } for i in range(len(ds))]
    return jsonify({'dataset': dataset, 'videos': videos, 'threshold': threshold_for(dataset)})


@app.route('/api/dataset/<dataset>/video/<name>')
def dataset_video(dataset, name):
    if dataset not in ('ucf', 'sht'):
        abort(404)
    ds = get_test_set(dataset)
    try:
        i = ds.index_of(name)
    except ValueError:
        abort(404)
    p = get_dataset_predictions(dataset)[i]
    gt = snippet_gt(ds.frame_gt[i])
    threshold = float(request.args.get('threshold', threshold_for(dataset)))
    return jsonify({
        'name': name,
        'category': ds.categories[i],
        'label': int(ds.labels[i]),
        'video_score': round(p['video'], 4),
        'scores': clean(p['corrected']),
        'raw_scores': clean(p['snippet']),
        'gt': clean(gt),
        'intervals': find_intervals(p['corrected'], threshold, fps=30.0),
        'gt_intervals': find_intervals(gt, 0.5, fps=30.0),
    })


# --------------------------------------------------------------------------- uploads

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_job(job_id):
    job = jobs[job_id]
    try:
        model, _ = get_model('ucf')
        with gpu_lock:
            job.update(status='loading_backbone', message='Loading Video Swin-B backbone...')
            extractor = get_extractor()

            def on_progress(done, total):
                job.update(status='extracting', progress=int(80 * done / max(total, 1)),
                           message=f'Extracting Video Swin features: frame {done}/{total}')

            t0 = time.time()
            extracted = extractor.extract(job['video_path'], on_progress)
            job.update(status='scoring', progress=80, message='Running MSL model...')
            pred = predict_features(model, torch.from_numpy(extracted['features']), DEVICE)
        extract_time = time.time() - t0

        scores = pred['corrected']
        fps = extracted['fps']
        threshold = job['threshold']
        intervals = find_intervals(scores, threshold, fps)

        job.update(status='rendering', progress=85, message='Rendering annotated video...')
        out_name = f'{job_id}_annotated.mp4'
        render_annotated_video(job['video_path'], str(RESULTS_DIR / out_name), scores, threshold,
                               progress=lambda f: job.update(progress=85 + int(15 * f)))

        result = {
            'id': job_id,
            'filename': job['filename'],
            'created': job['created'],
            'fps': round(fps, 2),
            'num_frames': extracted['num_frames'],
            'duration_s': round(extracted['num_frames'] / fps, 2),
            'snippets': len(scores),
            'video_score': round(pred['video'], 4),
            'max_score': round(float(scores.max()), 4),
            'mean_score': round(float(scores.mean()), 4),
            'threshold': threshold,
            'is_suspicious': bool(len(intervals) > 0),
            'intervals': intervals,
            'scores': clean(scores),
            'raw_scores': clean(pred['snippet']),
            'annotated_video': f'/static/results/{out_name}',
            'processing_time_s': round(extract_time, 1),
        }
        (RESULTS_DIR / f'{job_id}.json').write_text(json.dumps(result))
        np.save(RESULTS_DIR / f'{job_id}_features.npy', extracted['features'])
        job.update(status='complete', progress=100, message='Done', result=result)
    except Exception as e:  # report any failure to the UI instead of hanging
        job.update(status='error', message=f'{type(e).__name__}: {e}')
        app.logger.exception('analysis failed')


@app.route('/api/upload', methods=['POST'])
def upload():
    file = request.files.get('video')
    if file is None or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'}), 400

    job_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
    filename = secure_filename(file.filename)
    path = UPLOAD_DIR / f'{job_id}_{filename}'
    file.save(path)
    threshold = float(request.form.get('threshold') or threshold_for('ucf'))

    jobs[job_id] = {'status': 'queued', 'progress': 0, 'message': 'Queued', 'filename': filename,
                    'video_path': str(path), 'threshold': threshold,
                    'created': datetime.now().isoformat(timespec='seconds')}
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/api/job/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if job is None:
        saved = RESULTS_DIR / f'{secure_filename(job_id)}.json'
        if saved.exists():
            return jsonify({'status': 'complete', 'progress': 100, 'result': json.loads(saved.read_text())})
        abort(404)
    return jsonify({k: v for k, v in job.items() if k != 'video_path'})


@app.route('/api/history')
def history():
    items = []
    for f in sorted(RESULTS_DIR.glob('*.json'), reverse=True):
        r = json.loads(f.read_text())
        items.append({k: r.get(k) for k in ('id', 'filename', 'created', 'duration_s', 'max_score',
                                            'video_score', 'is_suspicious', 'intervals')})
    return jsonify(items)


@app.route('/uploads/<path:name>')
def uploaded_file(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'File too large (max 500 MB)'}), 413


@app.errorhandler(503)
def unavailable(e):
    return jsonify({'error': e.description}), 503


if __name__ == '__main__':
    print(f'Shoplifting detection UI on http://localhost:5000  (device: {DEVICE})')
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
