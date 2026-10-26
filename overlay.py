"""
Turns per-snippet anomaly scores into suspicious intervals and an annotated video.
"""

from typing import Callable, List, Optional

import av
import cv2
import numpy as np

FRAMES_PER_SNIPPET = 16
TIMELINE_H = 70
RED = (60, 60, 235)
GREEN = (120, 200, 80)
WHITE = (240, 240, 240)


def find_intervals(scores, threshold: float, fps: float, min_snippets: int = 2, merge_gap: int = 1) -> List[dict]:
    """Runs of snippets with score >= threshold, as time intervals in seconds."""
    above = np.asarray(scores) >= threshold
    runs, start = [], None
    for i, a in enumerate(list(above) + [False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    sec = FRAMES_PER_SNIPPET / fps
    return [{
        'start_snippet': s, 'end_snippet': e,
        'start_s': round(s * sec, 2), 'end_s': round(e * sec, 2),
        'peak': round(float(np.max(scores[s:e])), 4),
    } for s, e in merged if e - s >= min_snippets]


def _timeline(scores: np.ndarray, threshold: float, width: int) -> np.ndarray:
    """Static score plot drawn once; the playhead is added per frame."""
    img = np.full((TIMELINE_H, width, 3), 25, np.uint8)
    pad = 6
    h = TIMELINE_H - 2 * pad
    xs = np.linspace(0, width - 1, len(scores)) if len(scores) > 1 else np.array([0.0])
    ys = pad + h - np.clip(scores, 0, 1) * h
    ty = int(pad + h - threshold * h)
    for x, y, s in zip(xs, ys, scores):
        if s >= threshold:
            cv2.line(img, (int(x), int(y)), (int(x), pad + h), (40, 40, 120), max(1, width // max(len(scores), 1)))
    for i in range(len(ys) - 1):
        color = RED if max(scores[i], scores[i + 1]) >= threshold else GREEN
        cv2.line(img, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])), color, 2, cv2.LINE_AA)
    for x in range(0, width, 12):
        cv2.line(img, (x, ty), (x + 6, ty), (150, 150, 150), 1)
    cv2.putText(img, f'threshold {threshold:.2f}', (6, max(ty - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def render_annotated_video(src: str, dst: str, scores, threshold: float, max_width: int = 960,
                           progress: Optional[Callable[[float], None]] = None):
    scores = np.asarray(scores, dtype=np.float32)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / w)
    out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    timeline = _timeline(scores, threshold, out_w)

    container = av.open(dst, mode='w')
    stream = container.add_stream('libx264', rate=int(round(fps)))
    stream.width, stream.height = out_w, out_h + TIMELINE_H
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '23', 'preset': 'veryfast', 'movflags': '+faststart'}

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (out_w, out_h))
        s_idx = min(i // FRAMES_PER_SNIPPET, len(scores) - 1)
        score = float(scores[s_idx])
        alarm = score >= threshold

        if alarm:
            cv2.rectangle(frame, (0, 0), (out_w - 1, out_h - 1), RED, 6)
        label = 'SUSPICIOUS ACTIVITY' if alarm else 'normal'
        cv2.rectangle(frame, (8, 8), (250, 58), (0, 0, 0), -1)
        cv2.putText(frame, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED if alarm else GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, f'anomaly score {score:.2f}   t={i / fps:5.1f}s', (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        tl = timeline.copy()
        x = int((s_idx + (i % FRAMES_PER_SNIPPET) / FRAMES_PER_SNIPPET) / max(len(scores), 1) * out_w)
        cv2.line(tl, (x, 0), (x, TIMELINE_H), WHITE, 1)

        composite = cv2.cvtColor(np.vstack([frame, tl]), cv2.COLOR_BGR2RGB)
        for packet in stream.encode(av.VideoFrame.from_ndarray(composite, format='rgb24')):
            container.mux(packet)
        i += 1
        if progress and i % 50 == 0:
            progress(min(i / total, 1.0))

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    cap.release()
