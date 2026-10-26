// Shoplifting detection UI: upload analysis, dataset explorer, model results.

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
const pct = (v) => (v == null ? '–' : (v * 100).toFixed(2) + '%');
const SNIPPET = 16;

let threshold = 0.5;
let modelInfo = null;
const charts = {};

async function api(url, opts) {
    const res = await fetch(url, opts);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
    return body;
}

// ------------------------------------------------------------------ tabs
document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
        document.querySelectorAll('.page').forEach((p) => p.classList.toggle('active', p.id === `page-${tab.dataset.page}`));
        if (tab.dataset.page === 'explorer' && !explorer.loaded) loadDataset($('ds-select').value);
        if (tab.dataset.page === 'model') renderModel();
    });
});

// ------------------------------------------------------------------ charts
Chart.defaults.color = '#8b96a5';
Chart.defaults.borderColor = '#2a3240';
Chart.defaults.font.family = 'Inter, sans-serif';

// Shades the region above the threshold and draws the threshold line
const thresholdPlugin = {
    id: 'threshold',
    beforeDatasetsDraw(chart, _, opts) {
        if (opts.value == null) return;
        const { ctx, chartArea: a, scales: { y } } = chart;
        const yt = y.getPixelForValue(opts.value);
        ctx.save();
        ctx.fillStyle = 'rgba(239,83,80,0.06)';
        ctx.fillRect(a.left, a.top, a.width, yt - a.top);
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = 'rgba(239,83,80,0.7)';
        ctx.beginPath(); ctx.moveTo(a.left, yt); ctx.lineTo(a.right, yt); ctx.stroke();
        ctx.restore();
    },
};
Chart.register(thresholdPlugin);

function scoreChart(canvasId, { seconds, scores, gt, onClick, cursor, threshold: thr = threshold }) {
    charts[canvasId]?.destroy();
    const datasets = [];
    if (gt) {
        datasets.push({
            label: 'Ground truth anomaly', data: gt, type: 'bar', backgroundColor: 'rgba(139,150,165,0.28)',
            barPercentage: 1, categoryPercentage: 1, order: 2,
        });
    }
    datasets.push({
        label: 'Anomaly score', data: scores, borderWidth: 2, pointRadius: 0, tension: 0.25, order: 1,
        segment: { borderColor: (c) => (Math.max(c.p0.parsed.y, c.p1.parsed.y) >= thr ? '#ef5350' : '#5b8def') },
        borderColor: '#5b8def',
    });
    charts[canvasId] = new Chart($(canvasId), {
        type: 'line',
        data: { labels: seconds, datasets },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { title: { display: true, text: 'time (s)' }, ticks: { maxTicksLimit: 12, callback(v) { return Math.round(this.getLabelForValue(v)); } } },
                y: { min: 0, max: 1, title: { display: true, text: 'score' } },
            },
            plugins: {
                legend: { display: !!gt, labels: { boxWidth: 12 } },
                threshold: { value: thr },
                tooltip: { callbacks: { title: (items) => `t = ${Number(items[0].label).toFixed(1)} s` } },
            },
            onClick: onClick ? (evt, _, chart) => {
                const idx = chart.scales.x.getValueForPixel(evt.x);
                if (idx != null) onClick(Math.max(0, Math.min(idx, seconds.length - 1)));
            } : undefined,
        },
        plugins: cursor ? [cursor] : [],
    });
    return charts[canvasId];
}

const secondsFor = (n, fps) => Array.from({ length: n }, (_, i) => +(i * SNIPPET / fps).toFixed(2));

function renderIntervals(listEl, intervals, onPick) {
    listEl.innerHTML = '';
    if (!intervals.length) {
        listEl.innerHTML = '<li class="none">No interval above the threshold.</li>';
        return;
    }
    intervals.forEach((iv) => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${iv.start_s.toFixed(1)}s → ${iv.end_s.toFixed(1)}s</span><span>peak ${iv.peak.toFixed(2)}</span>`;
        if (onPick) li.addEventListener('click', () => onPick(iv));
        listEl.appendChild(li);
    });
}

// ------------------------------------------------------------------ analyze
let selectedFile = null;
const drop = $('dropzone');
['dragenter', 'dragover'].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', (ev) => pickFile(ev.dataTransfer.files[0]));
$('file-input').addEventListener('change', (ev) => pickFile(ev.target.files[0]));

function pickFile(file) {
    if (!file) return;
    selectedFile = file;
    $('drop-label').textContent = `${file.name} (${(file.size / 1e6).toFixed(1)} MB)`;
    $('analyze-btn').disabled = false;
}

$('threshold').addEventListener('input', (ev) => {
    threshold = parseFloat(ev.target.value);
    $('threshold-value').textContent = threshold.toFixed(2);
});

$('analyze-btn').addEventListener('click', async () => {
    if (!selectedFile) return;
    $('analyze-btn').disabled = true;
    $('analyze-error').classList.add('hidden');
    $('progress').classList.remove('hidden');
    setProgress(0, 'Uploading…');
    try {
        const form = new FormData();
        form.append('video', selectedFile);
        form.append('threshold', threshold);
        const { job_id } = await api('/api/upload', { method: 'POST', body: form });
        pollJob(job_id);
    } catch (err) {
        showError(err.message);
    }
});

function setProgress(p, msg) {
    $('progress-fill').style.width = `${p}%`;
    $('progress-msg').textContent = msg;
}

function showError(msg) {
    $('analyze-error').textContent = msg;
    $('analyze-error').classList.remove('hidden');
    $('progress').classList.add('hidden');
    $('analyze-btn').disabled = false;
}

async function pollJob(id) {
    try {
        const job = await api(`/api/job/${id}`);
        setProgress(job.progress || 0, job.message || job.status);
        if (job.status === 'complete') {
            $('progress').classList.add('hidden');
            $('analyze-btn').disabled = false;
            showResult(job.result);
            loadHistory();
        } else if (job.status === 'error') {
            showError(job.message);
        } else {
            setTimeout(() => pollJob(id), 700);
        }
    } catch (err) {
        showError(err.message);
    }
}

function showResult(r) {
    threshold = r.threshold;
    $('threshold').value = threshold;
    $('threshold-value').textContent = threshold.toFixed(2);
    $('verdict-empty').classList.add('hidden');
    $('verdict').classList.remove('hidden');
    const banner = $('verdict-banner');
    banner.className = `verdict-banner ${r.is_suspicious ? 'alarm' : 'ok'}`;
    banner.textContent = r.is_suspicious
        ? `Suspicious activity detected in ${r.intervals.length} interval${r.intervals.length > 1 ? 's' : ''}`
        : 'No suspicious activity above the threshold';
    $('r-max').textContent = fmt(r.max_score);
    $('r-video').textContent = fmt(r.video_score);
    $('r-duration').textContent = `${r.duration_s.toFixed(1)}s`;
    $('r-time').textContent = r.processing_time_s != null ? `${r.processing_time_s}s` : '–';

    const player = $('player');
    $('player-card').classList.remove('hidden');
    player.src = r.annotated_video;
    const seek = (t) => { player.currentTime = t; player.play(); };
    renderIntervals($('r-intervals'), r.intervals, (iv) => seek(iv.start_s));

    const seconds = secondsFor(r.scores.length, r.fps);
    const cursor = {
        id: 'cursor',
        afterDatasetsDraw(chart) {
            const { ctx, chartArea: a, scales: { x } } = chart;
            const px = x.getPixelForValue(player.currentTime / (SNIPPET / r.fps));
            if (px < a.left || px > a.right) return;
            ctx.save(); ctx.strokeStyle = '#e6edf3'; ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke(); ctx.restore();
        },
    };
    const chart = scoreChart('upload-chart', { seconds, scores: r.scores, onClick: (i) => seek(seconds[i]), cursor });
    player.ontimeupdate = () => chart.draw();
    $('player-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadHistory() {
    const items = await api('/api/history').catch(() => []);
    const body = $('history-table').querySelector('tbody');
    body.innerHTML = '';
    $('history-empty').classList.toggle('hidden', items.length > 0);
    $('history-table').classList.toggle('hidden', items.length === 0);
    items.forEach((h) => {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.innerHTML = `<td>${(h.created || '').replace('T', ' ')}</td><td>${h.filename}</td>
            <td>${fmt(h.duration_s, 1)}s</td><td class="mono">${fmt(h.max_score)}</td>
            <td><span class="pill ${h.is_suspicious ? 'alarm' : 'ok'}">${h.is_suspicious ? 'suspicious' : 'normal'}</span></td>`;
        tr.addEventListener('click', async () => showResult((await api(`/api/job/${h.id}`)).result));
        body.appendChild(tr);
    });
}

// ------------------------------------------------------------------ explorer
const explorer = { loaded: false, videos: [], dataset: 'ucf', selected: null, threshold: 0.5 };

async function loadDataset(ds) {
    explorer.dataset = ds;
    const body = $('videos-table').querySelector('tbody');
    body.innerHTML = '<tr><td colspan="3" class="muted">Scoring test videos with the model…</td></tr>';
    try {
        const data = await api(`/api/dataset/${ds}/videos`);
        explorer.videos = data.videos;
        explorer.threshold = data.threshold;
        explorer.loaded = true;
        const cats = [...new Set(data.videos.map((v) => v.category))].sort();
        $('cat-select').innerHTML = '<option value="">All categories</option>' + cats.map((c) => `<option>${c}</option>`).join('');
        if (ds === 'ucf' && cats.includes('Shoplifting')) $('cat-select').value = 'Shoplifting';
        renderVideoList();
    } catch (err) {
        body.innerHTML = `<tr><td colspan="3" class="error">${err.message}</td></tr>`;
    }
}

function renderVideoList() {
    const cat = $('cat-select').value;
    const q = $('search').value.toLowerCase();
    const body = $('videos-table').querySelector('tbody');
    body.innerHTML = '';
    explorer.videos
        .filter((v) => (!cat || v.category === cat) && (!q || v.name.toLowerCase().includes(q)))
        .forEach((v) => {
            const tr = document.createElement('tr');
            if (v.name === explorer.selected) tr.classList.add('selected');
            const alarm = v.max_score >= explorer.threshold;
            tr.innerHTML = `<td class="mono small">${v.name}</td><td>${v.category}</td>
                <td><span class="pill ${alarm ? 'alarm' : 'ok'} mono">${fmt(v.max_score)}</span></td>`;
            tr.addEventListener('click', () => {
                body.querySelectorAll('tr').forEach((r) => r.classList.remove('selected'));
                tr.classList.add('selected');
                showDatasetVideo(v.name);
            });
            body.appendChild(tr);
        });
}

async function showDatasetVideo(name) {
    explorer.selected = name;
    const d = await api(`/api/dataset/${explorer.dataset}/video/${encodeURIComponent(name)}?threshold=${explorer.threshold}`);
    $('detail-empty').classList.add('hidden');
    $('detail').classList.remove('hidden');
    $('d-name').textContent = d.name;
    $('d-cat').textContent = d.category;
    const detected = d.intervals.length > 0;
    const correct = detected === (d.label === 1);
    $('d-verdict').className = `pill ${correct ? 'ok' : 'alarm'}`;
    $('d-verdict').textContent = detected
        ? (d.label ? 'correctly flagged' : 'false alarm')
        : (d.label ? 'missed' : 'correctly normal');
    const peak = Math.max(...d.scores);
    $('d-max').textContent = fmt(peak);
    $('d-video').textContent = fmt(d.video_score);
    $('d-gt').textContent = d.gt_intervals.length ? d.gt_intervals.map((i) => `${i.start_s.toFixed(0)}–${i.end_s.toFixed(0)}s`).join(', ') : 'none';
    $('d-det').textContent = detected ? d.intervals.map((i) => `${i.start_s.toFixed(0)}–${i.end_s.toFixed(0)}s`).join(', ') : 'none';
    scoreChart('explorer-chart', { seconds: secondsFor(d.scores.length, 30), scores: d.scores, gt: d.gt, threshold: explorer.threshold });
}

$('ds-select').addEventListener('change', (e) => { explorer.selected = null; loadDataset(e.target.value); });
$('cat-select').addEventListener('change', renderVideoList);
$('search').addEventListener('input', renderVideoList);

// ------------------------------------------------------------------ model
function metric(label, value, sub) {
    return `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub || ''}</div></div>`;
}

function renderModel() {
    if (!modelInfo) return;
    const ucf = modelInfo.models.ucf;
    const sht = modelInfo.models.sht;
    let html = '';
    if (ucf?.test_results) {
        const t = ucf.test_results;
        html += metric('UCF-Crime frame AUC', pct(t.corrected.auc), `raw snippet scores: ${pct(t.snippet.auc)}`);
        html += metric('UCF-Crime frame AP', pct(t.corrected.ap), 'average precision');
        if (t.corrected.auc_Shoplifting != null) html += metric('Shoplifting vs normal AUC', pct(t.corrected.auc_Shoplifting), 'Shoplifting + normal test videos');
        html += metric('Video-level AUC', pct(t.video_auc), `NaN outputs: ${t.nan_videos}`);
    }
    if (sht?.test_results) {
        html += metric('ShanghaiTech frame AUC', pct(sht.test_results.corrected.auc), `AP ${pct(sht.test_results.corrected.ap)}`);
    }
    $('metrics').innerHTML = html || '<div class="empty">No trained model found. Run <span class="mono">cd MSL && python train.py</span>.</div>';

    const m = ucf || sht;
    if (!m) return;
    $('model-config').textContent = `hidden ${m.config.hidden_dim} · layers ${m.config.num_layers} · heads ${m.config.num_heads} · ${m.parameters.toLocaleString()} parameters · best: stage ${m.best?.stage}, K=${m.best?.k}, epoch ${m.best?.epoch}`;
    charts['train-chart']?.destroy();
    const h = m.history;
    charts['train-chart'] = new Chart($('train-chart'), {
        type: 'line',
        data: {
            labels: h.map((_, i) => i + 1),
            datasets: [
                { label: 'train loss', data: h.map((x) => x.loss), borderColor: '#8b96a5', yAxisID: 'y', pointRadius: 0, borderWidth: 2 },
                { label: 'validation video-AUC', data: h.map((x) => x.val_auc), borderColor: '#5b8def', yAxisID: 'y1', pointRadius: 0, borderWidth: 2 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { title: { display: true, text: 'epoch (all self-training stages)' } },
                y: { position: 'left', title: { display: true, text: 'loss' } },
                y1: { position: 'right', min: 0.5, max: 1, grid: { drawOnChartArea: false }, title: { display: true, text: 'AUC' } },
            },
            plugins: { legend: { labels: { boxWidth: 12 } }, threshold: { value: null },
                tooltip: { callbacks: { title: (it) => { const x = h[it[0].dataIndex]; return `stage ${x.stage} (K=${x.k}) epoch ${x.epoch}`; } } } },
        },
    });
}

// ------------------------------------------------------------------ init
(async () => {
    loadHistory();
    try {
        modelInfo = await api('/api/model');
        $('device-badge').textContent = modelInfo.device === 'cuda' ? 'GPU' : 'CPU';
        threshold = modelInfo.threshold;
        $('threshold').value = threshold;
        $('threshold-value').textContent = threshold.toFixed(2);
    } catch (err) {
        $('device-badge').textContent = 'model unavailable';
    }
})();
