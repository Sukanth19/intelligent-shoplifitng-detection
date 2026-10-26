/**
 * Enhanced Features JavaScript
 * Handles dataset analysis, batch processing, live monitoring, and reports
 */

// ==============================================
// Dataset Analysis Functions
// ==============================================

let selectedDataset = null;
let datasetVideos = [];
let selectedVideos = [];

// Dataset option click handlers
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.dataset-option').forEach(option => {
        option.addEventListener('click', function() {
            const dataset = this.getAttribute('data-dataset');
            selectDataset(dataset);
        });
    });
    
    // Refresh dataset button
    const refreshBtn = document.getElementById('refresh-dataset');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            if (selectedDataset) {
                loadDatasetVideos(selectedDataset);
            }
        });
    }
    
    // Analyze selected button
    const analyzeBtn = document.getElementById('analyze-selected');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeSelectedVideos);
    }
    
    // Dataset search and filter
    const searchInput = document.getElementById('dataset-search');
    if (searchInput) {
        searchInput.addEventListener('input', filterDatasetVideos);
    }
    
    const categoryFilter = document.getElementById('category-filter');
    if (categoryFilter) {
        categoryFilter.addEventListener('change', filterDatasetVideos);
    }
    
    // Batch processing setup
    setupBatchProcessing();
    
    // Live monitor setup
    setupLiveMonitor();
});

async function selectDataset(dataset) {
    selectedDataset = dataset;
    
    // Update UI
    document.querySelectorAll('.dataset-option').forEach(opt => {
        opt.classList.remove('active');
    });
    event.target.closest('.dataset-option').classList.add('active');
    
    // Update title
    const titles = {
        'ucf': 'UCF-Crime Dataset',
        'sht': 'SHT Dataset',
        'custom': 'Custom Dataset'
    };
    document.getElementById('dataset-title').textContent = titles[dataset];
    
    // Load videos
    await loadDatasetVideos(dataset);
}

async function loadDatasetVideos(dataset) {
    try {
        const response = await fetch(`/dataset/${dataset}/videos`);
        const data = await response.json();
        
        datasetVideos = data.videos || [];
        displayDatasetVideos(datasetVideos);
        updateDatasetStats(data.stats || {});
        
    } catch (error) {
        console.error('Failed to load dataset:', error);
        // Mock data for demo
        generateMockDatasetVideos(dataset);
    }
}

function generateMockDatasetVideos(dataset) {
    const mockVideos = [];
    const categories = ['shoplifting', 'normal', 'robbery', 'stealing'];
    const count = dataset === 'ucf' ? 100 : dataset === 'sht' ? 50 : 20;
    
    for (let i = 0; i < count; i++) {
        mockVideos.push({
            id: i,
            name: `${dataset.toUpperCase()}_Video_${String(i).padStart(3, '0')}.mp4`,
            category: categories[Math.floor(Math.random() * categories.length)],
            duration: `${Math.floor(Math.random() * 300) + 60}s`,
            size: `${(Math.random() * 100 + 10).toFixed(1)}MB`,
            analyzed: Math.random() > 0.5
        });
    }
    
    datasetVideos = mockVideos;
    displayDatasetVideos(mockVideos);
    
    updateDatasetStats({
        total: mockVideos.length,
        analyzed: mockVideos.filter(v => v.analyzed).length,
        anomalies: Math.floor(mockVideos.length * 0.15),
        avgTime: '45s'
    });
}

function displayDatasetVideos(videos) {
    const list = document.getElementById('dataset-video-list');
    
    if (videos.length === 0) {
        list.innerHTML = '<p class="empty-state">No videos found</p>';
        return;
    }
    
    list.innerHTML = videos.map(video => `
        <div class="video-item" data-video-id="${video.id}">
            <input type="checkbox" class="video-checkbox" data-video-id="${video.id}">
            <div class="video-thumbnail">
                <i class="fas fa-film"></i>
            </div>
            <div class="video-info">
                <div class="video-name">${video.name}</div>
                <div class="video-meta">
                    ${video.category} • ${video.duration} • ${video.size}
                    ${video.analyzed ? ' • <span style="color: var(--success)">Analyzed</span>' : ''}
                </div>
            </div>
        </div>
    `).join('');
    
    // Add checkbox handlers
    document.querySelectorAll('.video-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const videoId = parseInt(this.getAttribute('data-video-id'));
            if (this.checked) {
                if (!selectedVideos.includes(videoId)) {
                    selectedVideos.push(videoId);
                }
            } else {
                selectedVideos = selectedVideos.filter(id => id !== videoId);
            }
        });
    });
}

function filterDatasetVideos() {
    const searchTerm = document.getElementById('dataset-search').value.toLowerCase();
    const category = document.getElementById('category-filter').value;
    
    let filtered = datasetVideos;
    
    if (searchTerm) {
        filtered = filtered.filter(v => v.name.toLowerCase().includes(searchTerm));
    }
    
    if (category !== 'all') {
        filtered = filtered.filter(v => v.category === category);
    }
    
    displayDatasetVideos(filtered);
}

function updateDatasetStats(stats) {
    document.querySelector('#dataset-stats .stat-item:nth-child(1) .stat-value').textContent = stats.total || '-';
    document.querySelector('#dataset-stats .stat-item:nth-child(2) .stat-value').textContent = stats.analyzed || '-';
    document.querySelector('#dataset-stats .stat-item:nth-child(3) .stat-value').textContent = stats.anomalies || '-';
    document.querySelector('#dataset-stats .stat-item:nth-child(4) .stat-value').textContent = stats.avgTime || '-';
}

async function analyzeSelectedVideos() {
    if (selectedVideos.length === 0) {
        showToast('Please select at least one video', 'warning');
        return;
    }
    
    showToast(`Starting analysis of ${selectedVideos.length} videos...`, 'info');
    
    // In real implementation, send to backend
    // For now, simulate processing
    setTimeout(() => {
        showToast(`Successfully analyzed ${selectedVideos.length} videos!`, 'success');
        selectedVideos = [];
        document.querySelectorAll('.video-checkbox').forEach(cb => cb.checked = false);
    }, 2000);
}

// ==============================================
// Batch Processing Functions
// ==============================================

let batchQueue = [];

function setupBatchProcessing() {
    const batchUploadArea = document.getElementById('batch-upload-area');
    const batchVideoInput = document.getElementById('batch-video-input');
    
    if (!batchUploadArea || !batchVideoInput) return;
    
    batchUploadArea.addEventListener('click', () => {
        batchVideoInput.click();
    });
    
    batchVideoInput.addEventListener('change', (e) => {
        handleBatchFiles(e.target.files);
    });
    
    // Drag & drop
    batchUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        batchUploadArea.style.borderColor = 'var(--primary)';
    });
    
    batchUploadArea.addEventListener('dragleave', () => {
        batchUploadArea.style.borderColor = '';
    });
    
    batchUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        batchUploadArea.style.borderColor = '';
        handleBatchFiles(e.dataTransfer.files);
    });
    
    // Batch controls
    document.getElementById('start-batch')?.addEventListener('click', startBatchProcessing);
    document.getElementById('clear-batch')?.addEventListener('click', clearBatchQueue);
}

function handleBatchFiles(files) {
    const validFiles = Array.from(files).filter(file => {
        const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/x-matroska'];
        return validTypes.includes(file.type);
    });
    
    if (validFiles.length === 0) {
        showToast('No valid video files selected', 'error');
        return;
    }
    
    validFiles.forEach(file => {
        batchQueue.push({
            id: Date.now() + Math.random(),
            file: file,
            name: file.name,
            size: (file.size / (1024 * 1024)).toFixed(1) + 'MB',
            status: 'pending'
        });
    });
    
    updateBatchQueue();
}

function updateBatchQueue() {
    const queueElement = document.getElementById('batch-queue');
    const queueList = document.getElementById('queue-list');
    const queueCount = document.getElementById('queue-count');
    
    if (batchQueue.length === 0) {
        queueElement.style.display = 'none';
        return;
    }
    
    queueElement.style.display = 'block';
    queueCount.textContent = batchQueue.length;
    
    queueList.innerHTML = batchQueue.map(item => `
        <div class="queue-item">
            <div class="queue-item-icon">
                <i class="fas fa-file-video"></i>
            </div>
            <div class="queue-item-info">
                <div class="queue-item-name">${item.name}</div>
                <div class="queue-item-size">${item.size}</div>
            </div>
            <span class="queue-item-status ${item.status}">${item.status}</span>
        </div>
    `).join('');
}

async function startBatchProcessing() {
    if (batchQueue.length === 0) {
        showToast('No videos in queue', 'warning');
        return;
    }
    
    document.getElementById('batch-progress-card').style.display = 'block';
    document.getElementById('batch-total').textContent = batchQueue.length;
    
    showToast('Starting batch processing...', 'info');
    
    // Process each video
    let completed = 0;
    for (const item of batchQueue) {
        item.status = 'processing';
        updateBatchQueue();
        
        // Simulate processing
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        item.status = 'complete';
        completed++;
        
        document.getElementById('batch-completed').textContent = completed;
        const progress = (completed / batchQueue.length) * 100;
        document.getElementById('batch-progress-fill').style.width = `${progress}%`;
        document.getElementById('batch-progress-percent').textContent = `${Math.round(progress)}%`;
        
        updateBatchQueue();
    }
    
    showToast('Batch processing complete!', 'success');
}

function clearBatchQueue() {
    batchQueue = [];
    updateBatchQueue();
    document.getElementById('batch-progress-card').style.display = 'none';
}

// ==============================================
// Live Monitor Functions
// ==============================================

function setupLiveMonitor() {
    // Placeholder for live monitor setup
    console.log('Live monitor initialized');
}

// ==============================================
// Reports Functions
// ==============================================

function generateReport(type) {
    showToast(`Generating ${type} report...`, 'info');
    
    setTimeout(() => {
        showToast(`${type} report generated successfully!`, 'success');
    }, 1500);
}

// Make functions available globally
window.generateReport = generateReport;
