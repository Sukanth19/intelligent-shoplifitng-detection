/**
 * Shoplifting Detection System - Frontend JavaScript
 * Handles all UI interactions, video upload, and real-time updates
 */

// ==============================================
// Global State
// ==============================================

let currentVideo = null;
let analysisInterval = null;
let statsInterval = null;

// ==============================================
// Navigation
// ==============================================

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
    }
    
    // Activate corresponding nav link
    const targetLink = document.querySelector(`[href="#${sectionId}"]`);
    if (targetLink) {
        targetLink.classList.add('active');
    }
    
    // Load section-specific data
    if (sectionId === 'dashboard') {
        updateDashboardStats();
        loadRecentActivity();
    } else if (sectionId === 'history') {
        loadHistory();
    }
}

// Navigation click handlers
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = link.getAttribute('href').substring(1);
            showSection(sectionId);
        });
    });
    
    // Initialize
    showSection('dashboard');
    setupVideoUpload();
    setupSettings();
    
    // Start periodic stats update
    statsInterval = setInterval(updateDashboardStats, 30000); // Every 30 seconds
});

// ==============================================
// Dashboard Functions
// ==============================================

async function updateDashboardStats() {
    try {
        const response = await fetch('/stats');
        const stats = await response.json();
        
        // Update stat cards with animation
        animateValue('total-analyses', parseInt(document.getElementById('total-analyses').textContent), stats.total_analyses);
        animateValue('anomalies-detected', parseInt(document.getElementById('anomalies-detected').textContent), stats.anomalies_detected);
        
        document.getElementById('detection-rate').textContent = `${stats.detection_rate.toFixed(1)}%`;
        document.getElementById('avg-score').textContent = stats.avg_anomaly_score.toFixed(2);
        
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

function animateValue(elementId, start, end, duration = 1000) {
    const element = document.getElementById(elementId);
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

async function loadRecentActivity() {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        
        const activityList = document.getElementById('recent-activity');
        
        if (history.length === 0) {
            activityList.innerHTML = '<p class="empty-state">No recent activity</p>';
            return;
        }
        
        activityList.innerHTML = history.slice(0, 5).map(item => {
            const isAnomalous = item.summary.is_anomalous;
            const iconClass = isAnomalous ? 'danger' : 'success';
            const icon = isAnomalous ? 'fa-exclamation-triangle' : 'fa-check-circle';
            const status = isAnomalous ? 'Anomaly Detected' : 'Normal Activity';
            
            return `
                <div class="activity-item">
                    <div class="activity-icon ${iconClass}">
                        <i class="fas ${icon}"></i>
                    </div>
                    <div class="activity-content">
                        <div class="activity-title">${status}</div>
                        <div class="activity-time">${formatTimestamp(item.timestamp)}</div>
                    </div>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        console.error('Failed to load activity:', error);
    }
}

// ==============================================
// Video Upload & Analysis
// ==============================================

function setupVideoUpload() {
    const uploadArea = document.getElementById('upload-area');
    const videoInput = document.getElementById('video-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    // Click to browse
    uploadArea.addEventListener('click', () => {
        videoInput.click();
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary)';
        uploadArea.style.background = 'rgba(99, 102, 241, 0.05)';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleVideoFile(files[0]);
        }
    });
    
    // File input change
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoFile(e.target.files[0]);
        }
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', startAnalysis);
}

async function handleVideoFile(file) {
    // Validate file
    const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/x-matroska'];
    if (!validTypes.includes(file.type)) {
        showToast('Invalid file type. Please upload MP4, AVI, MOV, or MKV.', 'error');
        return;
    }
    
    if (file.size > 500 * 1024 * 1024) {
        showToast('File too large. Maximum size is 500MB.', 'error');
        return;
    }
    
    // Upload file
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        showToast('Uploading video...', 'info');
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentVideo = result.filename;
            
            // Update UI
            document.getElementById('upload-area').style.display = 'none';
            document.getElementById('upload-info').style.display = 'block';
            document.getElementById('filename').textContent = file.name;
            
            // Show video preview
            displayVideoPreview(file);
            
            showToast('Video uploaded successfully!', 'success');
        } else {
            showToast(result.error || 'Upload failed', 'error');
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        showToast('Upload failed. Please try again.', 'error');
    }
}

function displayVideoPreview(file) {
    const videoContainer = document.getElementById('video-container');
    const url = URL.createObjectURL(file);
    
    videoContainer.innerHTML = `
        <video controls>
            <source src="${url}" type="${file.type}">
            Your browser does not support the video tag.
        </video>
    `;
}

async function startAnalysis() {
    if (!currentVideo) {
        showToast('Please upload a video first.', 'error');
        return;
    }
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Show progress
            document.getElementById('progress-container').style.display = 'block';
            document.getElementById('analyze-btn').disabled = true;
            
            // Start polling for status
            analysisInterval = setInterval(updateAnalysisStatus, 1000);
            
            showToast('Analysis started...', 'info');
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        showToast('Failed to start analysis.', 'error');
    }
}

async function updateAnalysisStatus() {
    try {
        const response = await fetch('/status');
        const status = await response.json();
        
        // Update progress bar
        document.getElementById('progress-percent').textContent = `${status.progress}%`;
        document.getElementById('progress-fill').style.width = `${status.progress}%`;
        document.getElementById('progress-status').textContent = status.status.replace('_', ' ').toUpperCase();
        document.getElementById('progress-details').textContent = 
            `Frame ${status.current_frame} of ${status.total_frames}`;
        
        // Check if complete
        if (status.status === 'complete') {
            clearInterval(analysisInterval);
            displayResults(status);
            document.getElementById('analyze-btn').disabled = false;
            showToast('Analysis complete!', 'success');
        } else if (status.status === 'error') {
            clearInterval(analysisInterval);
            document.getElementById('analyze-btn').disabled = false;
            showToast('Analysis failed: ' + status.error, 'error');
        }
        
    } catch (error) {
        console.error('Status check error:', error);
    }
}

function displayResults(status) {
    const resultsCard = document.getElementById('results-card');
    resultsCard.style.display = 'block';
    
    // Update result status
    const resultStatus = document.getElementById('result-status');
    if (status.is_anomalous) {
        resultStatus.className = 'result-status anomalous';
        resultStatus.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>Anomaly Detected - Potential Shoplifting</span>
        `;
    } else {
        resultStatus.className = 'result-status normal';
        resultStatus.innerHTML = `
            <i class="fas fa-shield-alt"></i>
            <span>No Anomalies Detected - Normal Activity</span>
        `;
    }
    
    // Update score circle
    const scoreValue = Math.round(status.anomaly_score * 100);
    document.getElementById('anomaly-score').textContent = scoreValue;
    
    const circle = document.getElementById('circle-progress');
    const circumference = 283;
    const offset = circumference - (scoreValue / 100) * circumference;
    circle.style.strokeDashoffset = offset;
    
    if (scoreValue > 70) {
        circle.classList.add('danger');
    }
    
    // Display timeline and detections
    displayTimeline(status.anomaly_scores || []);
    displayDetectionList(status.detections || []);
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth' });
}

function displayTimeline(scores) {
    const timelineChart = document.getElementById('timeline-chart');
    timelineChart.innerHTML = '';
    
    scores.forEach((score, index) => {
        if (score > 0.7) {
            const segment = document.createElement('div');
            segment.className = 'timeline-segment';
            segment.style.left = `${(index / scores.length) * 100}%`;
            segment.style.width = `${(1 / scores.length) * 100}%`;
            segment.title = `Segment ${index}: ${(score * 100).toFixed(1)}%`;
            timelineChart.appendChild(segment);
        }
    });
}

function displayDetectionList(detections) {
    const detectionList = document.getElementById('detection-list');
    
    if (detections.length === 0) {
        detectionList.innerHTML = '<p class="empty-state">No anomalies detected</p>';
        return;
    }
    
    detectionList.innerHTML = detections.map(detection => `
        <div class="detection-item">
            <div class="detection-info">
                <div class="detection-time">@ ${detection.timestamp}</div>
                <div class="detection-type">${detection.type.replace('_', ' ')}</div>
            </div>
            <div class="detection-score">${(detection.score * 100).toFixed(0)}%</div>
        </div>
    `).join('');
}

// ==============================================
// History Functions
// ==============================================

async function loadHistory() {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        
        const historyList = document.getElementById('history-list');
        
        if (history.length === 0) {
            historyList.innerHTML = '<p class="empty-state">No analysis history yet</p>';
            return;
        }
        
        historyList.innerHTML = history.map(item => {
            const isAnomalous = item.summary.is_anomalous;
            const scoreClass = isAnomalous ? 'anomalous' : 'normal';
            const score = Math.round(item.summary.overall_score * 100);
            const label = isAnomalous ? 'Anomaly' : 'Normal';
            
            return `
                <div class="history-item">
                    <div class="history-info">
                        <div class="history-filename">Analysis #${item.timestamp}</div>
                        <div class="history-timestamp">${formatTimestamp(item.timestamp)}</div>
                    </div>
                    <div class="history-result">
                        <div class="history-score ${scoreClass}">${score}%</div>
                        <div class="history-label">${label}</div>
                    </div>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// History filters
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('history-search');
    const filterSelect = document.getElementById('history-filter');
    
    if (searchInput) {
        searchInput.addEventListener('input', filterHistory);
    }
    
    if (filterSelect) {
        filterSelect.addEventListener('change', filterHistory);
    }
});

function filterHistory() {
    // Implementation for filtering history
    loadHistory(); // Reload for now
}

// ==============================================
// Settings Functions
// ==============================================

function setupSettings() {
    // Threshold slider
    const thresholdSlider = document.getElementById('threshold');
    const thresholdValue = document.getElementById('threshold-value');
    
    if (thresholdSlider && thresholdValue) {
        thresholdSlider.addEventListener('input', (e) => {
            thresholdValue.textContent = `${e.target.value}%`;
        });
    }
}

// ==============================================
// Utility Functions
// ==============================================

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function formatTimestamp(timestamp) {
    // Format: YYYYMMDD_HHMMSS -> readable format
    const year = timestamp.substring(0, 4);
    const month = timestamp.substring(4, 6);
    const day = timestamp.substring(6, 8);
    const hour = timestamp.substring(9, 11);
    const minute = timestamp.substring(11, 13);
    
    return `${month}/${day}/${year} ${hour}:${minute}`;
}

// ==============================================
// Cleanup on page unload
// ==============================================

window.addEventListener('beforeunload', () => {
    if (analysisInterval) {
        clearInterval(analysisInterval);
    }
    if (statsInterval) {
        clearInterval(statsInterval);
    }
});
