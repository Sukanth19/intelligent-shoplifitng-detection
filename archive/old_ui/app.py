"""
Shoplifting Detection System - Web Interface
Modern Flask application with real-time video analysis
"""

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import threading
import queue

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Global variables for video processing
current_video_path = None
processing_status = {
    'status': 'idle',
    'progress': 0,
    'current_frame': 0,
    'total_frames': 0,
    'detections': [],
    'anomaly_score': 0.0
}
process_queue = queue.Queue()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model(checkpoint_path='checkpoints/best_model.pt'):
    """Load the trained MSL model"""
    try:
        # This will be implemented based on your model architecture
        # For now, return None to allow the app to run
        if os.path.exists(checkpoint_path):
            model = torch.load(checkpoint_path, map_location='cpu')
            return model
        return None
    except Exception as e:
        print(f"Model loading error: {e}")
        return None


def extract_features(video_path):
    """Extract VideoSwin features from video"""
    # Placeholder - integrate with your feature extraction pipeline
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    features = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        processing_status['current_frame'] = frame_count
        processing_status['total_frames'] = total_frames
        processing_status['progress'] = int((frame_count / total_frames) * 100)
        
        # Placeholder for actual feature extraction
        # features.append(extract_videoswin_features(frame))
        
    cap.release()
    return np.random.rand(100, 1024)  # Placeholder features


def detect_anomalies(features, model=None):
    """Run anomaly detection on extracted features"""
    # Placeholder for actual model inference
    # This should use your trained MSL model
    
    # Simulate detection results
    num_segments = len(features) if len(features) > 0 else 100
    anomaly_scores = np.random.rand(num_segments)
    
    # Identify high-confidence anomaly segments
    threshold = 0.7
    detections = []
    
    for idx, score in enumerate(anomaly_scores):
        if score > threshold:
            detections.append({
                'segment': idx,
                'score': float(score),
                'timestamp': f"{idx * 16 / 30:.2f}s",  # Assuming 16-frame segments at 30fps
                'type': 'suspicious_activity'
            })
    
    return {
        'anomaly_scores': anomaly_scores.tolist(),
        'detections': detections,
        'overall_score': float(np.mean(anomaly_scores)),
        'max_score': float(np.max(anomaly_scores)),
        'is_anomalous': np.max(anomaly_scores) > threshold
    }


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file upload"""
    global current_video_path, processing_status
    
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        current_video_path = filepath
        
        # Reset processing status
        processing_status = {
            'status': 'uploaded',
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'detections': [],
            'anomaly_score': 0.0,
            'video_path': filepath,
            'filename': filename
        }
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': filepath
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Start video analysis in background"""
    global processing_status
    
    if not current_video_path or not os.path.exists(current_video_path):
        return jsonify({'error': 'No video uploaded'}), 400
    
    # Start analysis in background thread
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Analysis started'})


def run_analysis():
    """Background task for video analysis"""
    global processing_status
    
    try:
        processing_status['status'] = 'processing'
        
        # Load model
        model = load_model()
        
        # Extract features
        processing_status['status'] = 'extracting_features'
        features = extract_features(current_video_path)
        
        # Detect anomalies
        processing_status['status'] = 'detecting_anomalies'
        results = detect_anomalies(features, model)
        
        # Update status with results
        processing_status.update({
            'status': 'complete',
            'progress': 100,
            'detections': results['detections'],
            'anomaly_score': results['overall_score'],
            'max_score': results['max_score'],
            'is_anomalous': results['is_anomalous'],
            'anomaly_scores': results['anomaly_scores']
        })
        
        # Save results to file
        result_filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"static/results/{result_filename}", 'w') as f:
            json.dump(results, f, indent=2)
        
    except Exception as e:
        processing_status['status'] = 'error'
        processing_status['error'] = str(e)


@app.route('/status')
def get_status():
    """Get current processing status"""
    return jsonify(processing_status)


@app.route('/video_feed')
def video_feed():
    """Stream video with overlay"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    """Generate video frames with detection overlay"""
    global current_video_path
    
    if not current_video_path or not os.path.exists(current_video_path):
        return
    
    cap = cv2.VideoCapture(current_video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add overlay based on detection status
        if processing_status.get('is_anomalous', False):
            cv2.putText(frame, 'ANOMALY DETECTED', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()


@app.route('/history')
def get_history():
    """Get analysis history"""
    results_dir = Path('static/results')
    history = []
    
    for result_file in results_dir.glob('*.json'):
        with open(result_file, 'r') as f:
            data = json.load(f)
            history.append({
                'timestamp': result_file.stem.replace('results_', ''),
                'filename': result_file.name,
                'summary': data
            })
    
    return jsonify(sorted(history, key=lambda x: x['timestamp'], reverse=True))


@app.route('/stats')
def get_stats():
    """Get overall statistics"""
    results_dir = Path('static/results')
    total_analyses = len(list(results_dir.glob('*.json')))
    
    anomalies_detected = 0
    avg_score = 0
    
    for result_file in results_dir.glob('*.json'):
        with open(result_file, 'r') as f:
            data = json.load(f)
            if data.get('is_anomalous', False):
                anomalies_detected += 1
            avg_score += data.get('overall_score', 0)
    
    if total_analyses > 0:
        avg_score /= total_analyses
    
    return jsonify({
        'total_analyses': total_analyses,
        'anomalies_detected': anomalies_detected,
        'avg_anomaly_score': avg_score,
        'detection_rate': (anomalies_detected / total_analyses * 100) if total_analyses > 0 else 0
    })


@app.route('/dataset/<dataset_name>/videos')
def get_dataset_videos(dataset_name):
    """Get list of videos from a dataset"""
    try:
        dataset_paths = {
            'ucf': 'data/UCF-VideoSwin-Features',
            'sht': 'data/SHT-VideoSwin-Features',
            'custom': 'data/custom'
        }
        
        dataset_path = dataset_paths.get(dataset_name)
        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Read video list files
        videos = []
        
        if dataset_name == 'ucf':
            # Read from ucf-train-filter.list
            train_list = os.path.join(dataset_path, 'ucf-train-filter.list')
            test_list = os.path.join(dataset_path, 'ucf-test.list')
            
            categories = ['Shoplifting', 'Stealing', 'Robbery', 'Normal', 'Burglary', 
                         'Arson', 'Assault', 'Fighting', 'Vandalism', 'Abuse']
            
            if os.path.exists(train_list):
                with open(train_list, 'r') as f:
                    for idx, line in enumerate(f):
                        filename = line.strip()
                        if filename:
                            # Extract category from filename
                            category = 'Normal'
                            for cat in categories:
                                if cat in filename:
                                    category = cat
                                    break
                            
                            videos.append({
                                'id': len(videos),
                                'name': filename,
                                'category': category,
                                'duration': '180s',
                                'size': f'{np.random.randint(30, 150)}MB',
                                'analyzed': np.random.random() > 0.7,
                                'split': 'train'
                            })
            
            if os.path.exists(test_list):
                with open(test_list, 'r') as f:
                    for idx, line in enumerate(f):
                        filename = line.strip()
                        if filename:
                            category = 'Normal'
                            for cat in categories:
                                if cat in filename:
                                    category = cat
                                    break
                            
                            videos.append({
                                'id': len(videos),
                                'name': filename,
                                'category': category,
                                'duration': '180s',
                                'size': f'{np.random.randint(30, 150)}MB',
                                'analyzed': np.random.random() > 0.7,
                                'split': 'test'
                            })
        
        elif dataset_name == 'sht':
            # Read from sh-train.list
            train_list = os.path.join(dataset_path, 'sh-train.list')
            test_list = os.path.join(dataset_path, 'sh-test.list')
            
            if os.path.exists(train_list):
                with open(train_list, 'r') as f:
                    for idx, line in enumerate(f):
                        filename = line.strip()
                        if filename:
                            # SHT is all shoplifting videos
                            videos.append({
                                'id': len(videos),
                                'name': filename.replace('.npy', '.mp4'),
                                'category': 'Shoplifting',
                                'duration': '120s',
                                'size': f'{np.random.randint(20, 80)}MB',
                                'analyzed': np.random.random() > 0.6,
                                'split': 'train'
                            })
            
            if os.path.exists(test_list):
                with open(test_list, 'r') as f:
                    for idx, line in enumerate(f):
                        filename = line.strip()
                        if filename:
                            videos.append({
                                'id': len(videos),
                                'name': filename.replace('.npy', '.mp4'),
                                'category': 'Shoplifting',
                                'duration': '120s',
                                'size': f'{np.random.randint(20, 80)}MB',
                                'analyzed': np.random.random() > 0.6,
                                'split': 'test'
                            })
        
        else:
            # Custom dataset - list all video files
            for file in os.listdir(dataset_path):
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    videos.append({
                        'id': len(videos),
                        'name': file,
                        'category': 'unknown',
                        'duration': '120s',
                        'size': '50MB',
                        'analyzed': False,
                        'split': 'custom'
                    })
        
        # Calculate stats
        analyzed_count = sum(1 for v in videos if v.get('analyzed', False))
        anomaly_count = sum(1 for v in videos if v.get('category') in ['Shoplifting', 'Stealing', 'Robbery'])
        
        stats = {
            'total': len(videos),
            'analyzed': analyzed_count,
            'anomalies': anomaly_count,
            'avgTime': '45s'
        }
        
        return jsonify({'videos': videos, 'stats': stats})
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch/upload', methods=['POST'])
def batch_upload():
    """Handle batch video upload"""
    if 'videos' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('videos')
    uploaded = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            uploaded.append({
                'filename': filename,
                'path': filepath
            })
    
    return jsonify({'success': True, 'uploaded': uploaded})


@app.route('/batch/analyze', methods=['POST'])
def batch_analyze():
    """Start batch analysis"""
    data = request.get_json()
    video_ids = data.get('video_ids', [])
    
    # In real implementation, queue these for processing
    return jsonify({'success': True, 'queued': len(video_ids)})


@app.route('/live/cameras')
def get_cameras():
    """Get list of available cameras"""
    cameras = [
        {'id': 1, 'name': 'Camera 1 - Entrance', 'status': 'offline'},
        {'id': 2, 'name': 'Camera 2 - Aisle 1', 'status': 'offline'},
        {'id': 3, 'name': 'Camera 3 - Checkout', 'status': 'offline'},
        {'id': 4, 'name': 'Camera 4 - Exit', 'status': 'offline'},
    ]
    return jsonify({'cameras': cameras})


@app.route('/reports/generate', methods=['POST'])
def generate_report():
    """Generate analysis report"""
    data = request.get_json()
    report_type = data.get('type', 'daily')
    
    # Generate report
    report = {
        'type': report_type,
        'generated_at': datetime.now().isoformat(),
        'summary': 'Report generated successfully'
    }
    
    return jsonify(report)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html',
                         error_code=404,
                         error_message='Page Not Found',
                         error_description='The page you are looking for does not exist.'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html',
                         error_code=500,
                         error_message='Internal Server Error',
                         error_description='Something went wrong on our end. Please try again.'), 500


@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     🛡️  SHOPLIFTING DETECTION SYSTEM - WEB UI  🛡️        ║
    ║     Access at: http://localhost:5000                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
