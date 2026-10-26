#!/usr/bin/env python3
"""
Demo Mode for Shoplifting Detection UI
Simulates video analysis without requiring a trained model
Perfect for UI testing and demonstrations
"""

import numpy as np
import time
from pathlib import Path
import json
from datetime import datetime

class DemoAnalyzer:
    """Simulates video analysis for demo purposes"""
    
    def __init__(self):
        self.demo_scenarios = {
            'normal': {
                'base_score': 0.2,
                'variance': 0.1,
                'anomaly_probability': 0.1
            },
            'suspicious': {
                'base_score': 0.5,
                'variance': 0.2,
                'anomaly_probability': 0.4
            },
            'shoplifting': {
                'base_score': 0.8,
                'variance': 0.15,
                'anomaly_probability': 0.8
            }
        }
    
    def analyze_video(self, video_path, scenario='random'):
        """
        Simulate video analysis
        
        Args:
            video_path: Path to video file
            scenario: 'normal', 'suspicious', 'shoplifting', or 'random'
        
        Returns:
            Dictionary with analysis results
        """
        # Determine scenario
        if scenario == 'random':
            scenario = np.random.choice(['normal', 'suspicious', 'shoplifting'], 
                                       p=[0.6, 0.3, 0.1])
        
        config = self.demo_scenarios.get(scenario, self.demo_scenarios['normal'])
        
        # Simulate video properties
        num_segments = np.random.randint(80, 120)
        duration = num_segments * 16 / 30  # Assuming 16-frame segments at 30fps
        
        # Generate anomaly scores
        scores = self._generate_scores(num_segments, config)
        
        # Identify detections
        detections = self._identify_detections(scores, num_segments)
        
        # Calculate overall metrics
        overall_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        is_anomalous = max_score > 0.7
        
        return {
            'scenario': scenario,
            'num_segments': num_segments,
            'duration': duration,
            'anomaly_scores': scores.tolist(),
            'detections': detections,
            'overall_score': overall_score,
            'max_score': max_score,
            'is_anomalous': is_anomalous,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
    
    def _generate_scores(self, num_segments, config):
        """Generate realistic anomaly scores"""
        base = config['base_score']
        variance = config['variance']
        
        # Generate base scores with some smoothness
        scores = np.random.normal(base, variance, num_segments)
        scores = np.clip(scores, 0, 1)
        
        # Add some temporal smoothing
        window = 5
        scores = np.convolve(scores, np.ones(window)/window, mode='same')
        
        # Add occasional spikes for anomalies
        if np.random.random() < config['anomaly_probability']:
            num_spikes = np.random.randint(1, 4)
            spike_positions = np.random.choice(num_segments, num_spikes, replace=False)
            
            for pos in spike_positions:
                spike_width = np.random.randint(3, 8)
                spike_height = np.random.uniform(0.7, 0.95)
                
                for i in range(spike_width):
                    if pos + i < num_segments:
                        scores[pos + i] = max(scores[pos + i], 
                                             spike_height * np.exp(-0.3 * i))
        
        return scores
    
    def _identify_detections(self, scores, num_segments):
        """Identify segments with high anomaly scores"""
        threshold = 0.7
        detections = []
        
        for idx, score in enumerate(scores):
            if score > threshold:
                timestamp = idx * 16 / 30  # Assuming 16-frame segments at 30fps
                
                # Determine detection type based on score
                if score > 0.85:
                    detection_type = 'suspected_shoplifting'
                elif score > 0.75:
                    detection_type = 'suspicious_behavior'
                else:
                    detection_type = 'unusual_activity'
                
                detections.append({
                    'segment': idx,
                    'score': float(score),
                    'timestamp': f"{timestamp:.2f}s",
                    'type': detection_type
                })
        
        return detections
    
    def simulate_processing(self, video_path, callback=None):
        """
        Simulate video processing with progress updates
        
        Args:
            video_path: Path to video file
            callback: Function to call with progress updates
        """
        num_frames = np.random.randint(800, 1200)
        
        for frame in range(num_frames):
            time.sleep(0.01)  # Simulate processing time
            
            if callback:
                callback({
                    'current_frame': frame + 1,
                    'total_frames': num_frames,
                    'progress': int((frame + 1) / num_frames * 100),
                    'status': 'processing'
                })
        
        # Final analysis
        results = self.analyze_video(video_path)
        
        if callback:
            callback({
                'status': 'complete',
                'progress': 100,
                **results
            })
        
        return results
    
    def save_demo_results(self, results, output_dir='static/results'):
        """Save demo results to file"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filename = f"demo_results_{results['timestamp']}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return str(filepath)


def create_demo_videos():
    """Create demo video metadata"""
    demo_videos = [
        {
            'name': 'Normal_Shopping_Activity.mp4',
            'scenario': 'normal',
            'description': 'Regular customer browsing and purchasing items'
        },
        {
            'name': 'Suspicious_Behavior.mp4',
            'scenario': 'suspicious',
            'description': 'Customer showing unusual behavior patterns'
        },
        {
            'name': 'Detected_Shoplifting.mp4',
            'scenario': 'shoplifting',
            'description': 'Clear shoplifting incident detected'
        }
    ]
    
    return demo_videos


def run_demo_analysis():
    """Run a demo analysis and display results"""
    print("\n🎬 Running Demo Analysis...\n")
    
    analyzer = DemoAnalyzer()
    
    # Test different scenarios
    scenarios = ['normal', 'suspicious', 'shoplifting']
    
    for scenario in scenarios:
        print(f"\n📹 Analyzing {scenario.upper()} scenario...")
        
        results = analyzer.analyze_video('demo_video.mp4', scenario=scenario)
        
        print(f"  ✓ Duration: {results['duration']:.1f}s")
        print(f"  ✓ Segments: {results['num_segments']}")
        print(f"  ✓ Overall Score: {results['overall_score']:.2f}")
        print(f"  ✓ Max Score: {results['max_score']:.2f}")
        print(f"  ✓ Anomalous: {'YES' if results['is_anomalous'] else 'NO'}")
        print(f"  ✓ Detections: {len(results['detections'])}")
        
        # Save results
        filepath = analyzer.save_demo_results(results)
        print(f"  ✓ Saved to: {filepath}")
    
    print("\n✅ Demo analysis complete!\n")


if __name__ == '__main__':
    # Run demo
    run_demo_analysis()
    
    # Show available demo videos
    print("📼 Available Demo Scenarios:")
    for video in create_demo_videos():
        print(f"  • {video['name']}: {video['description']}")
