#!/usr/bin/env python3
"""
Simple YouTube Video Processor for Soccer Player Recognition

Processes video without complex dependencies to demonstrate functionality.
"""

import cv2
import numpy as np
import os
import time
import json
from pathlib import Path

class SimpleSoccerProcessor:
    """Simple processor for soccer video analysis."""
    
    def __init__(self):
        self.frame_count = 0
        self.player_detections = []
        self.ball_detections = []
        
    def process_video(self, video_path):
        """Process the YouTube video and show analysis."""
        
        print("🎬 Soccer Player Recognition - YouTube Video Analysis")
        print("=" * 60)
        print(f"📹 Processing: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video")
            return
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video Information:")
        print(f"   • Resolution: {width}x{height}")
        print(f"   • FPS: {fps:.2f}")
        print(f"   • Total Frames: {total_frames}")
        print(f"   • Duration: {total_frames/fps:.2f} seconds")
        print()
        
        # Create output directory
        output_dir = "outputs/youtube_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process every 30th frame (1 second intervals for 30fps video)
        sample_interval = max(1, int(fps))
        
        processing_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process sample frames
            if self.frame_count % sample_interval == 0:
                start_time = time.time()
                
                # Analyze frame
                analysis = self.analyze_frame(frame)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Create annotated frame
                annotated = self.create_annotated_frame(frame, analysis)
                
                # Save annotated frame
                output_path = f"{output_dir}/frame_{self.frame_count:04d}.jpg"
                cv2.imwrite(output_path, annotated)
                
                print(f"🔄 Frame {self.frame_count:4d}/{total_frames} | "
                      f"Players: {analysis['players']} | Ball: {analysis['ball']} | "
                      f"Time: {processing_time:.3f}s | Saved: {os.path.basename(output_path)}")
                
            self.frame_count += 1
            
        cap.release()
        
        # Generate final report
        self.generate_report(processing_times, total_frames, fps, output_dir)
        
        print(f"\n🎉 Processing Complete!")
        print(f"📁 Results saved to: {output_dir}")
        
    def analyze_frame(self, frame):
        """Analyze frame for soccer elements (simulated AI detection)."""
        
        # Simulate AI model detection
        # In real system, this would call RF-DETR, SAM2, SigLIP, ResNet
        
        height, width = frame.shape[:2]
        
        # Simulate player detection using color and motion analysis
        players_detected = 0
        ball_detected = False
        
        # Simple green field detection (soccer field typically green)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (width * height)
        
        # If significant green detected, simulate player detection
        if green_ratio > 0.3:
            # Simulate finding players based on non-green regions
            players_detected = np.random.randint(2, 8)  # 2-8 players typical
            ball_detected = np.random.random() > 0.3  # 70% chance ball visible
            
        return {
            'players': players_detected,
            'ball': ball_detected,
            'field_ratio': green_ratio,
            'frame_size': (width, height)
        }
        
    def create_annotated_frame(self, frame, analysis):
        """Create annotated visualization frame."""
        
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        
        # Add analysis overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Add text information
        cv2.putText(annotated, f"Frame: {self.frame_count}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Players: {analysis['players']}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if analysis['players'] > 0 else (0, 0, 255), 2)
        cv2.putText(annotated, f"Ball: {'Yes' if analysis['ball'] else 'No'}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if analysis['ball'] else (0, 0, 255), 2)
        cv2.putText(annotated, f"Field: {analysis['field_ratio']:.1%}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Simulate player bounding boxes
        if analysis['players'] > 0:
            for i in range(analysis['players']):
                # Random but realistic player positions
                x = np.random.randint(50, width - 100)
                y = np.random.randint(100, height - 150)
                w = np.random.randint(30, 60)
                h = np.random.randint(60, 120)
                
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, f"P{i+1}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Simulate ball detection
        if analysis['ball']:
            bx = np.random.randint(100, width - 50)
            by = np.random.randint(50, height - 50)
            cv2.circle(annotated, (bx, by), 8, (0, 0, 255), -1)
            cv2.circle(annotated, (bx, by), 8, (255, 255, 255), 2)
            
        return annotated
        
    def generate_report(self, processing_times, total_frames, fps, output_dir):
        """Generate analysis report."""
        
        print("\n📊 GENERATING ANALYSIS REPORT")
        print("=" * 40)
        
        # Calculate statistics
        frames_processed = len(processing_times)
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        fps_processing = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        report = {
            'video_analysis': {
                'total_frames': total_frames,
                'frames_processed': frames_processed,
                'processing_fps': fps_processing,
                'avg_processing_time': avg_processing_time
            },
            'detection_summary': {
                'total_players_detected': sum(self.player_detections),
                'total_ball_detections': sum(self.ball_detections),
                'detection_rate': frames_processed / total_frames if total_frames > 0 else 0
            },
            'performance_metrics': {
                'min_processing_time': min(processing_times) if processing_times else 0,
                'max_processing_time': max(processing_times) if processing_times else 0,
                'std_processing_time': np.std(processing_times) if processing_times else 0
            }
        }
        
        # Save report
        report_path = f"{output_dir}/analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"🎯 ANALYSIS RESULTS:")
        print(f"   📹 Total Frames: {total_frames}")
        print(f"   🔍 Frames Processed: {frames_processed}")
        print(f"   ⚡ Processing Speed: {fps_processing:.2f} FPS")
        print(f"   ⏱️  Avg Processing Time: {avg_processing_time:.3f}s")
        print(f"   📊 Report Saved: {report_path}")

def main():
    """Main function."""
    
    video_path = "soccer_match_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file {video_path} not found!")
        return
        
    processor = SimpleSoccerProcessor()
    processor.process_video(video_path)

if __name__ == "__main__":
    main()