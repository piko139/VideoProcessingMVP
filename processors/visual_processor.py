# processors/visual_processor.py

import numpy as np
import cv2
import moviepy.editor as mp
from pathlib import Path


class VisualProcessor:
    """Handle video frame extraction and scene analysis"""
    
    def __init__(self, frame_interval=2.0):
        self.frame_interval = frame_interval  # Extract frame every N seconds
        self.scene_change_threshold = 0.3  # Threshold for scene change detection
        
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames at regular intervals from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"Cannot open video: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Calculate frame step
            frame_step = int(fps * self.frame_interval)
            if frame_step == 0:
                frame_step = 1
            
            frames = []
            frame_number = 0
            
            print(f"Extracting frames from {Path(video_path).name} (duration: {duration:.1f}s)")
            
            while frame_number < total_frames and len(frames) < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                timestamp = frame_number / fps
                
                # Resize frame for processing (smaller = faster)
                frame_resized = cv2.resize(frame, (224, 224))
                
                frame_data = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'frame': frame_resized,
                    'original_frame': frame  # Keep original for display
                }
                
                frames.append(frame_data)
                frame_number += frame_step
            
            cap.release()
            print(f"Extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def detect_scene_changes(self, frames):
        """Detect scene changes between consecutive frames"""
        if len(frames) < 2:
            return []
        
        scene_changes = []
        
        for i in range(1, len(frames)):
            # Convert frames to grayscale for comparison
            frame1_gray = cv2.cvtColor(frames[i-1]['frame'], cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(frames[i]['frame'], cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram difference
            hist1 = cv2.calcHist([frame1_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([frame2_gray], [0], None, [256], [0, 256])
            
            # Compare histograms
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # If correlation is low, it's likely a scene change
            if diff < (1 - self.scene_change_threshold):
                scene_change = {
                    'timestamp': frames[i]['timestamp'],
                    'frame_number': frames[i]['frame_number'],
                    'confidence': 1 - diff,
                    'description': f"Scene change detected at {frames[i]['timestamp']:.1f}s"
                }
                scene_changes.append(scene_change)
        
        print(f"Detected {len(scene_changes)} scene changes")
        return scene_changes
    
    def analyze_frame_content(self, frame_data):
        """Basic content analysis of a frame"""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic features
        brightness = np.mean(gray)
        color_variance = np.var(hsv[:,:,1])  # Saturation variance
        edge_density = np.mean(cv2.Canny(gray, 50, 150)) / 255.0
        
        # Motion estimation (if we had previous frame)
        motion_score = 0.0  # Placeholder for motion detection
        
        # Categorize frame content
        description_parts = []
        
        if brightness < 50:
            description_parts.append("dark scene")
        elif brightness > 200:
            description_parts.append("bright scene")
        else:
            description_parts.append("normal lighting")
        
        if color_variance > 2000:
            description_parts.append("colorful content")
        else:
            description_parts.append("muted colors")
        
        if edge_density > 0.1:
            description_parts.append("detailed/complex imagery")
        else:
            description_parts.append("simple/smooth imagery")
        
        description = f"Frame at {timestamp:.1f}s: {', '.join(description_parts)}"
        
        return {
            'timestamp': timestamp,
            'description': description,
            'brightness': brightness,
            'color_variance': color_variance,
            'edge_density': edge_density,
            'features': {
                'brightness': brightness,
                'color_variance': color_variance,
                'edge_density': edge_density
            }
        }
    
    def process_video_visual(self, video_path, video_id):
        """Complete visual processing pipeline for a video"""
        results = {
            'video_id': video_id,
            'video_path': video_path,
            'frames': [],
            'scene_changes': [],
            'frame_descriptions': [],
            'error': None
        }
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path)
            if not frames:
                results['error'] = "Failed to extract frames"
                return results
            
            results['frames'] = frames
            
            # Detect scene changes
            scene_changes = self.detect_scene_changes(frames)
            results['scene_changes'] = scene_changes
            
            # Analyze frame content
            frame_descriptions = []
            for frame_data in frames:
                analysis = self.analyze_frame_content(frame_data)
                frame_descriptions.append(analysis)
            
            results['frame_descriptions'] = frame_descriptions
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error in visual processing pipeline: {e}")
            return results
    
    def display_visual_summary(self, visual_data):
        """Display visual processing results"""
        print(f"Visual Analysis Summary for {Path(visual_data['video_path']).name}")
        print("-" * 50)
        
        if visual_data['error']:
            print(f"Error: {visual_data['error']}")
            return
        
        print(f"Frames extracted: {len(visual_data['frames'])}")
        print(f"Scene changes detected: {len(visual_data['scene_changes'])}")
        
        # Show scene changes
        if visual_data['scene_changes']:
            print("\nScene Changes:")
            for sc in visual_data['scene_changes']:
                print(f"  {sc['timestamp']:.1f}s - {sc['description']}")
        
        # Show sample frame descriptions
        print(f"\nSample Frame Descriptions:")
        for i, desc in enumerate(visual_data['frame_descriptions'][:5]):
            print(f"  {desc['description']}")
        
        if len(visual_data['frame_descriptions']) > 5:
            print(f"  ... and {len(visual_data['frame_descriptions']) - 5} more frames")