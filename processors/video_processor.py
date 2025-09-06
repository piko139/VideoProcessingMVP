# processors/video_processor.py

import cv2
import moviepy.editor as mp
from sqlalchemy import text
from pathlib import Path

class VideoFileManager:
    """Handle video file operations and metadata extraction"""
    
    def __init__(self, uploads_dir):
        self.uploads_dir = Path(uploads_dir)
        self.supported_formats = ['.mp4', '.mov', '.avi', '.mkv']
        
    def get_video_files(self):
        """Find all video files in uploads directory"""
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(self.uploads_dir.glob(f'*{ext}'))
        return sorted(video_files)
    
    def extract_video_metadata(self, video_path):
        """Extract basic metadata from video file"""
        try:
            # Load video with OpenCV for basic info
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            resolution = f"{width}x{height}"
            file_size = video_path.stat().st_size
            
            cap.release()
            
            # Use moviepy for additional audio info
            try:
                clip = mp.VideoFileClip(str(video_path))
                has_audio = clip.audio is not None
                audio_duration = clip.duration if clip.audio else 0
                clip.close()
            except Exception as e:
                print(f"Warning: Could not extract audio info: {e}")
                has_audio = False
                audio_duration = 0
            
            metadata = {
                'filename': video_path.name,
                'filepath': str(video_path),
                'duration': duration,
                'fps': fps,
                'resolution': resolution,
                'width': width,
                'height': height,
                'frame_count': frame_count,
                'file_size': file_size,
                'has_audio': has_audio,
                'audio_duration': audio_duration,
                'file_size_mb': round(file_size / (1024 * 1024), 2)
            }
            
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata from {video_path}: {e}")
            return None
    
    def register_video_in_db(self, metadata, engine):
        """Store video metadata in database"""
        try:
            with engine.connect() as conn:
                # Check if video already exists
                result = conn.execute(text("""
                    SELECT id FROM videos WHERE filename = :filename
                """), {'filename': metadata['filename']})
                
                existing = result.fetchone()
                
                if existing:
                    video_id = existing[0]
                    print(f"Video '{metadata['filename']}' already exists with ID: {video_id}")
                    return video_id
                
                # Insert new video
                result = conn.execute(text("""
                    INSERT INTO videos (filename, filepath, duration, fps, resolution, file_size)
                    VALUES (:filename, :filepath, :duration, :fps, :resolution, :file_size)
                    RETURNING id;
                """), {
                    'filename': metadata['filename'],
                    'filepath': metadata['filepath'],
                    'duration': metadata['duration'],
                    'fps': metadata['fps'],
                    'resolution': metadata['resolution'],
                    'file_size': metadata['file_size']
                })
                
                video_id = result.fetchone()[0]
                conn.commit()
                
                print(f"Registered video '{metadata['filename']}' with ID: {video_id}")
                return video_id
                
        except Exception as e:
            print(f"Error registering video in database: {e}")
            return None
    
    def display_video_summary(self, video_files):
        """Display summary of all video files"""
        print(f"Found {len(video_files)} video files:")
        print("-" * 80)
        
        total_size = 0
        total_duration = 0
        
        for video_path in video_files:
            metadata = self.extract_video_metadata(video_path)
            if metadata:
                print(f"File: {metadata['filename']}")
                print(f"  Size: {metadata['file_size_mb']} MB")
                print(f"  Duration: {metadata['duration']:.1f} seconds")
                print(f"  Resolution: {metadata['resolution']} @ {metadata['fps']:.1f} FPS")
                print(f"  Audio: {'Yes' if metadata['has_audio'] else 'No'}")
                print()
                
                total_size += metadata['file_size_mb']
                total_duration += metadata['duration']
        
        print("-" * 80)
        print(f"Total: {total_size:.1f} MB, {total_duration:.1f} seconds")
        
        # Check GitHub size limit
        if total_size > 100:
            print(f"⚠️  Warning: Total size ({total_size:.1f} MB) exceeds GitHub's 100MB limit!")
        else:
            print(f"✅ Total size is within GitHub's 100MB limit")