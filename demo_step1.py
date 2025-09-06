#!/usr/bin/env python3
"""
Video Highlights Processing Pipeline Demo
Demonstrates the complete video processing workflow from Step 1
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import your modules
from processors.video_processor import VideoFileManager
from processors.audio_processor import AudioProcessor
from processors.visual_processor import VisualProcessor
from llm.highlight_detector import HighlightDetector
from llm.embedding_generator import EmbeddingGenerator
from database.db_manager import DatabaseManager
from database.models import setup_database

def main():
    """Run the complete video processing pipeline"""
    
    print("Video Highlights Processing Pipeline Demo")
    print("=" * 60)
    
    # Configuration
    UPLOADS_DIR = PROJECT_ROOT / "uploads"
    UPLOADS_DIR.mkdir(exist_ok=True)  # Ensure uploads directory exists
    
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://video_user:video_pass@postgres:5432/video_highlights')
    
    # Step 1: Initialize services
    print("\nStep 1: Initializing services...")
    
    # Database setup
    engine = setup_database(DATABASE_URL)
    if not engine:
        print("Failed to connect to database")
        return False
        
    db_manager = DatabaseManager(engine)
    
    # Video processing
    video_manager = VideoFileManager(UPLOADS_DIR)
    audio_processor = AudioProcessor()
    visual_processor = VisualProcessor()
    
    # LLM services
    highlight_detector = HighlightDetector()
    embedding_generator = EmbeddingGenerator()
    
    # Load models
    print("Loading AI models...")
    if not audio_processor.load_whisper_model():
        print("Failed to load Whisper model")
        return False
        
    if not highlight_detector.initialize_llm():
        print("Failed to initialize LLM")
        return False
        
    if not embedding_generator.load_model():
        print("Failed to load embedding model")
        return False
    
    print("All services initialized successfully")
    
    # Step 2: Process videos
    print("\nStep 2: Processing videos...")
    
    video_files = video_manager.get_video_files()
    if not video_files:
        print("No video files found in uploads directory")
        print(f"Please add video files to: {UPLOADS_DIR}")
        return False
    
    print(f"Found {len(video_files)} video files")
    
    all_highlights = []
    successful_videos = 0
    
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        
        try:
            # Extract metadata and register video
            metadata = video_manager.extract_video_metadata(video_path)
            if not metadata:
                print(f"Failed to extract metadata from {video_path.name}")
                continue
                
            video_id = db_manager.register_video(metadata)
            if not video_id:
                print(f"Failed to register {video_path.name} in database")
                continue
            
            # Process audio
            print("Processing audio...")
            audio_result = audio_processor.process_video_audio(video_path, video_id)
            
            # Process visual
            print("Processing visual content...")
            visual_result = visual_processor.process_video_visual(video_path, video_id)
            
            # Create timeline
            print("Creating timeline...")
            timeline = create_timeline(audio_result, visual_result)
            
            # Generate highlights with LLM
            print("Generating highlights with LLM...")
            highlights = highlight_detector.detect_highlights(timeline, video_path.name, video_id)
            
            # Validate highlights before storing
            if highlights:
                validated_highlights = []
                for highlight in highlights:
                    validated = highlight_detector.validate_highlight(highlight, metadata['duration'])
                    if validated:
                        validated_highlights.append(validated)
                
                if validated_highlights:
                    print(f"Generated {len(validated_highlights)} valid highlights")
                    all_highlights.extend(validated_highlights)
                    successful_videos += 1
                else:
                    print("No valid highlights after validation")
            else:
                print("No highlights generated")
                            
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            print("Continuing with next video...")
            continue
    
    if not all_highlights:
        print("No highlights were generated from any videos")
        return False
    
    # Step 3: Generate embeddings and store
    print(f"\nStep 3: Storing {len(all_highlights)} highlights with embeddings...")
    
    stored_count = 0
    
    for highlight in all_highlights:
        try:
            # Generate embedding
            combined_text = f"{highlight['description']} {highlight['summary']}"
            embedding = embedding_generator.generate_embedding(combined_text)
            
            if embedding:
                # Store in database
                db_record = {
                    'video_id': highlight['video_id'],
                    'timestamp_start': highlight['start_time'],
                    'timestamp_end': highlight['end_time'],
                    'description': highlight['description'],
                    'llm_summary': highlight['summary'],
                    'highlight_type': highlight['type'],
                    'confidence_score': highlight['confidence'],
                    'embedding': embedding
                }
                
                highlight_id = db_manager.store_highlight(db_record)
                if highlight_id:
                    stored_count += 1
                else:
                    print("Failed to store highlight")
            else:
                print("Failed to generate embedding for highlight")
                
        except Exception as e:
            print(f"Error storing highlight: {e}")
            continue
    
    print(f"Successfully stored {stored_count} highlights")
    
    # Step 4: Test similarity search
    print("\nStep 4: Testing similarity search...")
    
    test_queries = [
        "magnetic reconnection and science",
        "scene changes and transitions",
        "important speech or dialogue"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        try:
            query_embedding = embedding_generator.generate_embedding(query)
            
            if query_embedding:
                results = db_manager.search_similar_highlights(query_embedding, limit=3)
                
                if results:
                    for result in results:
                        similarity = 1 - result['distance']
                        print(f"  {result['video']} at {result['timestamp']:.1f}s")
                        print(f"     Similarity: {similarity:.1%}")
                        print(f"     {result['description'][:80]}...")
                else:
                    print("  No relevant highlights found")
            else:
                print("  Failed to generate query embedding")
                
        except Exception as e:
            print(f"  Error in similarity search: {e}")
    
    print("\nPipeline demo completed!")
    print(f"Successfully processed {successful_videos}/{len(video_files)} videos")
    print(f"Generated and stored {stored_count} highlights")
    
    return True

def create_timeline(audio_result, visual_result):
    """Combine audio and visual results into a timeline"""
    timeline = []
    
    # Add audio segments
    if audio_result and audio_result.get('transcription'):
        for segment in audio_result['transcription']['segments']:
            timeline.append({
                'timestamp': segment['start'],
                'type': 'speech',
                'content': f"Speech: {segment['text']}",
                'data': segment
            })
    
    # Add visual events
    if visual_result and not visual_result.get('error'):
        # Add frame descriptions
        for frame_desc in visual_result.get('frame_descriptions', []):
            timeline.append({
                'timestamp': frame_desc['timestamp'],
                'type': 'visual',
                'content': frame_desc['description'],
                'data': frame_desc
            })
        
        # Add scene changes
        for scene_change in visual_result.get('scene_changes', []):
            timeline.append({
                'timestamp': scene_change['timestamp'],
                'type': 'scene_change',
                'content': scene_change['description'],
                'data': scene_change
            })
    
    # Sort by timestamp
    timeline.sort(key=lambda x: x['timestamp'])
    return timeline

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)