# processors/audio_processor.py

from pathlib import Path
import moviepy.editor as mp
import librosa
import whisper


class AudioProcessor:
    """Handle audio extraction and speech-to-text conversion"""
    
    def __init__(self):
        self.whisper_model = None
        self.sample_rate = 16000  # Whisper's expected sample rate
        
    def load_whisper_model(self, model_size="base"):
        """Load Whisper model for speech-to-text"""
        try:
            print(f"Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size)
            print(f"Whisper model loaded successfully on device: {self.whisper_model.device}")
            return True
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return False
    
    def extract_audio_from_video(self, video_path, output_path=None):
        """Extract audio track from video file"""
        try:
            video_path = Path(video_path)
            
            if output_path is None:
                output_path = video_path.parent / f"{video_path.stem}_audio.wav"
            
            print(f"Extracting audio from: {video_path.name}")
            
            # Load video and extract audio
            clip = mp.VideoFileClip(str(video_path))
            
            if clip.audio is None:
                print(f"Warning: No audio track found in {video_path.name}")
                clip.close()
                return None
            
            # Write audio to file
            clip.audio.write_audiofile(str(output_path), 
                                     fps=self.sample_rate,
                                     verbose=False,
                                     logger=None)
            clip.close()
            
            print(f"Audio extracted to: {output_path.name}")
            return output_path
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def transcribe_audio(self, audio_path, video_name="unknown"):
        """Transcribe audio to text with timestamps"""
        if self.whisper_model is None:
            print("Whisper model not loaded. Please run load_whisper_model() first.")
            return None
        
        try:
            print(f"Transcribing audio: {Path(audio_path).name}")
            
            # Transcribe with Whisper (without word_timestamps for compatibility)
            result = self.whisper_model.transcribe(
                str(audio_path),
                verbose=False
            )
            
            # Process segments
            segments = []
            for segment in result['segments']:
                segment_data = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'confidence': segment.get('avg_logprob', 0.0),
                    'video_name': video_name
                }
                segments.append(segment_data)
            
            print(f"Transcription completed: {len(segments)} segments found")
            return {
                'full_text': result['text'],
                'segments': segments,
                'language': result.get('language', 'unknown')
            }
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    def display_transcription(self, transcription_data):
        """Display transcription results in a readable format"""
        if not transcription_data:
            print("No transcription data to display")
            return
        
        print("TRANSCRIPTION RESULTS")
        print("=" * 60)
        print(f"Language: {transcription_data['language']}")
        print(f"Full text: {transcription_data['full_text'][:200]}...")
        print("\nSegments with timestamps:")
        print("-" * 60)
        
        for i, segment in enumerate(transcription_data['segments'][:10]):  # Show first 10 segments
            print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}")
        
        if len(transcription_data['segments']) > 10:
            print(f"... and {len(transcription_data['segments']) - 10} more segments")
    
    def process_video_audio(self, video_path, video_id):
        """Complete audio processing pipeline for a video"""
        results = {
            'video_id': video_id,
            'video_path': video_path,
            'audio_path': None,
            'transcription': None,
            'error': None
        }
        
        try:
            # Extract audio
            audio_path = self.extract_audio_from_video(video_path)
            if not audio_path:
                results['error'] = "Failed to extract audio"
                return results
            
            results['audio_path'] = audio_path
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path, Path(video_path).name)
            if not transcription:
                results['error'] = "Failed to transcribe audio"
                return results
            
            results['transcription'] = transcription
            
            # Clean up audio file to save space
            if audio_path.exists():
                audio_path.unlink()
                print(f"Cleaned up temporary audio file: {audio_path.name}")
            
            return results
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Error in audio processing pipeline: {e}")
            return results