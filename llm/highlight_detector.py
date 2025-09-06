#llm/highlight_detector.py

import requests
import json
import re
import ollama
from langchain_ollama import OllamaLLM

class HighlightDetector:
    """Use LLM to identify important moments in videos"""
    
    def __init__(self):
        self.llm = None
        self.highlight_types = [
            'speech_important', 'action_sequence', 'scene_transition', 
            'emotional_moment', 'key_information', 'visual_highlight'
        ]
    
    def initialize_llm(self, model_name="llama3.2:3b"):
        """Initialize Ollama LLM connection"""
        try:
            # Test connection to Ollama
            response = ollama.list()
            available_models = [model['name'] for model in response.get('models', [])]
            
            if model_name not in available_models:
                print(f"Model {model_name} not found. Available models: {available_models}")
                if available_models:
                    model_name = available_models[0]
                    print(f"Using first available model: {model_name}")
                else:
                    print("No models available in Ollama")
                    return False
            
            self.llm = OllamaLLM(model=model_name)
            print(f"LLM initialized successfully with model: {model_name}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            # Fallback: try direct connection test
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models_data = response.json().get('models', [])
                    available_models = [model['name'] if isinstance(model, dict) else str(model) for model in models_data]
                    print(f"Direct connection successful. Available models: {available_models}")
                    
                    # Try to initialize with direct model name
                    self.llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
                    print(f"LLM initialized with direct connection: {model_name}")
                    return True
                else:
                    print(f"HTTP error: {response.status_code}")
                    return False
            except Exception as e2:
                print(f"Fallback connection also failed: {e2}")
                return False
    
    def create_timeline_prompt(self, timeline, video_name):
        """Create a structured prompt from video timeline"""
        prompt = f"""Analyze this video timeline for "{video_name}" and identify the most important moments that would make good highlights.

Video Timeline:
"""
        
        for event in timeline[:20]:  # Limit to first 20 events to avoid token limits
            timestamp = event['timestamp']
            event_type = event['type']
            content = event['content']
            prompt += f"[{timestamp:.1f}s] {event_type.upper()}: {content}\n"
        
        prompt += """
Based on this timeline, identify 3-5 key highlights that would be most interesting to viewers. For each highlight, provide:

1. Start timestamp (in seconds)
2. End timestamp (estimate based on content)
3. Type of highlight (speech_important, action_sequence, scene_transition, emotional_moment, key_information, or visual_highlight)
4. Brief description (1-2 sentences)
5. Detailed summary (2-3 sentences explaining why this moment is significant)
6. Confidence score (0.0 to 1.0)

Format your response as JSON:
{
  "highlights": [
    {
      "start_time": 10.5,
      "end_time": 25.2,
      "type": "speech_important",
      "description": "Brief description here",
      "summary": "Detailed explanation of why this moment matters",
      "confidence": 0.85
    }
  ]
}

Focus on moments with clear speech, important information, or significant visual changes.
"""
        return prompt
    
    def parse_llm_response(self, response_text):
        """Parse LLM response and extract highlights"""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                highlights_data = json.loads(json_str)
                return highlights_data.get('highlights', [])
            else:
                print("No valid JSON found in LLM response")
                return []
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: {response_text[:500]}...")
            return []
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []
    
    def detect_highlights(self, timeline, video_name, video_id):
        """Use LLM to detect highlights from timeline"""
        if not self.llm:
            print("LLM not initialized")
            return []
        
        try:
            print(f"Analyzing timeline for highlights: {video_name}")
            
            # Create prompt
            prompt = self.create_timeline_prompt(timeline, video_name)
            
            # Get LLM response
            print("Sending timeline to LLM for analysis...")
            response = self.llm.invoke(prompt)
            
            # Parse response
            highlights = self.parse_llm_response(response)
            
            # Add video_id to each highlight
            for highlight in highlights:
                highlight['video_id'] = video_id
                highlight['video_name'] = video_name
            
            print(f"LLM identified {len(highlights)} highlights")
            return highlights
            
        except Exception as e:
            print(f"Error in highlight detection: {e}")
            return []
    
    def validate_highlight(self, highlight, video_duration):
        """Validate and clean highlight data"""
        # Ensure required fields
        required_fields = ['start_time', 'end_time', 'type', 'description', 'summary']
        for field in required_fields:
            if field not in highlight:
                print(f"Missing required field: {field}")
                return None
        
        # Validate timestamps
        start_time = float(highlight['start_time'])
        end_time = float(highlight['end_time'])
        
        if start_time < 0:
            start_time = 0
        if end_time > video_duration:
            end_time = video_duration
        if start_time >= end_time:
            end_time = start_time + 5.0  # Default 5-second highlight
        
        # Validate type
        if highlight['type'] not in self.highlight_types:
            highlight['type'] = 'key_information'
        
        # Ensure confidence score
        if 'confidence' not in highlight or not isinstance(highlight['confidence'], (int, float)):
            highlight['confidence'] = 0.5
        
        # Clean up text fields
        highlight['description'] = str(highlight['description']).strip()[:500]
        highlight['summary'] = str(highlight['summary']).strip()[:1000]
        
        # Update timestamps
        highlight['start_time'] = start_time
        highlight['end_time'] = end_time
        
        return highlight
    
    def display_highlights(self, highlights, video_name):
        """Display detected highlights in a readable format"""
        print(f"\nHighlights detected for {video_name}:")
        print("=" * 60)
        
        if not highlights:
            print("No highlights detected")
            return
        
        for i, highlight in enumerate(highlights, 1):
            print(f"\nHighlight #{i}")
            print(f"Time: {highlight['start_time']:.1f}s - {highlight['end_time']:.1f}s")
            print(f"Type: {highlight['type']}")
            print(f"Confidence: {highlight['confidence']:.2f}")
            print(f"Description: {highlight['description']}")
            print(f"Summary: {highlight['summary']}")
            print("-" * 40)