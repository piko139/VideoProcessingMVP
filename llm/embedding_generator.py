# llm/embedding_generator.py

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Generate embeddings for highlight descriptions"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
        self.embedding_dimension = 384  # Dimension for sentence-transformers
        
    def load_model(self):
        """Load sentence transformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully on device: {self.model.device}")
            return True
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            return False
    
    def generate_embedding(self, text):
        """Generate embedding for a single text"""
        if not self.model:
            print("Model not loaded")
            return None
        
        try:
            # Ensure text is string and not too long
            text = str(text)[:1000]  # Limit to 1000 characters
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=True)
            
            # Convert to numpy array and then to list for database storage
            embedding_list = embedding.cpu().numpy().tolist()
            
            return embedding_list
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts):
        """Generate embeddings for multiple texts"""
        if not self.model:
            print("Model not loaded")
            return []
        
        try:
            # Clean and prepare texts
            clean_texts = [str(text)[:1000] for text in texts]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(clean_texts, convert_to_tensor=True)
            
            # Convert to list format
            embeddings_list = embeddings.cpu().numpy().tolist()
            
            return embeddings_list
            
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return []