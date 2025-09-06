# main.py - FastAPI Backend for Video Highlights Chat

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import time

# Import your modules
from database.models import setup_database
from database.db_manager import DatabaseManager
from llm.embedding_generator import EmbeddingGenerator

# Initialize FastAPI
app = FastAPI(title="Video Highlights Chat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatQuery(BaseModel):
    question: str
    max_results: Optional[int] = 5

class HighlightResult(BaseModel):
    video_name: str
    timestamp_start: float
    description: str
    summary: str
    distance: float

class ChatResponse(BaseModel):
    question: str
    highlights: List[HighlightResult]
    total_found: int

# Global services
db_manager = None
embedding_generator = None

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global db_manager, embedding_generator
    
    # Database setup
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://video_user:video_pass@postgres:5432/video_highlights')
    engine = setup_database(DATABASE_URL)
    
    if not engine:
        raise Exception("Failed to connect to database")
    
    db_manager = DatabaseManager(engine)
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    if not embedding_generator.load_model():
        raise Exception("Failed to load embedding model")
    
    print("FastAPI services initialized successfully")

@app.get("/")
async def root():
    return {"message": "Video Highlights Chat API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "services": ["database", "embeddings"]}

@app.post("/chat", response_model=ChatResponse)
async def chat(query: ChatQuery):
    """Chat endpoint to search video highlights"""
    
    if not db_manager or not embedding_generator:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        start_time = time.time()
        
        # Generate embedding for the question
        expanded_query = f"{query.question} speech dialogue conversation scene visual transition"
        question_embedding = embedding_generator.generate_embedding(expanded_query)
        
        if not question_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding for question")
        
        # Try semantic search first
        results = db_manager.search_similar_highlights(question_embedding, limit=query.max_results)
        
        # Format results
        highlights = []
        for result in results:
            highlights.append(HighlightResult(
                video_name=result['video'],
                timestamp_start=result['timestamp'],
                description=result['description'],
                summary=result['summary'],
                distance=result['distance']
            ))
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            question=query.question,
            highlights=highlights,
            total_found=len(highlights)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/videos")
async def get_videos():
    """Get list of processed videos"""
    
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        with db_manager.engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("""
                SELECT v.id, v.filename, v.duration, COUNT(h.id) as highlight_count
                FROM videos v
                LEFT JOIN highlights h ON v.id = h.video_id
                GROUP BY v.id, v.filename, v.duration
                ORDER BY v.filename;
            """))
            
            videos = []
            for row in result.fetchall():
                videos.append({
                    "id": row[0],
                    "filename": row[1],
                    "duration": row[2],
                    "highlight_count": row[3]
                })
            
            return {"videos": videos}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get videos: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        with db_manager.engine.connect() as conn:
            from sqlalchemy import text
            
            # Get counts
            video_count = conn.execute(text("SELECT COUNT(*) FROM videos")).fetchone()[0]
            highlight_count = conn.execute(text("SELECT COUNT(*) FROM highlights")).fetchone()[0]
            
            # Get total duration
            total_duration = conn.execute(text("SELECT SUM(duration) FROM videos")).fetchone()[0] or 0
            
            return {
                "videos": video_count,
                "highlights": highlight_count,
                "total_duration_seconds": total_duration,
                "avg_highlights_per_video": round(highlight_count / max(video_count, 1), 1)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)