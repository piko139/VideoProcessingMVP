# database/models.py

from sqlalchemy import create_engine, text

def setup_database(database_url):
    """Create database connection and setup schema"""
    try:
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            print(f"Connected to PostgreSQL: {result.fetchone()[0]}")
            
        # Create schema
        create_schema(engine)
        
        return engine
    except Exception as e:
        print(f"Database setup failed: {e}")
        return None

def create_schema(engine):
    """Create database tables and indexes"""
    schema_sql = """
    CREATE EXTENSION IF NOT EXISTS vector;
    
    CREATE TABLE IF NOT EXISTS videos (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        filepath VARCHAR(500) NOT NULL,
        duration FLOAT NOT NULL,
        fps FLOAT NOT NULL,
        resolution VARCHAR(50) NOT NULL,
        file_size BIGINT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed_at TIMESTAMP,
        status VARCHAR(50) DEFAULT 'uploaded'
    );
    
    CREATE TABLE IF NOT EXISTS highlights (
        id SERIAL PRIMARY KEY,
        video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
        timestamp_start FLOAT NOT NULL,
        timestamp_end FLOAT NOT NULL,
        description TEXT NOT NULL,
        llm_summary TEXT,
        highlight_type VARCHAR(100),
        confidence_score FLOAT DEFAULT 0.0,
        embedding vector(384),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_highlights_video_id ON highlights(video_id);
    CREATE INDEX IF NOT EXISTS idx_highlights_timestamp ON highlights(timestamp_start);
    CREATE INDEX IF NOT EXISTS idx_highlights_embedding ON highlights 
    USING ivfflat (embedding vector_cosine_ops);
    """
    
    try:
        with engine.connect() as conn:
            for statement in schema_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
        print("Database schema created successfully!")
    except Exception as e:
        print(f"Schema creation failed: {e}")
