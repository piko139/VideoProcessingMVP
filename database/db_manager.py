#database/db_manager.py

from sqlalchemy import create_engine, text

class DatabaseManager:
    """Handle database operations for highlights"""
    
    def __init__(self, engine):
        self.engine = engine
    
    def store_highlight(self, highlight_data):
        """Store a single highlight in database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    INSERT INTO highlights (
                        video_id, timestamp_start, timestamp_end, description, 
                        llm_summary, highlight_type, confidence_score, embedding
                    ) VALUES (
                        :video_id, :timestamp_start, :timestamp_end, :description,
                        :llm_summary, :highlight_type, :confidence_score, :embedding
                    ) RETURNING id;
                """), highlight_data)
                
                highlight_id = result.fetchone()[0]
                conn.commit()
                
                return highlight_id
                
        except Exception as e:
            print(f"Error storing highlight: {e}")
            return None
    
    def store_highlights_batch(self, highlights_list):
        """Store multiple highlights in database"""
        stored_ids = []
        
        try:
            with self.engine.connect() as conn:
                for highlight in highlights_list:
                    result = conn.execute(text("""
                        INSERT INTO highlights (
                            video_id, timestamp_start, timestamp_end, description, 
                            llm_summary, highlight_type, confidence_score, embedding
                        ) VALUES (
                            :video_id, :timestamp_start, :timestamp_end, :description,
                            :llm_summary, :highlight_type, :confidence_score, :embedding
                        ) RETURNING id;
                    """), highlight)
                    
                    highlight_id = result.fetchone()[0]
                    stored_ids.append(highlight_id)
                
                conn.commit()
                print(f"Successfully stored {len(stored_ids)} highlights")
                
        except Exception as e:
            print(f"Error in batch storage: {e}")
        
        return stored_ids
    def keyword_search(self, query, limit=5):
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT h.id, h.description, h.llm_summary, h.timestamp_start, 
                    v.filename, 0.5 as distance
                FROM highlights h
                JOIN videos v ON h.video_id = v.id
                WHERE h.description ILIKE :query 
                OR h.llm_summary ILIKE :query
                OR v.filename ILIKE :query
                LIMIT :limit
            """), {'query': f'%{query}%', 'limit': limit})
            
            return [{'id': r[0], 'description': r[1], 'summary': r[2], 
                    'timestamp': r[3], 'video': r[4], 'distance': r[5]} 
                    for r in result.fetchall()]
                
    def search_similar_highlights(self, query_embedding, limit=5):
        """Search for similar highlights using vector similarity"""
        with self.engine.connect() as conn:
            # Fix: Ensure proper vector format with ALL values inside brackets
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            query_sql = f"""
                SELECT h.id, h.description, h.llm_summary, h.timestamp_start, 
                    v.filename, (h.embedding <=> '{embedding_str}'::vector) as distance
                FROM highlights h
                JOIN videos v ON h.video_id = v.id
                ORDER BY h.embedding <=> '{embedding_str}'::vector
                LIMIT {limit};
            """
            
            result = conn.execute(text(query_sql))
            
            results = []
            for row in result.fetchall():
                results.append({
                    'id': row[0],
                    'description': row[1],
                    'summary': row[2],
                    'timestamp': row[3],
                    'video': row[4],
                    'distance': row[5]
                })
            
            return results
            
            
    def register_video(self, metadata):
        """Store video metadata in database"""
        try:
            with self.engine.connect() as conn:
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