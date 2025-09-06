# Video Highlights Processing Pipeline

**Note: This is a basic MVP demonstration of video processing and semantic search capabilities.**

## Overview

This project demonstrates an AI-powered video processing pipeline that extracts highlights from videos using machine learning and allows users to search through video content using natural language queries. The system combines speech-to-text transcription, computer vision analysis, and large language model processing to create a searchable database of video moments.

## Architecture

### System Components

1. **PostgreSQL with pgvector** - Stores video metadata and highlights with vector embeddings
2. **Backend (FastAPI)** - Processes videos and provides chat API endpoints  
3. **Frontend (HTML/JS)** - Simple web interface for user queries
4. **Ollama LLM** - Generates intelligent video highlights
5. **Whisper** - Speech-to-text transcription
6. **OpenCV** - Video frame analysis and scene change detection
7. **SentenceTransformers** - Creates semantic embeddings for similarity search

### Chat Architecture

```
User Query → Frontend → FastAPI Backend → Embedding Generator → Vector Search → Results
```

1. User enters question in web interface
2. Frontend sends POST request to `/chat` endpoint
3. Backend generates embedding for user's question
4. System searches database using vector similarity (pgvector)
5. Results ranked by semantic similarity are returned
6. Frontend displays relevant video moments with timestamps

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with Docker GPU support (optional, will fallback to CPU)
- At least 8GB RAM for model loading

## Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/piko139/VideoProcessingMVP.git VideoProcessingMVP
cd VideoProcessingMVP
```

### 2. Start All Services

```bash
# Build and start all containers
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

This starts:
- **PostgreSQL**: Port 5432
- **Backend API**: Port 8000 
- **Frontend**: Port 3000
- **Ollama**: Port 11434

### 3. Process Videos

Run the demo script to process the included videos:

```bash
docker-compose exec backend python demo_step1.py
```

This will:
- Extract audio and transcribe speech using Whisper
- Analyze video frames and detect scene changes
- Generate highlights using Ollama LLM
- Create semantic embeddings and store in database

### 4. Access Chat Interface

Open http://localhost:3000 in your browser to start asking questions about your videos.

## API Endpoints

### Core Endpoints

- `GET /` - API status
- `GET /health` - Service health check
- `POST /chat` - Main chat endpoint for video queries

### Chat Endpoint

**URL**: `POST /chat`

**Request Body**:
```json
{
  "question": "What was said about magnetic reconnection?",
  "max_results": 5
}
```

**Response**:
```json
{
  "question": "What was said about magnetic reconnection?",
  "highlights": [
    {
      "video_name": "MMS_ChallengesNASA.mp4",
      "timestamp_start": 10.1,
      "description": "The speaker explains that magnetic reconnection is a process to get a 3D observation of this phenomenon.",
      "summary": "Key scientific explanation of plasma physics phenomenon",
      "distance": 0.35
    }
  ],
  "total_found": 1
}
```

### Additional Endpoints

- `GET /videos` - List all processed videos
- `GET /stats` - Database statistics

## Project Structure

```
VideoProcessingMVP/
├── processors/           # Video, audio, and visual processing modules
│   ├── video_processor.py
│   ├── audio_processor.py
│   └── visual_processor.py
├── llm/                  # LLM and embedding modules
│   ├── highlight_detector.py
│   └── embedding_generator.py
├── database/             # Database connection and models
│   ├── db_manager.py
│   └── models.py
├── frontend/             # Simple web interface
│   ├── src/index.html
│   └── Dockerfile
├── uploads/              # Video files for processing
├── demo_step1.py         # Complete pipeline demonstration
├── main.py               # FastAPI backend application
├── docker-compose.yml    # Container orchestration
├── Dockerfile           # Backend container definition
└── requirements.txt     # Python dependencies
```

## How It Works

### Step 1: Video Processing Pipeline

1. **Video Metadata Extraction** - Analyzes file properties (duration, resolution, etc.)
2. **Audio Processing** - Extracts audio and transcribes speech using Whisper
3. **Visual Analysis** - Extracts frames and detects scene changes using OpenCV
4. **Timeline Creation** - Combines audio and visual events into chronological timeline
5. **LLM Highlight Detection** - Sends timeline to Ollama LLM to identify important moments
6. **Embedding Generation** - Creates semantic vectors using SentenceTransformers
7. **Database Storage** - Stores highlights with embeddings in PostgreSQL

### Step 2: Interactive Search

1. **User Query** - Natural language question about video content
2. **Query Embedding** - Converts question to semantic vector
3. **Similarity Search** - Uses pgvector to find similar highlights
4. **Result Ranking** - Orders results by semantic similarity score
5. **Response Formatting** - Returns relevant video moments with context

## Limitations

This is a basic demonstration with several limitations:

- **Small Dataset**: Only processes 3 short videos
- **Simple LLM Prompting**: Basic highlight detection logic
- **Limited Search Quality**: Semantic matching depends on description quality
- **No User Management**: Single-user system without authentication
- **Basic Frontend**: Minimal HTML interface without advanced features
- **Error Handling**: Limited robustness for production use

## Troubleshooting

### Services Not Starting

```bash
# Check container status
docker-compose ps

# View logs for specific service
docker-compose logs backend
docker-compose logs postgres
```

### No Search Results

- Ensure videos have been processed: `docker-compose exec backend python demo_step1.py`
- Check database has highlights: `curl http://localhost:8000/stats`
- Try different search terms

### GPU Not Detected

The system will work on CPU but will be slower. Ensure:
- NVIDIA Docker runtime is installed
- GPU is available: `nvidia-smi`
- Docker has GPU access configured

## Development

### Adding New Videos

1. Place video files in `uploads/` directory
2. Run processing: `docker-compose exec backend python demo_step1.py`
3. Videos should be .mp4 or .mov format, 30 seconds to 90 seconds long

### Modifying Search Logic

Edit `database/db_manager.py` to adjust vector search parameters or add keyword search functionality.

### Improving LLM Prompts

Modify `llm/highlight_detector.py` to enhance highlight detection quality.

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, PyTorch
- **Database**: PostgreSQL with pgvector extension
- **ML Models**: OpenAI Whisper, SentenceTransformers, Ollama (Llama 3.2)
- **Computer Vision**: OpenCV, MoviePy
- **Frontend**: HTML, JavaScript, Nginx
- **Infrastructure**: Docker, Docker Compose

## License

This project is for educational and demonstration purposes.
