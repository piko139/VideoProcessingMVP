FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download required models during build
RUN ollama serve & \
    sleep 10 && \
    ollama pull llama3.2:3b && \
    ollama pull nomic-embed-text && \
    pkill ollama

# Copy application code
COPY processors/ ./processors/
COPY llm/ ./llm/
COPY database/ ./database/
COPY demo_step1.py .
COPY main.py .
COPY uploads/ ./uploads/

# Expose ports
EXPOSE 8000 11434

# Start Ollama and FastAPI
CMD ["bash", "-c", "ollama serve & sleep 5 && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"]