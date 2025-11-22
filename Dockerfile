# Qwen Omni Voice Assistant Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install flash-attention (optional, may fail on some systems)
RUN pip3 install flash-attn --no-build-isolation || echo "Flash attention not installed"

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /app/models /app/conversation_history

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
CMD ["python3", "main.py", "--host", "0.0.0.0", "--port", "7860"]
