FROM python:3.11-slim

WORKDIR /app

# Install system dependencies optimized for PyTorch CPU
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libblas3 \
    liblapack3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install CPU-optimized PyTorch and dependencies
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy entrypoint and make it executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# REMOVED: Don't create appuser, stay as root
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

# Set CPU optimization environment variables
ENV TORCH_NUM_THREADS=2
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV NUMBA_NUM_THREADS=2
ENV MALLOC_ARENA_MAX=2
ENV PYTHONUNBUFFERED=1

# Expose port - Railway will override this
EXPOSE $PORT

# Extended health check for model loading time - increased timeouts for Railway
HEALTHCHECK --interval=60s --timeout=120s --start-period=300s --retries=5 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Single CMD at the end
CMD ["./entrypoint.sh"]