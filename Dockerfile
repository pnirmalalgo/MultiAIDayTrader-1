# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories for logs, plots, and generated scripts
RUN mkdir -p /app/generated_scripts /app/plots /app/logs

# Copy supervisor configuration to proper location
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7860 6379

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
