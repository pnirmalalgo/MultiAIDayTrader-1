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
# ✅ CHANGED: Added proper permissions to prevent write issues
RUN mkdir -p /app/generated_scripts /app/plots /app/logs && \
    chmod -R 755 /app/logs

# Copy supervisor configuration to proper location
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports (8000 for combined app, 6379 for Redis)
EXPOSE 8000 6379

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]