# Use Python 3.10 slim image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    redis-server \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/generated_scripts /app/plots /app/logs /shared && \
    chmod -R 777 /shared && \
    chmod -R 755 /app/logs

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000 6379

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
