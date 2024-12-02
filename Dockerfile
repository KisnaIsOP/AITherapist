# Use slim Python image to minimize resource usage
FROM python:3.11-slim-bullseye

# Set environment variables to reduce overhead
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1

# Set working directory
WORKDIR /app

# Install system dependencies with minimal footprint
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY requirements.txt .
COPY app.py .
COPY .env.example .env

# Install Python dependencies with minimal overhead
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn

# Resource-constrained Gunicorn configuration
COPY gunicorn_config.py .

# Expose port
EXPOSE 5000

# Run with limited resources
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
