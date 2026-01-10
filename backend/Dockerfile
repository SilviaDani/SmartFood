# Backend Dockerfile - Multi-stage build per distroless

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y libpq-dev gcc && pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN pip install -r requirements.txt

# Stage 2: Runtime (slim - molto leggero comunque)
FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Set ENV
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/smartfood

# Create non-root user per sicurezza
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run server from /app/smartfood directory
WORKDIR /app/smartfood
CMD ["python", "app.py"]
