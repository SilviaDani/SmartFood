# Distroless Backend Dockerfile - Multi-stage build

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y libpq-dev gcc && pip install --upgrade pip

# Copy requirements
COPY ../backend/requirements.txt .

# Install Python dependencies in a virtual environment
RUN pip install --user -r requirements.txt

# Stage 2: Runtime distroless
FROM gcr.io/distroless/python3-nonroot:latest

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/nonroot/.local

# Copy application code
COPY ../backend/ .

# Set PATH to use user packages
ENV PATH=/home/nonroot/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "manage", "runserver", "0.0.0.0:8000"]
