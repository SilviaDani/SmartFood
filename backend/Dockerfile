# Backend Dockerfile
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y libpq-dev gcc && pip install --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy Django app
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
