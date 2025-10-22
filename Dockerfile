# Use an official Python slim image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Avoid warnings
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system deps (for shap, etc. if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose port
EXPOSE 8080

# Use uvicorn to run FastAPI app (module path may vary)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]