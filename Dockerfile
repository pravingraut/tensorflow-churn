FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]

"docker build -t churn-api:latest ."
"docker run -p 8080:8080 churn-api:latest"