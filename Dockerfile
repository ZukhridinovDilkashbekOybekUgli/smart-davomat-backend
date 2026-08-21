FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN curl -L --fail --retry 3 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx -o yolov8n.onnx
COPY app.py .
COPY index.html .
EXPOSE 10000
CMD ["gunicorn","--bind","0.0.0.0:10000","--workers","1","--timeout","120","app:app"]
