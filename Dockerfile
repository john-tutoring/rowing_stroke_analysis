# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgles2 \
    libegl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY joint_info.py extraction.py split_strokes.py feature_extraction.py app.py index.html ./
COPY static ./static
COPY pose_landmarker_lite.task ./
RUN mkdir -p artifacts
COPY artifacts/model.joblib artifacts/model_meta.json ./artifacts/

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT} -t 300 --workers 1 app:app"]
