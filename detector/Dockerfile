FROM python:3.11-slim

# Установка зависимостей для OpenCV + инструменты разработки
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY motion_detector.py .

ENV RTMP_URL=rtmp://nginx-rtmp/live
ENV OUTPUT_DIR=/recordings
ENV BUFFER_SECONDS=5
ENV POST_MOTION_SECONDS=5
ENV MIN_AREA=500
ENV AUTO_START_RECORDING=false

CMD ["python", "-u", "motion_detector.py"]
