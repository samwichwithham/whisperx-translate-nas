FROM python:3.11-slim
LABEL maintainer="whisperx-batch"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 tzdata && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python wheels ----------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ---------- non-root user ----------
RUN useradd -ms /bin/bash transcriber
USER transcriber

# ---------- model cache volume ----------
ENV TRANSFORMERS_CACHE=/models
VOLUME ["/models"]

# ---------- runtime defaults ----------
# default thread pool – change at runtime with  -e OMP_NUM_THREADS=
ENV OMP_NUM_THREADS=8
ENV TZ=UTC
ENV TELEGRAM_BOT_TOKEN=
ENV TELEGRAM_CHAT_ID=

WORKDIR /app
COPY whisperx_batch_transcriber.py /app/

CMD ["python", "-u", "whisperx_batch_transcriber.py", "--device", "cpu"]
