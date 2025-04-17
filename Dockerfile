
FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:$PATH"

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \
    git \
    g++ \
    libsndfile1 \
    ffmpeg \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel
RUN pip install flask
RUN pip install git+https://github.com/coqui-ai/TTS

CMD ["python3", "scripts/server.py"]
