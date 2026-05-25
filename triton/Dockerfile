FROM nvcr.io/nvidia/tritonserver:24.12-py3

ENV PYTHONNOUSERSITE=True
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# cu118: laurel 서버(드라이버 535.x)에서 검증된 조합
# torch가 CUDA 11.8 라이브러리를 번들로 포함하므로 베이스 이미지 CUDA 버전 영향 없음
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0

RUN python3 -m pip install --no-cache-dir \
    "numpy<2" \
    "transformers==5.8.1" \
    "accelerate==1.13.0" \
    "librosa==0.10.2" \
    "soundfile==0.12.1" \
    "sentencepiece" \
    "safetensors" \
    "pillow" \
    "supertonic" \
    "langdetect" \
    "silero-vad"
