FROM nvcr.io/nvidia/tritonserver:23.10-py3

# gemma_s2tt / gemma_t2tt 공통 Python 의존성
# torch는 CUDA 12.2 wheel 사용 (드라이버 535.x와 호환)
RUN pip install --no-cache-dir \
    "torch>=2.1.0" \
    --extra-index-url https://download.pytorch.org/whl/cu122

RUN pip install --no-cache-dir \
    "transformers>=4.47.0" \
    "accelerate>=1.2.1" \
    "librosa>=0.10.2" \
    "soundfile>=0.12.1" \
    "pillow"
