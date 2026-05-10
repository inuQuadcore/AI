FROM nvcr.io/nvidia/tritonserver:24.12-py3

# gemma_s2tt / gemma_t2tt 공통 Python 의존성
# torch는 CUDA 12.4 wheel 사용 (CUDA 12.6 런타임과 호환)
RUN pip install --no-cache-dir \
    "torch>=2.5.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    "transformers>=4.47.0" \
    "accelerate>=1.2.1" \
    "librosa>=0.10.2" \
    "soundfile>=0.12.1"
