FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 \
    torchvision==0.20.1

RUN pip install --no-cache-dir \
    "transformers==4.47.0" \
    "accelerate==1.2.1" \
    "librosa==0.10.2" \
    "soundfile==0.12.1" \
    "pillow"
