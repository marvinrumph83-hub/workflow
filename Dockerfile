# RunPod Serverless ComfyUI + Wan 2.1 Text-to-Video
# GPU: Recommended RTX 4090 / A100 / H100 (24GB+ VRAM for 1.3B model)

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1 \
    COMFYUI_PATH=/comfyui

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI
WORKDIR /
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH}

WORKDIR ${COMFYUI_PATH}

# Install ComfyUI dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for video generation
RUN pip install --no-cache-dir \
    runpod \
    requests \
    opencv-python-headless \
    imageio \
    imageio-ffmpeg \
    safetensors \
    accelerate \
    xformers

# Create model directories (correct ComfyUI paths)
RUN mkdir -p models/unet \
    && mkdir -p models/clip \
    && mkdir -p models/vae \
    && mkdir -p output

# Download Wan 2.1 models to CORRECT ComfyUI directories
RUN wget -q --show-progress -O models/unet/wan2.1_t2v_1.3B_fp16.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"

RUN wget -q --show-progress -O models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

RUN wget -q --show-progress -O models/vae/wan_2.1_vae.safetensors \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

# Copy handler and workflow
COPY handler.py /handler.py
COPY text_to_video_wan_api.json ${COMFYUI_PATH}/workflows/text_to_video_wan_api.json

# Expose ComfyUI port (optional, for debugging)
EXPOSE 8188

# Set the handler as entrypoint for RunPod Serverless
CMD ["python", "-u", "/handler.py"]
