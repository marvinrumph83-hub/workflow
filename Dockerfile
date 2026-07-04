FROM runpod/worker-comfyui:5.8.4-base

# Create model directories at the correct path (ComfyUI expects /ComfyUI/models, NOT /comfyui/models)
RUN mkdir -p \
    /ComfyUI/models/diffusion_models \
    /ComfyUI/models/text_encoders \
    /ComfyUI/models/vae

# Download diffusion model
RUN wget -O /ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors

# Download text encoder
RUN wget -O /ComfyUI/models/text_encoders/qwen_3_4b.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors

# Download VAE
RUN wget -O /ComfyUI/models/vae/ae.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors

# Install custom nodes required for this workflow
# ModelSamplingAuraFlow is NOT a built-in node — it comes from a custom node package
# This workflow requires the ComfyUI-AuraFlow custom node
RUN git clone https://github.com/LarryAURA/ComfyUI-AuraFlow.git /ComfyUI/custom_nodes/ComfyUI-AuraFlow 2>/dev/null || \
    echo "ComfyUI-AuraFlow repo not available, trying alternative..."
    
# Install any Python dependencies for custom nodes
RUN pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-AuraFlow/requirements.txt 2>/dev/null || true

# Verify models were placed correctly
RUN ls -la /ComfyUI/models/diffusion_models/ && \
    ls -la /ComfyUI/models/text_encoders/ && \
    ls -la /ComfyUI/models/vae/