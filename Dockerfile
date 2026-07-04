FROM runpod/worker-comfyui:5.8.4-base

# Create model directories
RUN mkdir -p \
    /comfyui/models/diffusion_models \
    /comfyui/models/text_encoders \
    /comfyui/models/vae

# Download diffusion model
RUN wget -O /comfyui/models/diffusion_models/z_image_turbo_bf16.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors

# Download text encoder
RUN wget -O /comfyui/models/text_encoders/qwen_3_4b.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors

# Download VAE
RUN wget -O /comfyui/models/vae/ae.safetensors \
    https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors
