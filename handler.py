"""
RunPod Serverless Handler for ComfyUI Wan 2.1 Text-to-Video

This handler receives text prompts and returns generated video URLs.
"""

import runpod
import subprocess
import time
import json
import requests
import base64
import os
import random
import threading
from pathlib import Path

# Configuration
COMFYUI_PATH = "/comfyui"
COMFYUI_PORT = 8188
OUTPUT_DIR = f"{COMFYUI_PATH}/output"

# ComfyUI process
comfy_process = None
comfy_ready = False


def start_comfyui():
    """Start ComfyUI server in the background."""
    global comfy_process, comfy_ready
    
    if comfy_process is not None:
        return
    
    print("=" * 60)
    print("Starting ComfyUI server...")
    print("=" * 60)
    
    # Start ComfyUI process
    comfy_process = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", str(COMFYUI_PORT), "--disable-auto-launch"],
        cwd=COMFYUI_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream ComfyUI output in background thread
    def stream_output():
        for line in comfy_process.stdout:
            print(f"[ComfyUI] {line.strip()}")
    
    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()
    
    # Wait for ComfyUI to be ready (up to 10 minutes for model loading)
    max_retries = 600  # 10 minutes
    print(f"Waiting for ComfyUI to load models (max {max_retries}s)...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"http://127.0.0.1:{COMFYUI_PORT}/system_stats", timeout=5)
            if response.status_code == 200:
                print("=" * 60)
                print(f"✅ ComfyUI ready after {i+1} seconds!")
                print("=" * 60)
                comfy_ready = True
                return
        except requests.exceptions.RequestException:
            pass
        
        if i % 30 == 0 and i > 0:
            print(f"Still waiting... ({i}s elapsed)")
        
        time.sleep(1)
    
    raise RuntimeError("ComfyUI failed to start within 10 minutes")


def wait_for_comfyui():
    """Wait for ComfyUI to be ready."""
    global comfy_ready
    
    if comfy_ready:
        return True
    
    # Wait up to 5 minutes if called before ready
    for i in range(300):
        if comfy_ready:
            return True
        time.sleep(1)
    
    return False


def load_workflow():
    """Load the Wan 2.1 text-to-video workflow."""
    workflow_path = f"{COMFYUI_PATH}/workflows/text_to_video_wan_api.json"
    with open(workflow_path, 'r') as f:
        return json.load(f)


def update_workflow(workflow, prompt, negative_prompt=None, width=480, height=832, frames=160, steps=30, cfg=6, seed=None, fps=16):
    """Update workflow with user parameters."""
    
    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**63 - 1)
    
    # Default negative prompt (Chinese - optimized for Wan 2.1)
    default_negative = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    if negative_prompt is None:
        negative_prompt = default_negative
    
    # Update workflow nodes based on your API format
    # Node 6: Positive prompt
    if "6" in workflow:
        workflow["6"]["inputs"]["text"] = prompt
    
    # Node 7: Negative prompt
    if "7" in workflow:
        workflow["7"]["inputs"]["text"] = negative_prompt
    
    # Node 40: EmptyHunyuanLatentVideo (video dimensions and frames)
    if "40" in workflow:
        workflow["40"]["inputs"]["width"] = width
        workflow["40"]["inputs"]["height"] = height
        workflow["40"]["inputs"]["length"] = frames
    
    # Node 3: KSampler (seed, steps, cfg)
    if "3" in workflow:
        workflow["3"]["inputs"]["seed"] = seed
        workflow["3"]["inputs"]["steps"] = steps
        workflow["3"]["inputs"]["cfg"] = cfg
    
    # Node 49: CreateVideo (fps)
    if "49" in workflow:
        workflow["49"]["inputs"]["fps"] = fps
    
    return workflow, seed


def queue_prompt(workflow):
    """Queue a prompt in ComfyUI and return the prompt ID."""
    response = requests.post(
        f"http://127.0.0.1:{COMFYUI_PORT}/prompt",
        json={"prompt": workflow},
        timeout=30
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to queue prompt: {response.text}")
    
    return response.json()["prompt_id"]


def wait_for_completion(prompt_id, timeout=1800):
    """Wait for the prompt to complete (30 min timeout for video)."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"http://127.0.0.1:{COMFYUI_PORT}/history/{prompt_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    return history[prompt_id]
        except requests.exceptions.RequestException as e:
            print(f"Warning: Error checking history: {e}")
        
        time.sleep(3)
    
    raise RuntimeError(f"Prompt timed out after {timeout} seconds")


def get_output_video(history):
    """Extract the output video path from the history."""
    outputs = history.get("outputs", {})
    
    for node_id, node_output in outputs.items():
        # Check for videos in SaveVideo node output
        if "videos" in node_output:
            for video in node_output["videos"]:
                subfolder = video.get("subfolder", "")
                filename = video["filename"]
                if subfolder:
                    video_path = os.path.join(OUTPUT_DIR, subfolder, filename)
                else:
                    video_path = os.path.join(OUTPUT_DIR, filename)
                if os.path.exists(video_path):
                    return video_path
        
        # Also check for 'gifs' key (some video nodes use this)
        if "gifs" in node_output:
            for video in node_output["gifs"]:
                subfolder = video.get("subfolder", "")
                filename = video["filename"]
                if subfolder:
                    video_path = os.path.join(OUTPUT_DIR, subfolder, filename)
                else:
                    video_path = os.path.join(OUTPUT_DIR, filename)
                if os.path.exists(video_path):
                    return video_path
    
    return None


def video_to_base64(video_path):
    """Convert video file to base64 string."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    """
    RunPod handler function for Wan 2.1 Text-to-Video.
    
    Expected input:
    {
        "input": {
            "prompt": "a fox running in snow",           # Required: text prompt
            "negative_prompt": "blurry, low quality",   # Optional: negative prompt
            "width": 480,                                # Optional: video width (default 480)
            "height": 832,                               # Optional: video height (default 832)
            "frames": 160,                               # Optional: number of frames (default 160)
            "steps": 30,                                 # Optional: sampling steps (default 30)
            "cfg": 6,                                    # Optional: CFG scale (default 6)
            "fps": 16,                                   # Optional: frames per second (default 16)
            "seed": 12345                                # Optional: random seed
        }
    }
    
    Returns:
    {
        "video_base64": "...",  # Base64 encoded MP4 video
        "seed": 12345,          # Seed used for generation
        "prompt_id": "...",     # ComfyUI prompt ID
        "status": "success"
    }
    """
    try:
        # Wait for ComfyUI to be ready
        if not wait_for_comfyui():
            return {"error": "ComfyUI failed to start", "status": "failed"}
        
        # Parse input
        job_input = job.get("input", {})
        
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing required 'prompt' in input"}
        
        negative_prompt = job_input.get("negative_prompt")
        width = job_input.get("width", 480)
        height = job_input.get("height", 832)
        frames = job_input.get("frames", 160)
        steps = job_input.get("steps", 30)
        cfg = job_input.get("cfg", 6)
        fps = job_input.get("fps", 16)
        seed = job_input.get("seed")
        
        print(f"Generating video for prompt: {prompt[:100]}...")
        print(f"Settings: {width}x{height}, {frames} frames @ {fps}fps, {steps} steps, cfg={cfg}")
        
        # Load and update workflow
        workflow = load_workflow()
        workflow, actual_seed = update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            cfg=cfg,
            seed=seed,
            fps=fps
        )
        
        # Queue prompt and wait
        prompt_id = queue_prompt(workflow)
        print(f"Queued prompt: {prompt_id}")
        
        history = wait_for_completion(prompt_id)
        print("Generation complete!")
        
        # Get output video
        video_path = get_output_video(history)
        if not video_path:
            return {"error": "No video output found in ComfyUI history", "status": "failed"}
        
        print(f"Output video: {video_path}")
        
        # Convert to base64
        video_base64 = video_to_base64(video_path)
        
        # Cleanup output file
        try:
            os.remove(video_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file: {e}")
        
        return {
            "video_base64": video_base64,
            "video_format": "mp4",
            "seed": actual_seed,
            "prompt_id": prompt_id,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


# ========================================
# STARTUP: Start ComfyUI before accepting jobs
# ========================================
print("=" * 60)
print("RunPod Wan 2.1 Text-to-Video Handler Starting")
print("=" * 60)

# Start ComfyUI immediately at container boot
start_comfyui()

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
