import os
from pathlib import Path
import torch
from PIL import Image

# Monkey Patch, or degrade to 0.25.2 version
import huggingface_hub
from huggingface_hub import hf_hub_download

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = hf_hub_download

from py_real_esrgan.model import RealESRGAN

root = os.getcwd()
script_dir = Path(__file__).parent
# script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
cwd = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RealESRGAN(device, scale=4)
ckpt_path = "ckpts/RealESRGAN_x4.pth"
model.load_weights(ckpt_path, download=True)

image_path = Path("bridge_12.jpg")
if not image_path.exists():
    print(f"Image {image_path} not found.")
    exit()

try:
    image = Image.open(image_path).convert("RGB")
    sr_image = model.predict(image)
except Exception as e:
    print(f"Error processing image {image_path}: {e}")
    exit()

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, f"sr_{image_path.name}")
sr_image.save(save_path)
print(f"SR image saved to {save_path}")
