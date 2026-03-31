import argparse
import os
import time
from pathlib import Path

import bm3d
import cv2
from skimage.util import random_noise
from skimage.restoration import estimate_sigma


root_dir = Path(__file__).resolve().parent
os.chdir(root_dir)

parser = argparse.ArgumentParser(description="BM3D 去噪示例")
parser.add_argument(
    "--input", default="test_images/Bridge_bridge_45.jpg", help="输入图像路径"
)
parser.add_argument(
    "--sigma", type=float, default=None, help="噪声标准差（0-255），不传则自动估计"
)
parser.add_argument("--add-noise", action="store_true", help="是否先人工加高斯噪声")
parser.add_argument(
    "--average-sigmas",
    dest="average_sigmas",
    action="store_true",
    help="自动估计sigma时是否对多通道求平均（默认开启）",
)
parser.add_argument(
    "--no-average-sigmas",
    dest="average_sigmas",
    action="store_false",
    help="自动估计sigma时不对多通道求平均",
)
parser.set_defaults(average_sigmas=True)
args = parser.parse_args()

# args.sigma = 5

path = Path(args.input)
img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"输入图像不存在或不可读: {path}")

img_f = img.astype("float32") / 255.0
if args.add_noise:
    input_for_denoise = random_noise(
        img_f, mode="gaussian", var=(args.sigma / 255.0) ** 2
    )
else:
    input_for_denoise = img_f

start = time.perf_counter()

if args.sigma is None:
    channel_axis = -1 if input_for_denoise.ndim == 3 else None
    # 目前最常用、最稳健的小波域噪声水平估计器，专为高斯噪声设计，对真实图像也很有效
    sigma_est = estimate_sigma(
        input_for_denoise,
        channel_axis=channel_axis,
        average_sigmas=args.average_sigmas,
    )  # average_sigmas=True，能防止因某个波段噪声过大或过小导致的去噪过度/不足
    if isinstance(sigma_est, (list, tuple)):
        sigma_psd = [float(v) for v in sigma_est]
    else:
        sigma_psd = float(sigma_est)
    print(f"Estimated sigma_psd: {sigma_psd}")
else:
    sigma_psd = float(args.sigma) / 255.0
    print(f"Manual sigma_psd: {sigma_psd}")

# sigma_psd 通常给噪声标准差 float （灰度）或 list[float] （彩色每通道一个）
denoised = bm3d.bm3d(
    input_for_denoise, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES
)
if denoised.ndim == 3 and img.ndim == 2:
    denoised = cv2.cvtColor(denoised.astype("float32"), cv2.COLOR_BGR2GRAY)
elapsed = time.perf_counter() - start

denoised_u8 = (denoised.clip(0, 1) * 255).astype("uint8")
suffix = "_noisy_denoised_bm3d" if args.add_noise else "_denoised_bm3d"
out_path = Path.cwd() / f"{path.stem}{suffix}{path.suffix}"
cv2.imwrite(str(out_path), denoised_u8)
print(f"Saved: {out_path}")
print(f"Elapsed: {elapsed:.4f}s")
