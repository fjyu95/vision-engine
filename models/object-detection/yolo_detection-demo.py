# ln -s /data/yufangjie/code/vision-engine/ckpts/yolo ckpts

import argparse
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# 将工作目录切换到当前脚本所在目录，便于使用相对路径
root_dir = Path(__file__).resolve().parent
os.chdir(root_dir)

# 常用模型名（如 yolo11n-obb.pt）在本地不存在时会自动下载
# Linux 默认缓存目录通常为 ~/.cache/ultralytics
# 若设置环境变量 ULTRALYTICS_CACHE，会优先使用该目录
ultralytics_cache_dir = Path(
    os.environ.get("ULTRALYTICS_CACHE", str(Path.home() / ".cache" / "ultralytics"))
)
# print(f"Ultralytics cache dir: {ultralytics_cache_dir}")


parser = argparse.ArgumentParser(description="YOLO OBB 目标检测示例")
parser.add_argument(
    "--input", default="test_images/Airport_airport_49.jpg", help="输入图像路径"
)
parser.add_argument(
    "--model", default="ckpts/yolo11n-obb.pt", help="模型权重路径或模型名"
)
parser.add_argument(
    "--imgsz",
    type=int,
    default=1024,
    help="推理输入尺寸,会resize到此大小,遥感通常1024/1280",
)
parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
parser.add_argument("--device", default="0", help="推理设备，如 cpu、0、0,1")
parser.add_argument(
    "--half", dest="half", action="store_true", help="是否开启 FP16 半精度推理"
)
parser.add_argument(
    "--no-half", dest="half", action="store_false", help="关闭 FP16 半精度推理"
)
parser.set_defaults(half=True)
args = parser.parse_args()

# 先检查输入图像是否可读，避免在推理阶段才报错
path = Path(args.input)
img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"输入图像不存在或不可读: {path}")

# 加载模型：
# - 传本地路径：直接加载本地权重
# - 传模型名：若本地未命中，Ultralytics 会自动下载到缓存目录
# 可提前手动下载（只下载，不推理）：
# python -c "from ultralytics import YOLO; YOLO('yolo11n-obb.pt')"
model = YOLO(args.model)
print(f"Model names: {model.names}")

# 单张图推理
start = time.perf_counter()
results = model(
    str(path),
    imgsz=args.imgsz,
    conf=args.conf,
    iou=args.iou,
    device=args.device,
    half=args.half,
    verbose=False,
    # save=True,  # 会生成到runs文件夹下
)
elapsed = time.perf_counter() - start

if not results:
    raise RuntimeError("未获得推理结果")

# 可视化并保存结果图
result = results[0]
plotted = result.plot()
out_path = Path.cwd() / f"{path.stem}_detected_obb{path.suffix}"
cv2.imwrite(str(out_path), plotted)

print(f"Saved: {out_path}")
print(f"Elapsed: {elapsed:.4f}s")
detections = 0
# OBB 模型优先读取 result.obb；非 OBB 模型回退到 result.boxes
if result.obb is not None:
    detections = len(result.obb)
elif result.boxes is not None:
    detections = len(result.boxes)
print(f"Detections: {detections}")
