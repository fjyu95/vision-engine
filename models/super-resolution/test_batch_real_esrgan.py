# python models/super-resolution/batch_real_esrgan.py models/super-resolution/test_images --scales 2 4 8

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

try:
    import huggingface_hub
    from huggingface_hub import hf_hub_download

    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = hf_hub_download
except ImportError:
    huggingface_hub = None

try:
    from py_real_esrgan.model import RealESRGAN
except ImportError:
    RealESRGAN = None


# SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
# SUPPORTED_EXTENSIONS: Set[str] = {".tif", ".tiff", ".TIF", ".TIFF"}
SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".TIF",
    ".TIFF",
}


@dataclass
class BatchConfig:
    input_dir: Path
    output_dir: Path
    scales: List[int]
    device: str
    workers: int
    resume: bool
    quality: int
    weights_dir: Path
    progress_file: str


@dataclass
class ScaleStats:
    scale: int
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0


@dataclass
class BatchReport:
    started_at: float
    finished_at: float
    total_seconds: float
    total_images: int
    success: int
    failed: int
    skipped: int
    device: str
    cpu_memory_mb: float
    gpu_memory_mb: Optional[float]
    per_scale: List[ScaleStats]


class ProgressStore:
    def __init__(self, progress_path: Path):
        self.progress_path = progress_path
        self.lock = threading.Lock()
        self.data = {"processed": {}, "failed": {}}
        if self.progress_path.exists():
            try:
                self.data = json.loads(self.progress_path.read_text(encoding="utf-8"))
                if "processed" not in self.data:
                    self.data["processed"] = {}
                if "failed" not in self.data:
                    self.data["failed"] = {}
            except Exception:
                self.data = {"processed": {}, "failed": {}}

    def is_processed(self, key: str) -> bool:
        return key in self.data["processed"]

    def mark_processed(self, key: str, output_path: str) -> None:
        with self.lock:
            self.data["processed"][key] = output_path
            if key in self.data["failed"]:
                del self.data["failed"][key]
            self._flush()

    def mark_failed(self, key: str, error: str) -> None:
        with self.lock:
            self.data["failed"][key] = error
            self._flush()

    def _flush(self) -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.progress_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_path.replace(self.progress_path)


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("batch_real_esrgan")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "batch_sr.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def discover_images(input_dir: Path) -> List[Path]:
    images = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    return sorted(images)


def resolve_weights_path(weights_dir: Path, scale: int) -> Path:
    path = weights_dir / f"RealESRGAN_x{scale}.pth"
    if not path.exists():
        raise FileNotFoundError(f"未找到权重文件: {path}")
    return path


def create_output_path(output_root: Path, scale: int, input_path: Path) -> Path:
    out_dir = output_root / f"x{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{input_path.stem}_x{scale}{input_path.suffix}"


def get_task_key(scale: int, input_path: Path) -> str:
    return f"x{scale}:{str(input_path.resolve())}"


def detect_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("未安装 torch，无法运行超分辨率推理")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("已指定 CUDA，但当前环境不可用")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_cpu_memory_mb() -> float:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    except Exception:
        return 0.0


def get_gpu_memory_mb(device: str) -> Optional[float]:
    if device != "cuda":
        return None
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round(memory.used / 1024 / 1024, 2)
    except Exception:
        return None


def resolve_target_mode(input_mode: str) -> str:
    if input_mode in {"1", "L", "I", "F", "I;16", "I;16B", "I;16L", "P", "LA"}:
        return "L"
    return "RGB"


def normalize_input_for_model(input_image: Image.Image) -> Tuple[Image.Image, str]:
    target_mode = resolve_target_mode(input_image.mode)
    if target_mode == "L":
        normalized = input_image.convert("L").convert("RGB")
    else:
        normalized = input_image.convert("RGB")
    return normalized, target_mode


def save_image_with_quality(
    output_image: Image.Image, target_mode: str, output_path: Path, quality: int
) -> None:
    save_kwargs: Dict[str, int] = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        if target_mode == "L":
            if output_image.mode != "L":
                output_image = output_image.convert("L")
        else:
            if output_image.mode != "RGB":
                output_image = output_image.convert("RGB")
        save_kwargs["quality"] = quality
    elif output_image.mode != target_mode:
        try:
            output_image = output_image.convert(target_mode)
        except Exception:
            pass
    output_image.save(output_path, **save_kwargs)


def run_batch(config: BatchConfig) -> BatchReport:
    if RealESRGAN is None:
        raise RuntimeError("未安装 py-real-esrgan，无法加载 RealESRGAN 模型")
    if not config.input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {config.input_dir}")
    if not config.input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {config.input_dir}")
    if any(scale not in {2, 4, 8} for scale in config.scales):
        raise ValueError("仅支持超分倍数: 2, 4, 8")

    logger = setup_logger(config.output_dir)
    device_name = detect_device(config.device)
    logger.info("使用设备: %s", device_name)
    logger.info("输入目录: %s", config.input_dir)
    logger.info("输出目录: %s", config.output_dir)
    logger.info("处理倍数: %s", config.scales)

    image_paths = discover_images(config.input_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"输入目录中未找到图像文件，支持格式: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    progress = ProgressStore(config.output_dir / config.progress_file)
    worker_count = config.workers
    if device_name == "cuda" and worker_count > 1:
        logger.warning("CUDA 模式建议 workers=1，已自动调整为 1")
        worker_count = 1
    worker_count = max(1, worker_count)

    started_at = time.time()
    total_success = 0
    total_failed = 0
    total_skipped = 0
    per_scale_stats: List[ScaleStats] = []

    for scale in config.scales:
        scale_start = time.time()
        scale_stat = ScaleStats(scale=scale, total=len(image_paths))
        weight_path = resolve_weights_path(config.weights_dir, scale)
        logger.info("x%d 权重路径: %s", scale, weight_path)
        thread_local = threading.local()
        lock = threading.Lock()

        def get_model() -> RealESRGAN:
            model = getattr(thread_local, "model", None)
            if model is None:
                model = RealESRGAN(torch.device(device_name), scale=scale)
                model.load_weights(str(weight_path), download=True)
                thread_local.model = model
            return model

        def process_one(image_path: Path) -> Tuple[str, Optional[str]]:
            key = get_task_key(scale, image_path)
            out_path = create_output_path(config.output_dir, scale, image_path)
            if config.resume and out_path.exists():
                with lock:
                    scale_stat.skipped += 1
                progress.mark_processed(key, str(out_path))
                return "skipped", str(out_path)
            if config.resume and progress.is_processed(key) and out_path.exists():
                with lock:
                    scale_stat.skipped += 1
                return "skipped", str(out_path)
            try:
                model = get_model()
                with Image.open(image_path) as input_img:
                    source, target_mode = normalize_input_for_model(input_img)
                sr = model.predict(source)
                save_image_with_quality(sr, target_mode, out_path, config.quality)
                progress.mark_processed(key, str(out_path))
                with lock:
                    scale_stat.success += 1
                return "success", str(out_path)
            except (UnidentifiedImageError, OSError) as exc:
                err = f"图像格式错误或不可读: {exc}"
            except Exception as exc:
                err = str(exc)
            progress.mark_failed(key, err)
            with lock:
                scale_stat.failed += 1
            logger.error("处理失败 x%d %s: %s", scale, image_path, err)
            return "failed", None

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_one, path) for path in image_paths]
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"x{scale}",
                unit="img",
            ):
                pass

        scale_stat.duration_seconds = round(time.time() - scale_start, 4)
        total_success += scale_stat.success
        total_failed += scale_stat.failed
        total_skipped += scale_stat.skipped
        per_scale_stats.append(scale_stat)
        logger.info(
            "x%d 完成: success=%d failed=%d skipped=%d time=%.2fs",
            scale,
            scale_stat.success,
            scale_stat.failed,
            scale_stat.skipped,
            scale_stat.duration_seconds,
        )

    finished_at = time.time()
    report = BatchReport(
        started_at=started_at,
        finished_at=finished_at,
        total_seconds=round(finished_at - started_at, 4),
        total_images=len(image_paths) * len(config.scales),
        success=total_success,
        failed=total_failed,
        skipped=total_skipped,
        device=device_name,
        cpu_memory_mb=get_cpu_memory_mb(),
        gpu_memory_mb=get_gpu_memory_mb(device_name),
        per_scale=per_scale_stats,
    )
    report_dict = asdict(report)
    report_dict["per_scale"] = [asdict(stat) for stat in per_scale_stats]
    report_path = config.output_dir / "benchmark_report.json"
    report_path.write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "批量处理完成: total=%d success=%d failed=%d skipped=%d time=%.2fs",
        report.total_images,
        report.success,
        report.failed,
        report.skipped,
        report.total_seconds,
    )
    logger.info("基准报告: %s", report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-ESRGAN 批量超分辨率处理")
    parser.add_argument("input_dir", type=str, help="测试图像目录路径")
    parser.add_argument(
        "--output-dir", type=str, default="results", help='输出目录路径，默认 "results"'
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[4],
        choices=[2, 4, 8],
        help="超分倍数，可多选，例如: --scales 2 4 8",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="运行设备选择",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="并行线程数",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(Path(__file__).parent / "ckpts"),
        help="权重文件目录",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG 输出质量")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="启用断点续处理",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="禁用断点续处理",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--progress-file", type=str, default="progress.json", help="断点进度文件名"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BatchConfig(
        input_dir=Path(args.input_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        scales=args.scales,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        quality=args.quality,
        weights_dir=Path(args.weights_dir).resolve(),
        progress_file=args.progress_file,
    )
    try:
        run_batch(config)
        return 0
    except Exception as exc:
        print(f"批量处理失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
