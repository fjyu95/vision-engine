"""BM3D 批量去噪脚本

用法示例：
1) 基础批处理（默认不加噪）
   python test_batch_bm3d_denoise.py ./test_images --output-dir ./results_bm3d --recursive

2) 先加噪再去噪
   python test_batch_bm3d_denoise.py ./test_images --output-dir ./results_bm3d_noisy --add-noise --sigma 25 --recursive

3) 多进程与断点续跑
   python test_batch_bm3d_denoise.py ./test_images --output-dir ./results_bm3d --workers 4 --resume --recursive

说明：
- 默认不加噪，只有传入 --add-noise 才会人工加高斯噪声。
- 自动支持彩色/灰度输入，输出保持原扩展名与原始数据类型（uint8/uint16/float）。
- 结果会写入 bm3d_batch_report.json，包含成功/失败/耗时统计。
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import bm3d
import cv2
from skimage.restoration import estimate_sigma
from skimage.util import random_noise
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    tifffile = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class TaskResult:
    input_path: str
    output_path: Optional[str]
    status: str
    elapsed_seconds: float
    error: Optional[str] = None


@dataclass
class BatchReport:
    total: int
    success: int
    failed: int
    skipped: int
    workers: int
    add_noise: bool
    sigma: Optional[float]
    total_seconds: float
    avg_seconds_per_file: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM3D 批量去噪（多进程）",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  python test_batch_bm3d_denoise.py ./test_images --output-dir ./results --recursive\n"
            "  python test_batch_bm3d_denoise.py ./test_images --output-dir ./results --add-noise --sigma 25 --recursive\n"
            "  python test_batch_bm3d_denoise.py ./test_images --output-dir ./results --workers 4 --resume --recursive"
        ),
    )
    parser.add_argument("input_dir", type=str, help="输入目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="进程数"
    )
    parser.add_argument(
        "--sigma", type=float, default=None, help="噪声标准差（0-255），不传则自动估计"
    )
    parser.add_argument(
        "--add-noise", action="store_true", help="是否先人工加噪，默认不加"
    )
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
    parser.add_argument(
        "--suffix", type=str, default="_denoised_bm3d", help="输出文件后缀"
    )
    parser.add_argument("--resume", action="store_true", help="若输出已存在则跳过")
    parser.add_argument("--flat", action="store_true", help="输出目录不保留原始层级")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument(
        "--opencv-log-level",
        type=str,
        default="ERROR",
        choices=["SILENT", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="OpenCV日志级别",
    )
    return parser.parse_args()


def discover_images(input_dir: Path, recursive: bool) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = [
        p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images)


def infer_scale_and_dtype(img) -> Tuple[float, str]:
    if img.dtype == "uint8":
        return 255.0, "uint8"
    if img.dtype == "uint16":
        return 65535.0, "uint16"
    return 1.0, "float32"


def to_float01(img, scale: float):
    img_f = img.astype("float32")
    if scale > 1.0:
        img_f = img_f / scale
    return img_f.clip(0.0, 1.0)


def to_original_dtype(img_f, dtype_name: str):
    img_f = img_f.clip(0.0, 1.0)
    if dtype_name == "uint16":
        return (img_f * 65535.0).round().astype("uint16")
    if dtype_name == "uint8":
        return (img_f * 255.0).round().astype("uint8")
    return img_f.astype("float32")


def build_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    suffix: str,
    flat: bool,
) -> Path:
    if flat:
        target_dir = output_root
    else:
        rel_parent = input_path.parent.relative_to(input_root)
        target_dir = output_root / rel_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def set_opencv_log_level(level: str) -> None:
    level_map = {
        "SILENT": 0,
        "FATAL": 1,
        "ERROR": 2,
        "WARNING": 3,
        "INFO": 4,
        "DEBUG": 5,
    }
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(level_map.get(level, 2))


def read_image(path: Path):
    if path.suffix.lower() in {".tif", ".tiff"} and tifffile is not None:
        img = tifffile.imread(str(path))
        if img is None:
            return None, "UNKNOWN"
        if img.ndim == 3 and img.shape[2] == 3:
            return img, "RGB"
        return img, "GRAY"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, "UNKNOWN"
    if img.ndim == 3 and img.shape[2] == 3:
        return img, "BGR"
    return img, "GRAY"


def write_image(path: Path, img, color_order: str) -> bool:
    suffix = path.suffix.lower()
    output = img
    if suffix in {".tif", ".tiff"} and tifffile is not None:
        if output.ndim == 3 and output.shape[2] == 3 and color_order == "BGR":
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        tifffile.imwrite(str(path), output)
        return True
    if output.ndim == 3 and output.shape[2] == 3 and color_order == "RGB":
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(path), output)


def process_one(
    input_path: str,
    input_root: str,
    output_root: str,
    sigma: Optional[float],
    add_noise: bool,
    average_sigmas: bool,
    suffix: str,
    resume: bool,
    flat: bool,
    opencv_log_level: str,
) -> TaskResult:
    start = time.perf_counter()
    src = Path(input_path)
    in_root = Path(input_root)
    out_root = Path(output_root)
    dst = build_output_path(src, in_root, out_root, suffix, flat)

    if resume and dst.exists():
        return TaskResult(
            str(src), str(dst), "skipped", time.perf_counter() - start, None
        )

    set_opencv_log_level(opencv_log_level)
    img, color_order = read_image(src)
    if img is None:
        return TaskResult(
            str(src), None, "failed", time.perf_counter() - start, "输入图像不可读"
        )

    is_gray = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
    working_color_order = color_order
    if not is_gray and color_order == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        working_color_order = "RGB"
    scale, dtype_name = infer_scale_and_dtype(img)
    img_f = to_float01(img, scale)

    if add_noise:
        noise_sigma = 25.0 if sigma is None else float(sigma)
        var = (noise_sigma / 255.0) ** 2
        denoise_input = random_noise(img_f, mode="gaussian", var=var)
    else:
        denoise_input = img_f

    try:
        if sigma is None:
            channel_axis = -1 if denoise_input.ndim == 3 else None
            sigma_est = estimate_sigma(
                denoise_input,
                channel_axis=channel_axis,
                average_sigmas=average_sigmas,
            )
            if isinstance(sigma_est, (list, tuple)):
                sigma_psd = [float(v) for v in sigma_est]
            else:
                sigma_psd = float(sigma_est)
        else:
            sigma_psd = float(sigma) / 255.0
        denoised = bm3d.bm3d(denoise_input, sigma_psd=sigma_psd)
        if is_gray and denoised.ndim == 3:
            denoised = cv2.cvtColor(denoised.astype("float32"), cv2.COLOR_RGB2GRAY)
        denoised_out = to_original_dtype(denoised, dtype_name)
        ok = write_image(dst, denoised_out, working_color_order)
        if not ok:
            return TaskResult(
                str(src),
                str(dst),
                "failed",
                time.perf_counter() - start,
                "输出写入失败",
            )
    except Exception as e:
        return TaskResult(
            str(src), str(dst), "failed", time.perf_counter() - start, str(e)
        )

    return TaskResult(str(src), str(dst), "success", time.perf_counter() - start, None)


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

    images = discover_images(input_dir, args.recursive)
    # images = images[::2]
    images = images[:10]
    if not images:
        raise FileNotFoundError(
            f"未发现可处理图像，支持: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    started = time.perf_counter()
    success = failed = skipped = 0
    results: List[TaskResult] = []

    workers = max(1, int(args.workers))
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(
                process_one,
                str(p),
                str(input_dir),
                str(output_dir),
                args.sigma,
                bool(args.add_noise),
                bool(args.average_sigmas),
                str(args.suffix),
                bool(args.resume),
                bool(args.flat),
                str(args.opencv_log_level),
            )
            for p in images
        ]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="BM3D", unit="img"
        ):
            res = fut.result()
            results.append(res)
            if res.status == "success":
                success += 1
            elif res.status == "failed":
                failed += 1
            else:
                skipped += 1

    total_seconds = time.perf_counter() - started
    report = BatchReport(
        total=len(images),
        success=success,
        failed=failed,
        skipped=skipped,
        workers=workers,
        add_noise=bool(args.add_noise),
        sigma=args.sigma,
        total_seconds=total_seconds,
        avg_seconds_per_file=total_seconds / max(1, len(images)),
    )

    failed_items = [asdict(r) for r in results if r.status == "failed"]
    summary = {
        "report": asdict(report),
        "failed_items": failed_items,
    }
    report_path = output_dir / "bm3d_batch_report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(
        f"完成: total={report.total}, success={report.success}, "
        f"failed={report.failed}, skipped={report.skipped}, "
        f"time={report.total_seconds:.4f}s"
    )
    print(f"报告: {report_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
