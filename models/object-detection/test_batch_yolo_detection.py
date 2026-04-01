"""YOLO11/YOLO26 批量目标检测

运行示例：
1) 最基础（无需传参，默认 input_dir=./test_images）
   python test_batch_yolo_detection.py

2) 仅指定输入目录
   python test_batch_yolo_detection.py ./test_images

3) 指定输入和输出目录
   python test_batch_yolo_detection.py ./test_images --output-dir ./results

4) 指定模型代际与尺寸（OBB）
   python test_batch_yolo_detection.py ./test_images --families 11,26 --variants n,s --task obb --recursive

5) 自定义模型列表
   python test_batch_yolo_detection.py ./test_images --models yolo11n-obb.pt,yolo26n-obb.pt

6) 指定本地模型根目录
   python test_batch_yolo_detection.py ./test_images --model-root ./ckpts
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

try:
    import tifffile
except ImportError:
    tifffile = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TASK_SUFFIX = {
    "detect": ".pt",
    "seg": "-seg.pt",
    "pose": "-pose.pt",
    "cls": "-cls.pt",
    "obb": "-obb.pt",
}


@dataclass
class DetectionItem:
    model: str
    input_path: str
    output_path: Optional[str]
    detections: int
    imgsz_used: int
    retried: bool
    elapsed_seconds: float
    status: str
    error: Optional[str] = None


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_classes(value: Optional[str]) -> Optional[list[int]]:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def discover_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_model_names(families: list[str], variants: list[str], task: str) -> list[str]:
    suffix = TASK_SUFFIX[task]
    return [
        f"yolo{family}{variant}{suffix}" for family in families for variant in variants
    ]


def resolve_model_source(model_name: str, model_root: Path) -> str:
    model_path = Path(model_name)
    if model_path.is_absolute():
        return str(model_path)
    if "/" in model_name or "\\" in model_name:
        return str(model_path)
    candidate = (model_root / model_name).resolve()
    if candidate.exists():
        return str(candidate)
    return model_name


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


def to_uint8(img) -> "cv2.typing.MatLike":
    if img.dtype == "uint8":
        return img
    img_f = img.astype("float32")
    min_v = float(img_f.min())
    max_v = float(img_f.max())
    if max_v <= min_v:
        return (img_f * 0).astype("uint8")
    img_f = (img_f - min_v) / (max_v - min_v)
    return (img_f * 255.0).clip(0, 255).astype("uint8")


def read_for_inference(path: Path):
    suffix = path.suffix.lower()
    from_tifffile = False
    if suffix in {".tif", ".tiff"} and tifffile is not None:
        img = tifffile.imread(str(path))
        from_tifffile = True
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
        if from_tifffile:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        return None
    return to_uint8(img)


def round_up_to_32(value: int) -> int:
    return max(32, ((value + 31) // 32) * 32)


def choose_imgsz(img, target_imgsz: int, mode: str) -> int:
    if mode == "fixed":
        return int(target_imgsz)
    h, w = img.shape[:2]
    adaptive_size = round_up_to_32(max(h, w))
    return int(min(target_imgsz, adaptive_size))


def build_detection_matrix(
    items: list[DetectionItem], model_names: list[str], images: list[Path]
) -> tuple[list[str], list[dict[str, object]]]:
    image_keys = [str(p.resolve()) for p in images]
    image_labels = [p.name for p in images]
    lookup: dict[tuple[str, str], DetectionItem] = {}
    for item in items:
        lookup[(item.model, str(Path(item.input_path).resolve()))] = item
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        values: list[str] = []
        total = 0
        for key in image_keys:
            hit = lookup.get((model_name, key))
            if hit is None:
                values.append("-")
            elif hit.status == "success":
                values.append(str(hit.detections))
                total += int(hit.detections)
            elif hit.status == "skipped":
                values.append("S")
            else:
                values.append("E")
        rows.append({"model": model_name, "total": str(total), "values": values})
    return ["TOTAL", *image_labels], rows


def format_detection_matrix_lines(
    image_labels: list[str], rows: list[dict[str, object]]
) -> list[str]:
    if not image_labels or not rows:
        return []
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if "total" in row:
            row_values = [str(row["total"]), *list(row["values"])]
        else:
            row_values = list(row["values"])
        normalized_rows.append({"model": row["model"], "values": row_values})
    model_width = max(len("MODEL"), max(len(str(r["model"])) for r in rows))
    col_widths: list[int] = []
    for idx, label in enumerate(image_labels):
        max_val_len = max(len(str(r["values"][idx])) for r in normalized_rows)
        col_widths.append(max(len(label), max_val_len, 3))
    header = "MODEL".ljust(model_width)
    for idx, label in enumerate(image_labels):
        header += " | " + label.ljust(col_widths[idx])
    sep = "-" * len(header)
    lines = [header, sep]
    for row in normalized_rows:
        line = str(row["model"]).ljust(model_width)
        values = row["values"]
        for idx in range(len(values)):
            line += " | " + str(values[idx]).ljust(col_widths[idx])
        lines.append(line)
    return lines


def build_timing_matrix(
    items: list[DetectionItem], model_names: list[str], images: list[Path]
) -> tuple[list[str], list[dict[str, object]]]:
    image_keys = [str(p.resolve()) for p in images]
    image_labels = [p.name for p in images]
    lookup: dict[tuple[str, str], DetectionItem] = {}
    for item in items:
        lookup[(item.model, str(Path(item.input_path).resolve()))] = item
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        values: list[str] = []
        for key in image_keys:
            hit = lookup.get((model_name, key))
            if hit is None:
                values.append("-")
            elif hit.status == "success":
                values.append(f"{hit.elapsed_seconds:.4f}")
            elif hit.status == "skipped":
                values.append("S")
            else:
                values.append("E")
        rows.append({"model": model_name, "values": values})
    return image_labels, rows


def format_matrix_lines(
    title: str, row_header: str, col_labels: list[str], rows: list[dict[str, object]]
) -> list[str]:
    if not col_labels or not rows:
        return []
    row_width = max(len(row_header), max(len(str(r["model"])) for r in rows))
    col_widths: list[int] = []
    for idx, label in enumerate(col_labels):
        max_val_len = max(len(str(r["values"][idx])) for r in rows)
        col_widths.append(max(len(label), max_val_len, 3))
    header = row_header.ljust(row_width)
    for idx, label in enumerate(col_labels):
        header += " | " + label.ljust(col_widths[idx])
    sep = "-" * len(header)
    lines = [title, header, sep]
    for row in rows:
        line = str(row["model"]).ljust(row_width)
        values = row["values"]
        for idx in range(len(values)):
            line += " | " + str(values[idx]).ljust(col_widths[idx])
        lines.append(line)
    return lines


def format_model_timing_summary(
    items: list[DetectionItem], model_names: list[str]
) -> list[str]:
    lines = ["Per-model timing summary:"]
    headers = [
        "MODEL",
        "RUNS",
        "SUCCESS",
        "FAILED",
        "SKIPPED",
        "TOTAL(s)",
        "AVG(s)",
        "MIN(s)",
        "MAX(s)",
    ]
    grouped: dict[str, list[DetectionItem]] = {m: [] for m in model_names}
    for item in items:
        grouped.setdefault(item.model, []).append(item)
    rows: list[list[str]] = []
    for model_name in model_names:
        model_items = grouped.get(model_name, [])
        runs = len(model_items)
        success = sum(1 for i in model_items if i.status == "success")
        failed = sum(1 for i in model_items if i.status == "failed")
        skipped = sum(1 for i in model_items if i.status == "skipped")
        times = [i.elapsed_seconds for i in model_items]
        total_t = sum(times)
        avg_t = (total_t / runs) if runs else 0.0
        min_t = min(times) if times else 0.0
        max_t = max(times) if times else 0.0
        rows.append(
            [
                model_name,
                str(runs),
                str(success),
                str(failed),
                str(skipped),
                f"{total_t:.4f}",
                f"{avg_t:.4f}",
                f"{min_t:.4f}",
                f"{max_t:.4f}",
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, v in enumerate(row):
            widths[idx] = max(widths[idx], len(v))
    header_line = " | ".join(
        headers[idx].ljust(widths[idx]) for idx in range(len(headers))
    )
    sep_line = "-" * len(header_line)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))
    return lines


def format_image_timing_summary(
    items: list[DetectionItem], images: list[Path]
) -> list[str]:
    lines = ["Per-image timing summary:"]
    headers = [
        "IMAGE",
        "RUNS",
        "SUCCESS",
        "FAILED",
        "SKIPPED",
        "TOTAL(s)",
        "AVG(s)",
        "MIN(s)",
        "MAX(s)",
    ]
    grouped: dict[str, list[DetectionItem]] = {}
    for item in items:
        grouped.setdefault(str(Path(item.input_path).resolve()), []).append(item)
    rows: list[list[str]] = []
    for img in images:
        key = str(img.resolve())
        img_items = grouped.get(key, [])
        runs = len(img_items)
        success = sum(1 for i in img_items if i.status == "success")
        failed = sum(1 for i in img_items if i.status == "failed")
        skipped = sum(1 for i in img_items if i.status == "skipped")
        times = [i.elapsed_seconds for i in img_items]
        total_t = sum(times)
        avg_t = (total_t / runs) if runs else 0.0
        min_t = min(times) if times else 0.0
        max_t = max(times) if times else 0.0
        rows.append(
            [
                img.name,
                str(runs),
                str(success),
                str(failed),
                str(skipped),
                f"{total_t:.4f}",
                f"{avg_t:.4f}",
                f"{min_t:.4f}",
                f"{max_t:.4f}",
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, v in enumerate(row):
            widths[idx] = max(widths[idx], len(v))
    header_line = " | ".join(
        headers[idx].ljust(widths[idx]) for idx in range(len(headers))
    )
    sep_line = "-" * len(header_line)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))
    return lines


def build_output_path(
    src: Path,
    input_root: Path,
    output_root: Path,
    model_name: str,
    flat: bool,
    suffix: str,
) -> Path:
    model_dir = output_root / model_name.replace(".pt", "")
    if flat:
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{src.stem}{suffix}{src.suffix}"
    rel_parent = src.parent.relative_to(input_root)
    target_dir = model_dir / rel_parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{src.stem}{suffix}{src.suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11/YOLO26 批量目标检测",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  python test_batch_yolo_detection.py\n"
            "  python test_batch_yolo_detection.py ./test_images\n"
            "  python test_batch_yolo_detection.py ./test_images --output-dir ./results\n"
            "  python test_batch_yolo_detection.py ./test_images --families 11,26 --variants n,s --task obb --recursive\n"
            "  python test_batch_yolo_detection.py ./test_images --models yolo11n-obb.pt,yolo26n-obb.pt\n"
            "  python test_batch_yolo_detection.py ./test_images --model-root ./ckpts"
        ),
    )
    parser.add_argument(
        "input_dir", nargs="?", default="./test_images", type=str, help="输入目录"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results_batch_yolo", help="输出目录"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="obb",
        choices=["detect", "seg", "pose", "cls", "obb"],
        help="任务类型",
    )
    parser.add_argument(
        "--families", type=str, default="11,26", help="模型代际，例如 11,26"
    )
    parser.add_argument(
        "--variants", type=str, default="n,s,m,l,x", help="模型尺寸，例如 n,s,m,l,x"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="自定义模型列表，逗号分隔，若传入则覆盖 families/variants/task",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="./ckpts",
        help="模型根目录，默认 ./ckpts；当模型名非路径时优先从该目录拼接加载",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="推理输入尺寸")
    parser.add_argument(
        "--imgsz-mode",
        type=str,
        default="adaptive",
        choices=["fixed", "adaptive"],
        help="imgsz 使用模式：fixed=所有图统一尺寸，adaptive=小图不强制放大到 imgsz",
    )
    parser.add_argument(
        "--retry-imgsz",
        type=int,
        default=0,
        help="首轮无目标时可用更大尺寸重试（0 表示关闭）",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=300, help="单图最大检测数量")
    parser.add_argument(
        "--device", type=str, default="0", help="推理设备，如 cpu、0、0,1"
    )
    parser.add_argument(
        "--half", dest="half", action="store_true", help="开启 FP16 半精度推理"
    )
    parser.add_argument(
        "--no-half", dest="half", action="store_false", help="关闭 FP16 推理"
    )
    parser.set_defaults(half=True)
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="按类别 ID 过滤，逗号分隔，例如 0,1,5",
    )
    parser.add_argument("--recursive", action="store_true", help="递归扫描输入目录")
    parser.add_argument("--resume", action="store_true", help="输出已存在则跳过")
    parser.add_argument("--flat", action="store_true", help="输出目录不保留原始层级")
    parser.add_argument("--suffix", type=str, default="_detected", help="输出文件后缀")
    parser.add_argument(
        "--opencv-log-level",
        type=str,
        default="ERROR",
        choices=["SILENT", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="OpenCV 日志级别，用于抑制 TIFF 警告",
    )
    return parser.parse_args()


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)

    args = parse_args()
    set_opencv_log_level(args.opencv_log_level)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_root = Path(args.model_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {input_dir}")

    classes = parse_classes(args.classes)
    images = discover_images(input_dir, args.recursive)
    if not images:
        raise FileNotFoundError(f"未发现可处理图像: {input_dir}")

    if args.models:
        model_names = parse_csv(args.models)
    else:
        families = parse_csv(args.families)
        variants = parse_csv(args.variants)
        model_names = build_model_names(families, variants, args.task)

    items: list[DetectionItem] = []
    total_start = time.perf_counter()

    print(f"Images: {len(images)}")
    print(f"Models: {len(model_names)}")
    print(f"Output dir: {output_dir}")
    print(f"Model root: {model_root}")

    for model_name in model_names:
        print(f"\n[MODEL] {model_name}")
        model_source = resolve_model_source(model_name, model_root)
        print(f"Model source: {model_source}")
        model = YOLO(model_source)
        print(f"Model names: {model.names}")
        for src in images:
            dst = build_output_path(
                src=src,
                input_root=input_dir,
                output_root=output_dir,
                model_name=model_name,
                flat=args.flat,
                suffix=args.suffix,
            )
            if args.resume and dst.exists():
                items.append(
                    DetectionItem(
                        model=model_name,
                        input_path=str(src),
                        output_path=str(dst),
                        detections=0,
                        imgsz_used=0,
                        retried=False,
                        elapsed_seconds=0.0,
                        status="skipped",
                    )
                )
                print(f"[SKIPPED] {model_name} | {src.name} | elapsed=0.0000s")
                continue
            start = time.perf_counter()
            try:
                infer_img = read_for_inference(src)
                if infer_img is None:
                    raise RuntimeError(f"输入图像不可读或通道不支持: {src}")
                imgsz_used = choose_imgsz(infer_img, args.imgsz, args.imgsz_mode)
                retried = False
                results = model(
                    infer_img,
                    imgsz=imgsz_used,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    device=args.device,
                    half=args.half,
                    classes=classes,
                    verbose=False,
                )
                if not results:
                    raise RuntimeError("未获得推理结果")
                result = results[0]
                if result.obb is not None:
                    detections = len(result.obb)
                elif result.boxes is not None:
                    detections = len(result.boxes)
                else:
                    detections = 0
                if (
                    detections == 0
                    and args.retry_imgsz > 0
                    and args.retry_imgsz != imgsz_used
                ):
                    retry_results = model(
                        infer_img,
                        imgsz=args.retry_imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        max_det=args.max_det,
                        device=args.device,
                        half=args.half,
                        classes=classes,
                        verbose=False,
                    )
                    if retry_results:
                        retry_result = retry_results[0]
                        if retry_result.obb is not None:
                            retry_detections = len(retry_result.obb)
                        elif retry_result.boxes is not None:
                            retry_detections = len(retry_result.boxes)
                        else:
                            retry_detections = 0
                        if retry_detections >= detections:
                            result = retry_result
                            detections = retry_detections
                            imgsz_used = args.retry_imgsz
                            retried = True
                plotted = result.plot()
                ok = cv2.imwrite(str(dst), plotted)
                if not ok:
                    raise RuntimeError(f"写入失败: {dst}")
                elapsed = time.perf_counter() - start
                items.append(
                    DetectionItem(
                        model=model_name,
                        input_path=str(src),
                        output_path=str(dst),
                        detections=detections,
                        imgsz_used=imgsz_used,
                        retried=retried,
                        elapsed_seconds=elapsed,
                        status="success",
                    )
                )
                print(
                    f"[OK] {model_name} | {src.name} | det={detections} | imgsz={imgsz_used} | retried={retried} | elapsed={elapsed:.4f}s"
                )
            except Exception as e:
                elapsed = time.perf_counter() - start
                items.append(
                    DetectionItem(
                        model=model_name,
                        input_path=str(src),
                        output_path=None,
                        detections=0,
                        imgsz_used=0,
                        retried=False,
                        elapsed_seconds=elapsed,
                        status="failed",
                        error=str(e),
                    )
                )
                print(
                    f"[FAILED] {model_name} | {src.name} | elapsed={elapsed:.4f}s | error={e}"
                )

    total_elapsed = time.perf_counter() - total_start
    success = sum(1 for i in items if i.status == "success")
    failed = sum(1 for i in items if i.status == "failed")
    skipped = sum(1 for i in items if i.status == "skipped")
    total_detections = sum(i.detections for i in items if i.status == "success")
    matrix_images, matrix_rows = build_detection_matrix(items, model_names, images)
    matrix_aligned_lines = format_detection_matrix_lines(matrix_images, matrix_rows)
    timing_images, timing_rows = build_timing_matrix(items, model_names, images)
    timing_aligned_lines = format_matrix_lines(
        title="Per-image inference time by model (seconds):",
        row_header="MODEL",
        col_labels=timing_images,
        rows=timing_rows,
    )
    model_timing_summary_lines = format_model_timing_summary(items, model_names)
    image_timing_summary_lines = format_image_timing_summary(items, images)
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "task": args.task,
        "models": model_names,
        "images": len(images),
        "runs": len(items),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "total_detections": total_detections,
        "total_seconds": total_elapsed,
        "avg_seconds_per_run": (total_elapsed / len(items)) if items else 0.0,
        "detection_matrix_images": matrix_images,
        "detection_matrix_aligned_lines": matrix_aligned_lines,
        "timing_matrix_images": timing_images,
        "timing_matrix_aligned_lines": timing_aligned_lines,
        "model_timing_summary_lines": model_timing_summary_lines,
        "image_timing_summary_lines": image_timing_summary_lines,
        "items": [asdict(i) for i in items],
    }

    report_path = output_dir / "yolo_batch_detection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nDone.")
    print(f"Success: {success}, Failed: {failed}, Skipped: {skipped}")
    print(f"Total detections: {total_detections}")
    print(f"Elapsed: {total_elapsed:.2f}s")
    print("")
    for line in model_timing_summary_lines:
        print(line)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
