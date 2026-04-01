# 网络不佳时可能需要开通代理
# ssh -R 8889:localhost:8889 user@server_IP
# export http_proxy=http://127.0.0.1:8889 && export https_proxy=http://127.0.0.1:8889

import argparse
import os
import shutil
from pathlib import Path
from typing import Any


TASK_SUFFIX = {
    "detect": ".pt",
    "seg": "-seg.pt",
    "pose": "-pose.pt",
    "cls": "-cls.pt",
    "obb": "-obb.pt",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_model_names(
    families: list[str], variants: list[str], tasks: list[str]
) -> list[str]:
    names: list[str] = []
    for family in families:
        for variant in variants:
            for task in tasks:
                suffix = TASK_SUFFIX[task]
                names.append(f"yolo{family}{variant}{suffix}")
    return names


def resolve_checkpoint_path(model: Any, fallback_name: str) -> Path:
    ckpt_path = getattr(model, "ckpt_path", None)
    if ckpt_path:
        return Path(ckpt_path)
    return Path(fallback_name)


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(
        description="批量下载 Ultralytics YOLO11/YOLO26 权重"
    )
    parser.add_argument(
        "--families", default="11,26", help="模型代际，逗号分隔，例如 11,26"
    )
    parser.add_argument(
        "--variants", default="n,s,m,l,x", help="模型尺寸，逗号分隔，例如 n,s,m,l,x"
    )
    parser.add_argument(
        "--tasks",
        default="obb",
        help="任务类型，支持 detect,seg,pose,cls,obb，逗号分隔",
    )
    parser.add_argument(
        "--output-dir", default="weights", help="下载后统一拷贝到该目录"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查 tasks 参数并展示将要下载的模型名，不实际下载",
    )
    args = parser.parse_args()

    families = parse_csv(args.families)
    variants = parse_csv(args.variants)
    tasks = parse_csv(args.tasks)

    invalid_tasks = [task for task in tasks if task not in TASK_SUFFIX]
    if invalid_tasks:
        raise ValueError(
            f"不支持的 tasks: {invalid_tasks}，可选: {list(TASK_SUFFIX.keys())}"
        )

    model_names = build_model_names(families, variants, tasks)
    if args.check_only:
        print(f"tasks 参数有效: {tasks}")
        print(f"将生成模型数: {len(model_names)}")
        for name in model_names:
            print(name)
        return

    from ultralytics import YOLO

    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    success: list[str] = []
    failed: list[str] = []

    print(f"Output dir: {output_dir}")
    print(f"Total models: {len(model_names)}")

    for name in model_names:
        print(f"[DOWNLOADING] {name}")
        try:
            model = YOLO(name)
            src = resolve_checkpoint_path(model, name)
            if not src.exists():
                raise FileNotFoundError(f"未找到下载结果: {src}")
            dst = output_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            success.append(name)
            print(f"[OK] {name} -> {dst}")
        except Exception as e:
            failed.append(name)
            print(f"[FAILED] {name}: {e}")

    print("")
    print(f"Done. success={len(success)}, failed={len(failed)}")
    if failed:
        print("Failed list:")
        for name in failed:
            print(f"- {name}")


if __name__ == "__main__":
    main()
