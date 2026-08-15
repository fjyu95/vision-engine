#!/usr/bin/env python3
"""
将 examples/sample_images 下的所有图片解析为普通 8-bit 图像，
统一保存到 examples/sample_images_8bit/，不保留子目录结构。

依赖：
    - raw_utils.cp38-win_amd64.pyd（位于 examples/lib/，仅 Windows 下可用）
    - numpy
    - imageio / opencv-python / Pillow 任一即可读取 TIFF
"""
import sys
from pathlib import Path

import numpy as np

# 让 raw_utils 可导入
LIB_DIR = Path(__file__).resolve().parent / 'lib'
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import raw_utils


SAMPLE_DIR = Path(__file__).resolve().parent / 'sample_images'
OUTPUT_DIR = Path(__file__).resolve().parent / 'sample_images_8bit'
OUTPUT_DIR.mkdir(exist_ok=True)

# 需要处理的图片后缀（小写）
IMAGE_EXTS = {'.raw', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def to_u8(img, p_low=1, p_high=99):
    """按百分位拉伸到 0-255，便于显示（不改变原始数据）。"""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


_reader_cache = {}


def get_reader(ctype):
    """按相机类型复用 RawLoader 实例。"""
    if ctype not in _reader_cache:
        LoaderCls, CamInfoCls = raw_utils.get_loader_caminfo(ctype)
        _reader_cache[ctype] = (LoaderCls(), CamInfoCls)
    return _reader_cache[ctype]


def read_raw(path: Path) -> np.ndarray:
    """使用 raw_utils 读取 .raw 图像。"""
    ctype = raw_utils.detect_camera_type(str(path))
    loader, _ = get_reader(ctype)
    loader.load(str(path))
    return loader.get_raw_img()


def _try_imageio(path: Path):
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(path))
    except Exception:
        return None


def _try_cv2(path: Path):
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        # OpenCV 彩色图为 BGR，转成 RGB/RGBA
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img
    except Exception:
        return None


def _try_pil(path: Path):
    try:
        from PIL import Image
        return np.asarray(Image.open(path))
    except Exception:
        return None


def read_common_image(path: Path) -> np.ndarray:
    """用 imageio / cv2 / PIL 读取普通图像文件。"""
    for reader in (_try_imageio, _try_cv2, _try_pil):
        img = reader(path)
        if img is not None:
            return img
    raise RuntimeError(f'无法读取图像：{path}')


def normalize_shape(img: np.ndarray) -> np.ndarray:
    """统一为 (H, W) 或 (H, W, C)，通道置于最后一维。"""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f'不支持的图像维度：{arr.ndim}')

    h, w = arr.shape[0], arr.shape[1]
    c = arr.shape[2]

    # 常见 channel-last：最后一维是 1/3/4，且 H, W 明显大于通道数
    if c in (1, 3, 4) and h > c and w > c:
        if c == 1:
            return arr[..., 0]
        return arr

    # channel-first：(C, H, W)，C 为 1/3/4
    c_first = arr.shape[0]
    if c_first in (1, 3, 4) and c_first < h and c_first < w:
        if c_first == 1:
            return arr[0]
        return np.transpose(arr, (1, 2, 0))

    # 兜底：假设第一维为波段，取前 3 个作为 RGB，或单波段
    bands = min(c_first, 3)
    if bands == 1:
        return arr[0]
    return np.transpose(arr[:bands], (1, 2, 0))


def flat_output_name(rel_path: Path) -> str:
    """生成扁平输出文件名，包含原始扩展名标记以保证唯一性。"""
    parts = list(rel_path.with_suffix('').parts)
    src_ext = rel_path.suffix.lstrip('.').lower()
    stem = '_'.join(parts) + f'_{src_ext}'
    # 防止过长文件名
    if len(stem) > 200:
        stem = stem[:200]
    return stem + '.png'


def write_png(path: Path, img_u8: np.ndarray):
    """写 8-bit PNG，优先 imageio，其次 cv2 / PIL。"""
    last_err = None
    try:
        import imageio.v3 as iio
        iio.imwrite(path, img_u8)
        return
    except Exception as e:
        last_err = e

    if img_u8.ndim == 3 and img_u8.shape[-1] == 3:
        # cv2 需要 BGR
        try:
            import cv2
            cv2.imwrite(str(path), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
            return
        except Exception:
            pass
    elif img_u8.ndim == 3 and img_u8.shape[-1] == 4:
        try:
            import cv2
            cv2.imwrite(str(path), cv2.cvtColor(img_u8, cv2.COLOR_RGBA2BGRA))
            return
        except Exception:
            pass
    else:
        try:
            import cv2
            cv2.imwrite(str(path), img_u8)
            return
        except Exception:
            pass

    try:
        from PIL import Image
        Image.fromarray(img_u8).save(path)
        return
    except Exception as e:
        last_err = e

    raise RuntimeError(f'无法写入 {path}：{last_err}')


def main():
    if not SAMPLE_DIR.exists():
        raise FileNotFoundError(f'样例目录不存在：{SAMPLE_DIR}')

    # 收集所有图片文件
    files = []
    for fp in sorted(SAMPLE_DIR.rglob('*')):
        if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
            files.append(fp)

    if not files:
        print(f'未在 {SAMPLE_DIR} 下找到图片')
        return

    print(f'共发现 {len(files)} 张图片，输出目录：{OUTPUT_DIR}')

    for fp in files:
        rel = fp.relative_to(SAMPLE_DIR)
        out_name = flat_output_name(rel)
        out_path = OUTPUT_DIR / out_name

        try:
            if fp.suffix.lower() == '.raw':
                img = read_raw(fp)
            else:
                img = read_common_image(fp)

            img = normalize_shape(img)
            img_u8 = to_u8(img)
            write_png(out_path, img_u8)

            print(f'  已保存：{out_name}  ({img_u8.shape}, {img_u8.dtype})')
        except Exception as e:
            print(f'  失败：{rel} -> {e}')

    print('处理完成。')


if __name__ == '__main__':
    main()