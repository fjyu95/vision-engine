# RealESRGAN

## 说明
- 推荐优先使用 `x2 / x4`：原生支持、速度更快、资源占用更低。
- `x8` 非必需：内存/显存/计算压力较大，且可能放大噪声或补出伪细节。

## 安装

```bash
# 推荐：sberbank-ai 版本，安装更简单，包含人脸优化
pip install py-real-esrgan
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git  # 同上

# Xintao原版，pypi和github上的最新版本都是0.3.0，python等版本都较老，不推荐使用
pip install realesrgan
```

PyPI: https://pypi.org/project/py-real-esrgan/

## 权重文件

```bash
git clone https://huggingface.co/ai-forever/Real-ESRGAN
# 或
git clone git@hf.co:ai-forever/Real-ESRGAN

hf download ai-forever/Real-ESRGAN --local-dir ckpts/
```

## 测试脚本用法

### 1) 单图测试脚本 `real_esrgan_demo.py`

```bash
python models/super-resolution/real_esrgan_demo.py
```

- 默认读取：`models/super-resolution/bridge_12.jpg`
- 默认权重：`models/super-resolution/ckpts/RealESRGAN_x4.pth`
- 默认输出：`models/super-resolution/results/sr_bridge_12.jpg`

### 2) 批量测试脚本 `test_batch_real_esrgan.py`

```bash
# 基础用法（x2/x4）
python models/super-resolution/test_batch_real_esrgan.py models/super-resolution/test_images --scales 2 4

# 指定输出目录
python models/super-resolution/test_batch_real_esrgan.py models/super-resolution/test_images --scales 2 4 --output-dir models/super-resolution/results_batch

# 指定设备（自动/CPU/CUDA）
python models/super-resolution/test_batch_real_esrgan.py models/super-resolution/test_images --scales 2 4 --device auto
```

- 支持格式：`.jpg .jpeg .png .bmp .tif .tiff`
- 输出目录结构：按倍率生成子目录，如 `results_batch/x2`、`results_batch/x4`
