# vision-engine

A high-performance vision inference engine with pipeline orchestration, multi-GPU scheduling, and modular algorithm integration.

## 项目目标

`vision-engine` 面向可扩展视觉算法工程化，采用模块化组织方式，支持按任务快速接入、验证和批量处理，当前包含：

- 目标检测（Object Detection）
- 图像去噪（Denoise）
- 超分辨率（Super Resolution）
- 切片压缩（Tiling Compress）
- 模板模块（Template，用于新算法快速接入）

## 目录结构

```text
vision-engine/
├── models/
│   ├── denoise/
│   ├── object-detection/
│   ├── super-resolution/
│   ├── tiling-compress/
│   └── template/
├── pyproject.toml
├── setup.py
├── requirements.txt
└── README.md
```

各算法目录通常包含：

- `README.md`：模块说明与使用方式
- `requirements.txt`：模块依赖
- `*-demo.py` / `test_batch_*.py`：单图与批处理示例
- `algorithms.md`：模型/算法选型与说明（按模块提供）

## 环境准备

当前推荐安装流程：

```bash
conda create -n vision-engine python=3.10
conda activate vision-engine
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

兼容性备注：

```text
torch260+cu124+flash273
torch280+cu128+flash283
```

## 使用方式（初始版）

建议按模块进入对应目录运行示例脚本，例如：

```bash
cd models/object-detection
python yolo_detection-demo.py
```

或执行批处理脚本（以模块内 README 为准）。

## 开发约定（初始）

- 新增算法优先放入 `models/<task-name>/`
- 目录内保持统一结构：`README + requirements + demo + batch script`
- 模块依赖尽量局部化，避免影响全局环境
- 先保证可运行，再逐步补充性能优化与工程化接口
