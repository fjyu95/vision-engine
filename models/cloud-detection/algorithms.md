# 航空/遥感云雾检测方案整合（保留原始有效信息）

## 1. 背景与结论

对于航空遥感影像，目前没有“绝对开箱即用且对所有传感器完美泛化”的通用预训练模型。多数开源方案最初面向 Sentinel-2 / Landsat 等卫星数据，但已有若干工具可直接用于高分辨率航空影像推理，并可通过微调显著提升效果。

## 2. 工具总览（整合）

| 工具 | 核心定位 | 安装方式 | 推荐指数 | 关键说明 |
|---|---|---|---|---|
| **DTACSNet** | 卫星影像云检测+大气校正 | `pip install dtacs` | ⭐⭐⭐⭐⭐ | ESA/NASA 相关项目，提供 `CDModel`，支持 `RGBNIR` 预训练模型，上手快 |
| **Fmask v5.0** | 经典物理规则+机器学习增强 | 源码安装（无 PyPI） | ⭐⭐⭐⭐ | 经典且稳健，含 UPL（UNet + LightGBM），但依赖数据包重 |
| **vhr-cloudmask (NASA)** | 面向高分辨率航空/无人机云+云影分割 | `pip install vhr-cloudmask` | ⭐⭐⭐⭐⭐ | 专为 VHR 场景，支持大图分块、GPU/CPU |
| **HRC_WHU 系模型** | 基于高分辨率数据集的 U-Net/Transformer 方案 | 参考仓库训练/推理 | ⭐⭐⭐⭐ | 薄云边界细节好，适合二次训练 |
| **OmniCloudMask** | 通用高分辨率 RGB/NIR 云+云影检测 | 参考仓库 | ⭐⭐⭐⭐ | 实用路线，推理速度较好 |
| **smp 自定义模型** | U-Net/DeepLabV3+/Transformer 灵活构建 | `pip install segmentation-models-pytorch` | ⭐⭐⭐⭐ | 最灵活，适合按传感器定制 |

## 3. 分场景建议

### 3.1 快速试用（最小工程成本）

1. 先试 `vhr-cloudmask`（高分辨率航空场景优先）  
2. 可并行试 `DTACSNet`（若有 `RGBNIR`）  
3. 对比输出云掩膜后决定是否微调

### 3.2 稳健备选（物理规则增强）

- `Fmask v5.0` 作为备选：泛化思路强，但工程门槛更高（无 PyPI、需辅助数据包、航空数据适配成本高）。

## 4. 安装与最小示例

### 4.1 DTACSNet

```bash
pip install dtacs
```

```python
from dtacs.model_wrapper import CDModel

model = CDModel(model_name="DTACS_CLOUD_RGBNIR")
model.load_weights()
# 输入通常为 (bands, height, width)，波段顺序建议 [R, G, B, NIR]
cloud_mask = model.predict(your_image_array)
```

说明：DTACS 预训练数据主要来自 Sentinel-2，直接迁移到低空无人机影像时可能有域偏移。

### 4.2 Fmask 5.0（源码）

```bash
git clone https://atomgit.com/gh_mirrors/fm/Fmask4.git
cd Fmask4
pip install -r requirements.txt
# 需下载约 3GB 辅助数据包（DEM 等）
python fmask.py --imagepath /path/to/your/geotiff_image --model UPL
```

航空数据适配注意：
- 需组织成其所需波段命名/格式
- 缺少 SWIR 等关键波段时可行性下降

### 4.3 vhr-cloudmask（高分辨率优先）

```bash
pip install vhr-cloudmask
```

```python
import rasterio
from vhr_cloudmask import CloudMaskPredictor

with rasterio.open("aerial_image.tif") as src:
    img = src.read()  # (bands, H, W)
    profile = src.profile

predictor = CloudMaskPredictor(model_type="unet++")
cloud_mask = predictor.predict(img)  # 常见: 0=背景, 1=云, 2=云影

profile.update(dtype="uint8", count=1, compress="lzw")
with rasterio.open("cloud_mask.tif", "w", **profile) as dst:
    dst.write(cloud_mask.astype("uint8"), 1)
```

### 4.4 segmentation_models_pytorch（自定义微调）

```bash
pip install segmentation-models-pytorch
```

```python
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,   # RGB=3, RGB+NIR=4
    classes=4,       # 背景/厚云/薄云/云影
    activation="softmax",
)
```

## 5. 推荐数据集（整合）

- **HRC_WHU**：150 张高分辨率 RGB 图像（约 0.5~15m），手工标注云掩膜  
- **AIR-CD / GF 系列高分辨率数据集**：航空/无人机场景风格  
- **自建数据集**：无人机样本 + Labelme/CVAT 标注

## 6. 参数设置（推理阶段）

### 6.1 vhr-cloudmask 关键参数（原始信息整合）

```python
from vhr_cloudmask import CloudMaskPredictor

predictor = CloudMaskPredictor(
    model_type="unet++",     # 或 deeplab / unet
    backbone="resnet34",     # 可换 resnet50 / efficientnet-b4
    pretrained=True,
    device="cuda:0",         # 或 cpu
    tile_size=512,           # 常用 512/768/1024
    overlap=0.25,            # 常用 0.2~0.35
    batch_size=4,            # 按显存调整
    threshold=0.5,           # 常用 0.4~0.6
    num_classes=4
)

cloud_mask = predictor.predict(
    image_array,
    normalize=True,
    augment=True             # TTA，提精度但更慢
)
```

### 6.2 smp/通用模型参数建议

- `encoder_name`：`resnet34`（速度-精度平衡）  
- `in_channels`：与波段一致（RGB=3，RGB+NIR=4）  
- `classes`：建议 3~5 类（厚云/薄云/云影分开）  
- `activation`：多类 `softmax`，二值 `sigmoid`  
- 推理加速：`half=True` 或 AMP

### 6.3 通用参数表（保留原信息）

| 参数类别 | 推荐设置（航空影像） | 说明 |
|---|---|---|
| 输入尺寸 | `tile_size=512~1024` | 太小丢上下文，太大易爆显存 |
| 重叠率 | `overlap=0.2~0.3` | 减少切块边缘伪影 |
| 归一化 | 与训练一致（ImageNet 或数据集统计） | 不一致会明显掉点 |
| 概率阈值 | `0.45~0.6` | 薄云多时适当降低 |
| 后处理 | 开闭运算 | 去噪、补洞 |
| TTA | 可开启 | 提精度，降速度 |
| Batch Size | 2~8（大图常 1~2） | 受显存约束 |
| 设备 | CUDA 优先 | GPU 提速明显 |

## 7. 航空影像实用技巧

- 大图必做分块 + 重叠融合（overlap 0.2~0.3）  
- 薄云/雾霾优先引入 NIR/红外通道  
- 云与亮地表/雪混淆时结合 NDVI 或热红外  
- 评估重点看薄云边界与云影（IoU / F1 / OA）  
- 掩膜输出后可转 GeoJSON/Shapefile 进入 GIS 流程

## 8. 推荐起步流程

1. 先用 `vhr-cloudmask` / `OmniCloudMask` 做零样本测试  
2. 若效果不足，基于 `HRC_WHU` 微调 U-Net++ / DeepLabV3+ / Transformer  
3. 生产化时加入分块并行、后处理和质量评估闭环

## 9. 补充工具（继续整合）

以下内容是对“开箱即用/PyPI 优先”方向的补充，整合进统一结构，避免与前文重复。

### 9.1 语义分割/通用大模型（RGB 场景）

| 工具 | 安装 | 核心特点 | 适用场景 |
|---|---|---|---|
| **samgeo (Segment-Geospatial)** | `pip install samgeo` | 基于 SAM，支持提示式分割（如 `cloud` 文本或点提示） | RGB 航空影像快速零样本试验 |
| **Grounded-SAM（Grounding DINO + SAM）** | 参考仓库安装 | DINO 先找目标框，SAM 出像素级掩膜 | 云与亮地物混淆较多的复杂背景 |

示例（samgeo）：

```python
from samgeo import SamGeo

sam = SamGeo(model_type="vit_h")
sam.predict(image, text_prompt="cloud", output="cloud_mask.tif")
```

### 9.2 多光谱/卫星风格工具（RGB+NIR 优先）

| 工具 | 安装 | 关键说明 |
|---|---|---|
| **ukis-csmask** | `pip install ukis-csmask` | DLR 出品，云/云影检测，适配卫星风格流程，也可迁移到 RGB+NIR 航空影像 |
| **s2cloudless** | `pip install s2cloudless` | 轻量级像素级云概率输出（0~1），适合大批量自动处理 |

### 9.3 预训练权重与训练框架补充

- **Cloud-Net（PyTorch/Keras）**：云检测经典网络，可从 GitHub 获取 `.pth/.h5` 权重。
- **torchgeo**（`pip install torchgeo`）：遥感数据与训练流程工具集，适合构建标准化训练/验证管线。
- **U-Net/Swin-UNet 等预训练路线**：适合已有标注数据后继续优化。

### 9.4 Fmask 的另一实现来源（补充）

除前文 `Fmask4` 路线外，历史上也有 `FCube` 相关实现版本；工程使用时建议固定到一个可复现仓库与数据包版本，避免环境/参数不一致。

## 10. 统一选型建议（最终整合）

| 需求场景 | 推荐方案 | 说明 |
|---|---|---|
| 纯 RGB 且希望快速见效 | `samgeo` / `vhr-cloudmask` | `samgeo` 零样本友好，`vhr-cloudmask` 更偏工程化稳定 |
| RGB+NIR | `DTACSNet` / `ukis-csmask` / `OmniCloudMask` | 光谱利用更充分，云与亮地物区分更稳 |
| 超大规模批处理 | `s2cloudless` + 分块并行 | 轻量、易自动化 |
| 追求最高精度 | `HRC_WHU` + U-Net++/DeepLab/Transformer 微调 | 需标注数据与训练资源 |
| 偏物理规则与稳健性 | `Fmask v5.0` | 适配复杂、部署成本相对更高 |

## 11. 最简落地建议

1. 先跑通：`vhr-cloudmask`（航空高分辨率）或 `samgeo`（RGB 零样本）。  
2. 有 NIR 时优先并行评估：`DTACSNet` / `ukis-csmask`。  
3. 建立统一评估：IoU、F1、云影准确率、薄云边界表现。  
4. 生产前补齐：分块推理、后处理、GIS 输出、批量监控与失败重试。  

参考划分（适当调整即可）：

                    ┌─── [1. 无云] ───────────────> 直接输出（或进行轻微色彩均衡）
                    │
                    ├─── [2. 薄雾] ───────────────> 运行常规去雾（注重保持纹理与对比度）
[航空RGB图] ──> [4分类模型]
                    ├─── [3. 薄云/浓雾] ──────────> 运行强去雾 + BM3D/DnCNN去噪（注重噪声抑制）
                    │
                    └─── [4. 厚云] ───────────────> 生成无效掩膜（Mask） ──> 启动时空修补/GAN生成
注意：避免将类2误判为类3，可能会过度平滑，丢失细节

                    ┌─── [1. 无云/薄雾] ───────────────> [100%可用数据] ──> 直接输出
                    │
                    ├─── [2. 贴地浓雾] ────────────────> [微弱信号提取] ──> 运行“红外细节增强 (CLAHE/引导滤波)” ──┐
[纯红外航空图] ──> [内部4分类模型]                                                                             ├─> 合并输出清晰图
                    ├─── [3. 高空薄云] ────────────────> [亮温能量补偿] ──> 运行“红外去云算法 (热对比度补偿/AI)” ──┘
                    │
                    └─── [4. 高空厚云] ────────────────> [0信息量数据] ───> 标记为无效掩膜 (Mask) 丢弃或时空修复                    


最终推荐方案（简化版）
RGB 推荐保持原 4 分类（已匹配最新需求）：
text[航空RGB图] ──> [4分类模型]
    ├── 1. 无云          → 直接输出（轻微色彩均衡）
    ├── 2. 薄雾          → 专门去雾算法（效果不严格，推荐轻量稳定方案）
    ├── 3. 薄云/浓雾     → 适当去噪 或 轻增强（BM3D/DnCNN 或 CLAHE）
    └── 4. 厚云          → 无效掩膜 → 时空修补/GAN 生成
IR 推荐保持原 4 分类（优势明显）：
text[纯红外航空图] ──> [内部4分类模型]
    ├── 1. 无云/薄雾     → 100%可用，直接输出
    ├── 2. 贴地浓雾      → 弱信号提取 + 红外细节增强（CLAHE/引导滤波）
    ├── 3. 高空薄云      → 亮温能量补偿 + 红外去云算法
    └── 4. 高空厚云      → 无效掩膜 → 时空修复