# Object Detection 使用说明

## 安装与参考链接

### 相关 PyPI 包

- https://pypi.org/project/remote-sensing-processor/
- `pip install Remote-Sensing-Analysis`
- `pip install remote-sensing-processor`

### 推荐方案（preferred）

- `pip install ultralytics`
- https://pypi.org/project/ultralytics/
- https://github.com/ultralytics/ultralytics

### 直接权重下载参考

- https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
- https://github.com/ultralytics/assets/releases

## 批量下载脚本

```bash
python download-yolo-models.py
```

默认会下载 YOLO11/YOLO26 的 `n,s,m,l,x` 两类任务权重：`detect,obb`，并统一放到 `weights/` 目录。

常用示例：

```bash
python download-yolo-models.py --families 11,26 --variants n,s --tasks detect,obb
python download-yolo-models.py --families 11 --variants n,s,m,l,x --tasks detect,seg,pose,cls,obb
python download-yolo-models.py --output-dir pretrained_weights
```

## 类别说明（OBB 常见遥感权重）

`classes` 参数的类别索引由具体权重文件决定，建议始终以 `model.names` 为准：

```python
from ultralytics import YOLO
model = YOLO("ckpts/yolo11n-obb.pt")
print(model.names)
```

常见 DOTA 类别体系：

- DOTA v1.0（15 类）
  - plane, ship, storage-tank, baseball-diamond, tennis-court, basketball-court, ground-track-field, harbor, bridge, large-vehicle, small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool
- DOTA v1.5（16 类）
  - 在 v1.0 基础上增加：container-crane
- DOTA v2.0（18 类）
  - 在 v1.5 基础上增加：airport, helipad

与军事目标相关的常见类别（广义）：

- 直接目标类：plane, helicopter, ship, large-vehicle
- 场景相关类：harbor, airport, helipad, bridge, storage-tank

说明：以上标签通常不直接区分军用/民用属性，如需“军事目标”识别，通常需要二次细分类或专门数据集。

## 推理参数参考（YOLO OBB）

- https://docs.ultralytics.com/modes/predict/#inference-arguments
- https://docs.ultralytics.com/zh/modes/predict/

以下是针对 YOLO OBB（Oriented Bounding Box）模型在遥感/航空光学图像（车辆、飞机、船舶等小目标、任意方向、大图场景）推理时的常用参数及推荐设置。

### 1. 常用推理参数一览（`model.predict()` 或 `yolo obb predict`）

| 参数 | 类型 | 默认值 | 含义说明 | 遥感/航空推荐设置 | 理由 |
|---|---|---|---|---|---|
| **source** | str/list | - | 输入图像/文件夹/视频/TIF 路径 | 你的卫星图像路径或文件夹 | - |
| **imgsz** | int / tuple | 640 | 输入图像尺寸（模型会 resize 到此大小） | **1024**（或 1280） | 遥感图像分辨率高，小目标（如车辆）需要更大输入尺寸以保留细节，DOTA 常用 1024 |
| **conf** | float | 0.25 | 置信度阈值（低于此值过滤掉） | **0.15 ~ 0.3**（先试 0.25） | 遥感小目标多，阈值适当降低可提高召回率，后续再结合 IoU 过滤假阳性 |
| **iou** | float | 0.7 | NMS 的 IoU 阈值（重叠框抑制） | **0.45 ~ 0.6** | 遥感中船舶/车辆常密集或旋转重叠，适当降低可保留更多有效框 |
| **max_det** | int | 300 | 每张图最多检测目标数 | **500 ~ 1000**（或更高） | 港口/停车场目标数量大，默认 300 容易漏检 |
| **device** | str | None | 运行设备（cpu / cuda:0 / 0 等） | **'0'** 或 **'cuda:0'**（有 GPU 时） | GPU 显著加速大图推理 |
| **half** | bool | False | 是否使用 FP16（半精度）加速 | **True**（GPU 上） | 速度提升明显，精度损失通常很小 |
| **batch** | int | 1 | 批量大小（处理文件夹/视频时有效） | **4 ~ 16**（根据显存） | 批量处理多张图像时提升吞吐量 |
| **save** | bool | False | 是否保存带标注的结果图 | **True** | 方便可视化检查 |
| **save_txt** | bool | False | 是否保存 txt 标签结果（含旋转框） | **True**（后处理需要时） | 输出旋转框坐标，便于 GIS 集成 |
| **stream** | bool | False | 流式返回结果（节省内存，适合大批量） | 大图或视频时 **True** | 避免一次性加载所有结果导致内存爆炸 |
| **classes** | list[int] | None | 只检测指定类别（按模型类别 ID） | 如 `[0, 1, 5]`（根据类别映射） | 只关注车辆/飞机/船舶时可加速并减少干扰 |
| **agnostic_nms** | bool | False | 是否类别无关 NMS | **True**（密集场景） | 遥感中同类目标密集时有用 |

#### imgsz 最大支持说明

- Ultralytics 对 `imgsz` 通常没有固定“硬上限”，上限主要由显存/内存决定。
- 实际可用最大值受 `model` 大小、`batch`、`half`、输入分辨率和设备影响。
- 工程上建议将 `imgsz` 设为 32 的倍数（常见如 640/768/1024/1280/1536/2048）。
- 遥感 OBB 常用范围：`1024 ~ 1280`；高显存设备可尝试 `1536 ~ 2048`。
- 若出现 OOM（显存不足），优先降低 `imgsz`，或同时开启 `half=True`、减小 `batch`。

其他实用参数：

- `verbose=True`：打印更多推理信息
- `augment=False`：推理时一般关闭 TTA（测试时增强），速度更快
- `line_width`、`font_size`：可视化时调整框和文字粗细

### 2. 针对遥感/航空任务的推荐设置示例

#### Python 代码推荐写法（最灵活）

```python
from ultralytics import YOLO

model = YOLO("yolo26n-obb.pt")   # 或你 fine-tune 后的 best.pt

results = model.predict(
    source="your_satellite_image.tif",   # 支持 .tif / .png / 文件夹
    imgsz=1024,                          # 关键：遥感推荐 1024 或 1280
    conf=0.25,                           # 小目标可适当降低到 0.2
    iou=0.5,                             # 密集旋转目标建议 0.45~0.55
    max_det=800,                         # 防止漏检大量小目标
    device=0,                            # GPU
    half=True,                           # FP16 加速
    save=True,                           # 保存可视化结果
    save_txt=True,                       # 保存标签（含旋转角度）
    stream=False,                        # 大图时可设 True
    classes=None,                        # 或指定类别 ID
    verbose=True
)
```

### 3. 遥感图像额外实用技巧

- 大图处理（强烈推荐）：遥感图像常几千 × 几千像素，直接用 `imgsz=1024` 会因 resize 损失细节。建议先做切片（slicing）+ 重叠融合
  - Ultralytics 内置支持 `imgsz` + 自动 letterbox，但超大图更建议配合 `ultralytics` predictor 或第三方切片工具（SAHI 常用）
- 小目标多：`imgsz` 调大 + `conf` 适当降低 + `max_det` 调高
- 旋转目标密集（船舶/车辆停放）：`iou` 调低到 0.45~0.55，避免过度抑制
- 显存不足：降低 `imgsz` 到 832，或使用 `half=True` + `batch=1` + `stream=True`
- 后处理：输出后可用 `result.obb` 提取旋转框，转成 GeoJSON 或 Shapefile 用于 GIS 分析

测试建议：

1. 先用 `imgsz=1024, conf=0.25, iou=0.5` 跑一张典型图像观察效果
2. 根据漏检/误检情况调整 `conf` 和 `iou`
3. 如果小目标（如车辆）召回率低，再尝试 `imgsz=1280` 或在 fine-tune 时增大输入尺寸
