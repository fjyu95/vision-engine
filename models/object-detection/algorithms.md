# 遥感/航空目标检测预训练模型整合（车辆、飞机、船舶）

## 1. 选型总览

| 需求场景 | 优先模型 | 备选模型 | 关键理由 |
|---|---|---|---|
| 实时检测 / 边缘部署 | YOLOv11-OBB、YOLOv8-OBB、YOLOv10 | YOLOv12、YOLOv26-OBB | 推理速度快，部署成本低，适合大图切片后批量或实时推理 |
| 高精度离线检测 / 科研 | MMRotate: Oriented R-CNN、RTMDet-Rotated、ReDet | RoI Transformer、S2ANet、OriMamba/TCA-Net/SA3Det | 旋转框检测能力强，对密集小目标和复杂背景表现更稳定 |
| Transformer 路线 / 跨域泛化 | Deformable DETR、RT-DETR、RF-DETR | DN-DETR、DAB-DETR、Conditional DETR | 对复杂场景和跨域迁移更友好，RF-DETR（DINOv2骨干）在小样本场景有优势 |
| 船舶专项 | ArcGIS Ship Detection (RGB)、HRSC2016系模型（如 ReBiDet） | Deformable DETR（SSDD） | 针对船舶形态与方向分布优化明显，旋转框收益高 |
| 多目标通用（飞机/船舶/车辆） | DOTA 预训练的 YOLOv9/YOLOv11、MMRotate 模型 | PSODNet | 覆盖类别全，便于统一训练与推理接口 |

## 2. 模型信息整合清单

### 2.1 YOLO 系列（速度优先）

- [YOLOv11](https://www.nature.com/articles/s41598-025-96314-x)：2025 主流版本之一，兼顾速度与精度。
- YOLOv8-OBB：工业应用广，DOTA 旋转目标上表现稳定。
- [YOLOv10](https://blog.roboflow.com/best-object-detection-models/)：强调极致速度（无 NMS 设计）。
- [YOLOv12](https://cv-tricks.com/how-to/real-time-object-detection-in-2025/)：在 v11 基础上继续提升精度。
- YOLOv26-OBB：2025-2026 新一代轻量 OBB 方案之一，偏实时部署。
- YOLOv9-DOTA / Remote-Sensing-Analysis：提供 DOTA 1.0（15 类）预训练权重，开箱可用。
- xView-YOLOv3：车辆专项历史方案，适配特定卫星场景（如 Barcelona 区域）。

#### 2.1.1 YOLO11 / YOLO12 / YOLO26 的命名与差异（详细）

Ultralytics 在 2024 年后将命名策略从传统版本号逐步转向“更强调发布时间语义”的方式，因此会出现 11、12、26 这种看起来不连续的命名。

| 型号 | 时间定位 | 关键变化 | 适用建议 |
|---|---|---|---|
| YOLO11 | 2024 年发布主力代 | 作为 v8 后的主线演进，重构 C3k2、C2PSA 等模块，兼顾轻量与精度，原生支持 OBB | 生产首选，稳定落地优先 |
| YOLO12 | 2025 年迭代代 | 更强调注意力机制，对复杂背景建模能力更强 | 背景复杂、误检敏感场景可优先尝试 |
| YOLO26 | 2026 年新代 | 端到端能力增强，弱化/取消传统 NMS 依赖，并针对大模型蒸馏做优化 | 新项目追求上限性能时可优先评估 |

命名理解要点：

- 从 YOLOv1 到 YOLOv10 属于经典连续版本号时代。
- 后续命名更强调“代际与发布时间识别”，便于快速判断模型新旧与生态阶段。
- 实际落地建议仍以工程稳定性、推理生态（TensorRT、Docker、部署链路）和任务指标为准，不只看数字大小。

工程选型建议（遥感 OBB）：

- 稳定优先：YOLO11-OBB。
- 性能上限优先：YOLO26-OBB。
- 过渡与兼容：YOLO8-OBB/YOLO11-OBB 并行验证后再迁移。

### 2.2 MMRotate / 旋转检测系列（精度优先）

- Oriented R-CNN：DOTA 上的强基线，速度和精度均衡。
- RTMDet-Rotated：高性能实时与精度折中方案。
- ReDet：对旋转不变特征建模效果好。
- [RoI Transformer](https://arxiv.org/pdf/2503.06146)：适合高长宽比目标（如大型船舶、客机）。
- S2ANet：特征对齐能力强，复杂背景下有优势。
- OriMamba / TCA-Net / SA3Det：2025 新兴模型，在 DOTA/HRSC2016 上有竞争力。

### 2.3 Transformer / DETR 系列

- Deformable DETR：飞机与船舶任务中均有很强表现，尤其适合密集目标。
- RT-DETR / RF-DETR：2025 热门路线，RF-DETR 的骨干迁移能力更强。
- PSODNet：场景感知检测网络，多源联合预训练，面向复杂背景干扰。

### 2.4 任务专项模型

- SAMRS（车辆）：基于 SAM 的遥感检测/分割适配方案，联合多数据集预训练。
- ArcGIS Ship Detection (RGB)：Mask R-CNN 架构，官方发布，适合高分辨率光学影像船舶检测。

## 3. 典型任务对应推荐

| 任务 | 推荐优先级 | 推荐模型 |
|---|---|---|
| 飞机检测 | 高 | Deformable DETR、YOLOv11-OBB、Oriented R-CNN |
| 船舶检测（光学） | 高 | ArcGIS Ship Detection、YOLOv11-OBB、MMRotate 系列 |
| 船舶检测（SAR） | 高 | Deformable DETR（SSDD）、YOLOv11（SSDD） |
| 车辆检测 | 高 | SAMRS、YOLOv9-DOTA、YOLOv11-OBB |
| 多类统一检测 | 高 | DOTA 预训练的 YOLO / MMRotate、PSODNet |

## 4. 预训练数据集与权重来源

优先使用领域数据集预训练权重，通常显著优于仅 ImageNet 预训练：

- DOTA（v1.0/1.5/2.0）：遥感 OBB 核心基准，覆盖飞机、船舶、大/小车辆等多类目标。
- [DIOR](http://121.5.109.38/?page_id=383)：20 类遥感目标，适合通用预训练。
- NWPU VHR-10：经典高分辨率目标检测数据集。
- FAIR1M-2.0、iSAID：高质量遥感场景补充数据。
- xView：车辆与城市场景目标常用。
- Pleiades Aircraft：飞机专项数据。
- HRSC2016：船舶专项（光学）代表数据集。
- SSDD：船舶专项（SAR）常用数据集。
- UCAS-AOD：飞机+车辆任务常用基准。

### 4.1 DOTA 各版本区别（落地视角）

| 版本 | 类别规模 | 标注特点 | 训练难度 | 适用阶段 |
|---|---|---|---|---|
| DOTA v1.0 | 15 类 | 标准 OBB 基准，生态最成熟 | 低到中 | 首次落地、先跑通流程 |
| DOTA v1.5 | 16 类 | 在 v1.0 基础上补充更小目标与额外类别 | 中 | 需要提升小目标召回时 |
| DOTA v2.0 | 18 类 | 类别更全、场景更复杂、长尾更明显 | 中到高 | 完整业务覆盖与精度冲刺 |

推荐顺序：先用 v1.0 跑通端到端，再切到 v1.5/v2.0 做类别扩展和精度优化。

## 5. 可直接获取的资源（统一评价与排序）

| 排序 | 资源/框架 | 推荐指数 | 主要价值 | 获取方式 |
|---|---|---|---|---|
| 1 | Ultralytics（YOLO/YOLO-OBB） | ★★★★★ | 最快落地，训练与推理一体化，工程部署最省心 | `pip install ultralytics` |
| 2 | MMRotate（OpenMMLab） | ★★★★★ | 旋转检测精度上限高，DOTA 生态最成熟 | `openmim` + Model Zoo（`mim install mmrotate`） |
| 3 | SAHI | ★★★★☆ | 遥感大图切片推理与结果融合，显著降低漏检 | `pip install sahi` |
| 4 | MMDetection | ★★★★ | 与 MMRotate 生态兼容，便于多任务扩展 | `openmim`（`mim install mmdet`） |
| 5 | Hugging Face Models | ★★★★ | 社区模型获取快，便于快速验证不同权重 | 模型页下载/`huggingface_hub` |
| 6 | Remote-Sensing-Analysis | ★★★☆ | 可快速验证 YOLOv9-DOTA 封装路线 | `pip install remote-sensing-analysis` |
| 7 | Rasterio | ★★★☆ | GeoTIFF 读写与栅格处理基础能力 | `pip install rasterio` |

建议使用顺序：`Ultralytics + SAHI` 先跑通，再按精度需求切到 `MMRotate/MMDetection`，需要快速试权重时补充 Hugging Face。

## 6. 最快最稳落地方案（先跑起来）

目标：先完成可用推理链路，再逐步扩类别和提精度。

1. 模型直接选 YOLOv11n-OBB 官方权重，不先改网络结构。
2. 数据先选 DOTA v1.0 标签体系，先对齐 15 类流程。
3. 输入先固定为切片推理（如 1024 + overlap），保证大图可稳定跑通。
4. 输出先统一 OBB + 最小字段（类别、置信度、旋转框坐标），先打通接口协议。
5. 验证先看是否稳定出结果与延迟可控，再进入类别裁剪和精度优化。

### 6.1 三阶段执行

| 阶段 | 目标 | 只做这些事 |
|---|---|---|
| Phase 1: 跑通 | 有检测结果、接口可调用 | 官方权重 + 默认参数 + 单卡推理 |
| Phase 2: 可用 | 误检漏检可接受 | 调整阈值、切片参数、NMS/后处理 |
| Phase 3: 优化 | 面向业务指标 | 切到 DOTA v1.5/v2.0、做微调与模型替换 |

### 6.2 YOLO OBB 小模型与速度优先选择

YOLO OBB 的常见规模规律：`n > s > m > l > x`（从快到慢、从小到大）。

| 速度优先排序 | 模型 | 特点 | 适用建议 |
|---|---|---|---|
| 1 | yolo11n-obb.pt | 最小、最快 | 默认首选，先跑通链路 |
| 2 | yolo8n-obb.pt | 很小、很快 | 兼容老版本流程时优先 |
| 3 | yolo11s-obb.pt | 速度与精度更平衡 | 跑通后提升精度的第一步 |
| 4 | yolo8s-obb.pt | 稳定平衡 | 需要 v8 生态时可选 |

补充：推理速度除模型规模外，还强依赖 `imgsz` 与切片参数；同模型下减小输入尺寸和重叠率通常会明显提速。

### 6.3 最小可运行命令（MVP）

```bash
pip install ultralytics
yolo obb predict model=yolo11n-obb.pt source=your_image_or_dir imgsz=1024 conf=0.25 save=True
```

如需更稳的工程基线，可先用 YOLOv8-OBB 替代，流程不变，后续再平滑切换到 v11/v26。

## 7. 最终推荐结论

- 默认首选：YOLOv11-OBB（实时与工程效率最佳）。
- 精度首选：MMRotate（Oriented R-CNN / RTMDet-Rotated）。
- Transformer 方案：Deformable DETR / RF-DETR（复杂场景与跨域更稳）。
- 船舶专项：ArcGIS Ship Detection（光学）+ SSDD/HRSC2016 权重补充。
- 数据优先级：DOTA > DIOR/FAIR1M/iSAID > 任务专项数据（HRSC2016、SSDD、Pleiades、xView）。
