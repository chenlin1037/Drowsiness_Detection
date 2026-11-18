---
license: apache-2.0
tasks:
  - image-classification
tags:
  - drowsiness-detection
  - eye-state
  - driver-monitoring
  - resnet
  - spp
  - fatigue-detection
language:
  - zh
---

# 眼睛状态分类器（疲劳检测）

基于 MRL Eye Dataset 的眼睛开/闭状态二分类模型，面向驾驶员疲劳检测
（Driver Monitoring）场景。模型在**受试者隔离划分**（同一被试不跨集合）
的 test 集上取得 **F1 0.9885**，眼镜佩戴子集 F1 0.9923。

- 代码仓库：[github.com/chenlin1037/Drowsiness_Detection](https://github.com/chenlin1037/Drowsiness_Detection)
- 训练过程：17 组消融实验，完整报告见仓库内《模型消融实验报告.md》

## 模型架构

| 组件 | 说明 |
| --- | --- |
| Backbone | ResNet34（ImageNet 预训练），首层卷积改为灰度单通道输入，用 RGB 权重通道均值初始化 |
| 特征聚合 | **SPP 空间金字塔池化**（默认启用）：对 layer4 特征图按 1/2/4 bin 自适应平均池化拼接，512×(1+4+16)=10752 维，与输入分辨率无关 |
| 元数据融合 | 分类头末端拼接 9 维条件属性（性别/眼镜/反光/光照/传感器） |
| 分类头 | MLP 256 → 128 → 1，输出单个二分类 logit |
| 判决阈值 | `logit >= 0` 为睁眼（阈值扫描显示该点已近似最优，无需调参） |

消融实验中评估的 SE 注意力、CBAM 空间门控、FiLM 元数据深度注入、
多尺度融合均以可选开关保留在代码中，默认关闭。

## 训练数据

- 数据集：[MRL Eye Dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)，
  37 名受试者，约 9 万张近红外眼部裁剪图（文件名含性别/眼镜/反光/光照等元数据）
- 划分：按**受试者**以 70%/15%/15% 隔离划分（种子 20261105），
  杜绝同一被试的图像同时出现在训练与测试集造成的泄漏
- 清单文件（`train/val/test.csv` + `metadata.csv`）随代码仓库发布，划分可精确复现

## 训练配置（最终模型 Run C）

- 输入：灰度图 112×112，`Normalize(mean=0.449, std=0.226)`
- YOLO 风格数据增强（对比度/亮度抖动、旋转 ±10°、平移、缩放、水平翻转）
- 困难条件（戴眼镜/大反光/差光照）加权过采样，强度 0.5
- 可疑标签软标签训练：75 个标签噪声可疑样本 1→0.8 / 0→0.2
- AdamW，lr 1e-4，weight decay 1e-4，BCEWithLogitsLoss

## 评估结果（test 集）

| 指标 | 数值 |
| --- | --- |
| Accuracy | 0.9897 |
| Precision | 0.9876 |
| Recall | 0.9893 |
| F1 | **0.9885** |
| FP / FN | 71 / 61 |
| 眼镜佩戴子集 F1 | 0.9923 |
| 最差被试（s0026）F1 | 0.9839 |

备选运行点 Run D：Recall 0.9953（FN 仅 27）但 FP 126，
适合漏报代价显著高于误报的场景。

## 快速使用

模型权重以 PyTorch checkpoint 发布（内含全部架构开关，加载时自动重建）。
需要配合 [GitHub 仓库](https://github.com/chenlin1037/Drowsiness_Detection)
中的 `src/model.py` 使用：

```python
import torch
from PIL import Image
from torchvision import transforms
from src.model import EyeStateClassifier, adapt_state_dict

# 1. 加载模型(架构开关从 checkpoint 自动恢复)
ckpt = torch.load("best.pt", map_location="cpu", weights_only=True)
model = EyeStateClassifier(
    pretrained=False,
    use_metadata=ckpt.get("use_metadata", True),
    use_se=ckpt.get("use_se", False),
    multi_scale=ckpt.get("multi_scale", False),
    spatial_attn=ckpt.get("spatial_attn", False),
    film=ckpt.get("film", False),
    use_spp=ckpt.get("spp", True),
)
model.load_state_dict(adapt_state_dict(ckpt["model_state_dict"]))
model.eval()

# 2. 预处理:灰度 + resize(与训练分辨率一致) + 归一化
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.449,), std=(0.226,)),
])
x = transform(Image.open("eye.png").convert("L")).unsqueeze(0)  # [1, 1, 112, 112]

# 3. 元数据: [性别, 眼镜, 反光==0, 反光==1, 反光==2, 光照, 传感器01, 传感器02, 传感器03]
#    取值均为 0/1;未知时可填 0,但建议尽量提供真实属性
metadata = torch.zeros(1, 9)

# 4. 推理
with torch.inference_mode():
    logit = model(x, metadata).item()
eye_state = "open" if logit >= 0 else "closed"
print(logit, eye_state)
```

输入要求：眼部裁剪图（MRL 风格的近红外/灰度眼睛区域特写），
启用 SPP 后任意输入尺寸均可，但建议与训练分辨率一致（112×112）。

## 局限性

- 仅针对 MRL 风格的**近红外眼部裁剪图**训练，未在可见光、车内全景
  或真实驾驶场景中验证，直接用于其他成像条件前需先微调评估；
- FN 略偏向睁眼误判（61 例），漏检真困倦的场景建议选用 Run D
  权重或部署端阈值微调（如 t=-0.25，F1 0.9888，FN 61→51）；
- metadata 为条件属性而非时序信息，模型不感知连续闭眼时长，
  疲劳判决（如 PERCLOS）需在上层业务逻辑中累积本模型输出。

## 引用

若使用本模型或数据划分方案，请引用原始数据集：

```text
MRL Eye Dataset — Márton, László et al., 
"A driver's eye state detection method for driver monitoring systems",
originally released at http://mrl.cs.vsb.cz/eyedataset
```

## 许可

- 代码：Apache-2.0
- 数据：遵循 MRL Eye Dataset 原始协议（学术研究用途）
