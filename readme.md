# 疲劳检测：眼部状态分类（MRL Eye Dataset）

基于 MRL Eye Dataset 的驾驶员眼睛开/闭状态二分类模型。最终模型
（消融实验 Run C）在受试者隔离的 test 集上取得 **F1 0.9885**，
眼镜佩戴子集 F1 0.9923。完整实验过程见
[模型消融实验报告.md](模型消融实验报告.md)。

## 数据集

原始数据集（[Kaggle: mrl-eye-dataset](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset)）
是分类数据集，没有标注框，图像文件名包含分类信息，两个文件夹分别为
睁眼（Open-Eyes）和闭眼（Close-Eyes）。

原始数据集的文件名形如：

```text
s0001_00123_0_1_1_0_1_01.png
```

按下划线分隔，字段依次为：

| 位置 | 含义 | 取值 |
| --- | --- | --- |
| 1 | 受试者 ID | s0001 到 s0037 |
| 2 | 图像 ID | 序号 |
| 3 | 性别 | 0 男 / 1 女 |
| 4 | 是否戴眼镜 | 0 否 / 1 是 |
| 5 | 眼睛状态 | 0 闭眼 / 1 睁眼 |
| 6 | 反光 | 0 无 / 1 小 / 2 大 |
| 7 | 光照 | 0 差 / 1 好 |
| 8 | 传感器 ID | 01、02、03 |

下载原始数据集：

```bash
python3 download_dataset.py   # kagglehub 拉取,输出数据集路径
```

## MRL 受试者隔离划分

`mrl_dataset.py` 审计文件名元数据，并以受试者为不可拆分分组搜索候选划分：

```bash
python3 mrl_dataset.py prepare
```

默认扫描 `dataset/mrleyedataset`，保留原始图片不变，并在
`dataset/mrl_split/` 实际复制出三个数据集目录：

```text
dataset/mrl_split/
├── train/
│   ├── Close-Eyes/
│   └── Open-Eyes/
├── val/
│   ├── Close-Eyes/
│   └── Open-Eyes/
└── test/
    ├── Close-Eyes/
    └── Open-Eyes/
```

可直接将这三个目录用于分类训练。再次执行命令会按同一划分更新图片；
若目录中发现不属于当前清单的旧图片，脚本会报错停止，避免旧数据残留
导致集合混杂。审计和清单文件也保存在 `dataset/mrl_split/`（已纳入版本
控制，保证划分可复现）：

- `metadata.csv`：全量图像的文件名元数据审计；
- `train.csv` / `val.csv` / `test.csv`：各集合逐图像清单，
  `split_image_path` 指向已复制到对应集合的图片。

按受试者数近似采用 70%/15%/15%。眼睛状态是主要分布目标，眼镜、光照、
反光和传感器是辅助目标；有至少两名受试者覆盖的每种条件，会尽量保证
训练集和测试集都包含该条件。该要求不会拆分受试者；若条件覆盖受试者太少，
则不可能让所有集合都包含该条件，需结合报告中的受试者支持数解读。
随机种子默认为 `20261105`，候选数默认为 `10000`，可通过参数更改。

## 模型架构

`src/model.py` 中的 `EyeStateClassifier`：

- **Backbone**：ResNet34（ImageNet 预训练），首层卷积改为灰度单通道输入，
  并用原 RGB 权重的通道均值初始化；
- **分类头**：3 层 MLP（256 → 128 → 1），末端拼接 9 维 metadata 特征
  （glasses / reflection / lighting 等条件属性，即文件名第 4/6/7 字段等）；
- **输出**：单个二分类 logit，配合 `BCEWithLogitsLoss`，判决阈值 `logits >= 0`。

为支持任意输入分辨率，模型提供 **SPP 空间金字塔池化**（`--spp`）：
对 layer4 特征图按 1/2/4 三组 bin 做自适应平均池化并拼接
（512×(1+4+16)=10752 维），特征维度与输入尺寸无关，`--img-size`
可取任意 ≥32 的值（默认路径为单一 GAP，仅 512 维）。

以下模块为消融实验能力，默认关闭，通过开关启用：

| 开关 | 模块 | 说明 |
| --- | --- | --- |
| `--spp` | SpatialPyramidPool | 空间金字塔池化，支持任意输入尺寸 |
| `--se` | SEBlock | SE 通道注意力，插在 layer3/layer4 之后 |
| `--multi-scale` | — | layer2/3/4 GAP 拼接（896 维），与 `--spp` 互斥 |
| `--spatial-attn` | SpatialGate | CBAM 风格空间门控（近恒等初始化） |
| `--film` | FiLMGenerator | metadata 深度注入调制特征（已被消融证伪） |
| `--no-metadata` | — | 分类头不拼接 metadata（消融对照） |

## 训练

训练入口为 `src/train.py`，数据读取与增强在 `src/dataset.py`，
验证循环在 `src/evaluate.py`。最终模型（Run C）的完整训练命令：

```bash
python3 -m src.train \
  --name run_c \
  --img-size 112 \
  --augment \
  --weighted-sampling \
  --soft-label-csv runs/error_analysis/train_label_noise.csv
```

关键训练选项：

- `--img-size`：输入分辨率（默认 74；Run C 用 112，配合 SPP 可取任意 ≥32 值）；
- `--augment`：YOLO 风格数据增强（对比度/亮度抖动、旋转、平移、缩放、翻转），
  单项参数可用 `--degrees`/`--translate` 等覆盖预设值；
- `--weighted-sampling`：按困难条件（glasses=1/reflection=2/lighting=0）
  加权过采样，`--oversample-weight` 控制强度（Run C 用 0.5）；
- `--soft-label-csv`：可疑标签 CSV（由 `detect_label_noise.py` 生成），
  命中样本用软标签训练（默认 1→0.8 / 0→0.2）；
- `--label-smoothing`、`--cosine`、`--patience`：标签平滑、cosine 退火、早停；
- `--no-pretrained`：不加载 ImageNet 预训练权重。

训练过程按 epoch 写入 `runs/{name}-result.csv`，最优验证损失模型保存到
`runs/checkpoints[-N]/best.pt`（同名自动版本化），checkpoint 内记录
全部架构开关，推理端据此重建模型。测试集不参与训练和模型选择。

## 预测与评估

对测试集生成预测 CSV：

```bash
python3 -m src.predict \
  --checkpoint runs/checkpoints-12/best.pt \
  --output dataset/mrl_split/test_predictions.csv
```

预测 CSV 包含全部测试图片，列为 `image_path,predicted_eye_state`，
闭眼为 `0`、睁眼为 `1`。随后生成整体和分条件测试指标：

```bash
python3 mrl_dataset.py evaluate --predictions dataset/mrl_split/test_predictions.csv
```

评估器会拒绝缺失、重复或非测试集预测，并将整体准确率、宏平均及类别
精确率/召回率/F1、混淆矩阵，以及按眼镜、光照、反光和传感器分组的指标
写入 `dataset/mrl_split/test_metrics.json`。每组同时报告图像数和受试者数。

## 实验脚本

根目录下的独立实验与分析脚本（完整结论见消融实验报告）：

| 脚本 | 用途 |
| --- | --- |
| `analyze_errors.py` | 错误样本归因分析（按条件分组统计 FP/FN） |
| `detect_label_noise.py` | 可疑标签检测，输出软标签训练所需的 CSV |
| `eval_test.py` | 候选 checkpoint 在 test 集上的最终裁决评估 |
| `threshold_scan.py` | 判决阈值扫描（P/R 运行点分析） |

## 最终模型结果（test 集）

| 配置 | Accuracy | Precision | Recall | F1 | FN / FP |
| --- | --- | --- | --- | --- | --- |
| Run C（最终） | 0.9897 | 0.9876 | 0.9893 | 0.9885 | 61 / 71 |
| Run D（备选，高召回） | 0.9881 | 0.9784 | 0.9953 | 0.9868 | 27 / 126 |

- Run C：软标签 + 强再平衡使 FP 减半，P/R 均衡，眼镜子集 0.9923；
- Run D：召回更高（FN 仅 27），适合漏报代价更高的场景；
- 阈值扫描结论：Run C 默认 `logits >= 0` 已近似最优，可直接部署。
