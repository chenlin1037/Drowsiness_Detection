原始数据集有两个文件夹，一个是睁眼，一个是闭眼
是一个分类数据集，没有标注框，图像文件名称包含了分类信息，

原始数据集的文件名形如：

s0001_00123_0_1_1_0_1_01.png

按下划线分隔，字段依次为：

位置	含义	取值
1	受试者 ID	s0001 到 s0037
2	图像 ID	序号
3	性别	0 男 / 1 女
4	是否戴眼镜	0 否 / 1 是
5	眼睛状态	0 闭眼 / 1 睁眼
6	反光	0 无 / 1 小 / 2 大
7	光照	0 差 / 1 好
8	传感器 ID	01、02、03

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
导致集合混杂。审计和清单文件也保存在 `dataset/mrl_split/`：

- `subject_audit.csv`：逐受试者统计各字段取值对应的图像数。
- `condition_coverage.csv`：各条件覆盖的受试者 ID、受试者数和图像数。
- `subject_split.csv` 和 `image_manifest.csv`：受试者划分及逐图像清单；
  `split_image_path` 指向已复制到对应集合的图片。
- `split_summary.json`：划分比例、种子、候选数、选中候选、文件名元数据指纹及各集合分布。

按受试者数近似采用 70%/15%/15%。眼睛状态是主要分布目标，眼镜、光照、
反光和传感器是辅助目标；有至少两名受试者覆盖的每种条件，会尽量保证
训练集和测试集都包含该条件。该要求不会拆分受试者；若条件覆盖受试者太少，
则不可能让所有集合都包含该条件，需结合报告中的受试者支持数解读。
随机种子默认为 `20261105`，候选数默认为 `10000`，可通过参数更改。

使用受试者隔离后的训练集和验证集训练分类器：

```bash
python3 -m src.train
```

数据读取和特征预处理位于 `src/dataset.py`，验证循环位于
`src/evaluate.py`，训练入口位于 `src/train.py`。默认训练 20 轮，使用
ResNet18 ImageNet 预训练权重，并将验证损失最低的模型保存到
`checkpoints/eye_state_classifier.pt`。可通过 `--data-dir`、`--output`、
`--epochs`、`--batch-size` 等参数调整训练；没有预训练权重或不希望下载时，
可添加 `--no-pretrained`。测试集不参与训练和模型选择。

使用 `src/predict.py` 对测试集生成预测 CSV：

```bash
python3 -m src.predict \
  --checkpoint checkpoints/eye_state_classifier.pt \
  --output dataset/mrl_split/test_predictions.csv
```

预测 CSV 会包含全部测试图片，列为
`image_path,predicted_eye_state`；路径需与 `image_manifest.csv` 一致，
闭眼为 `0`、睁眼为 `1`。随后可用以下命令生成整体和分条件测试指标：

```bash
python3 mrl_dataset.py evaluate --predictions path/to/test_predictions.csv
```

评估器会拒绝缺失、重复或非测试集预测，并默认将整体准确率、宏平均及类别
精确率/召回率/F1、混淆矩阵，以及按眼镜、光照、反光和传感器分组的指标
写入 `dataset/mrl_split/test_metrics.json`。每组同时报告图像数和受试者数。
