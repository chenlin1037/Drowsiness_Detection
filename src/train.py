import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import EyeStateDataset
from evaluate import evaluate
from model import EyeStateClassifier
from metrics import calculate_metrics, update_confusion_matrix
from progress import ProgressBar
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --augment 未显式覆盖时使用的默认增强参数(参考 YOLO 训练默认值)
AUGMENT_PRESET: dict[str, float] = {
    "hsv_s": 0.3,      # YOLO hsv_s 默认 0.7,灰度图映射为对比度抖动,取较小值
    "hsv_v": 0.4,      # YOLO hsv_v 默认 0.4,映射为亮度抖动
    "degrees": 10.0,   # 随机旋转角度
    "translate": 0.1,  # YOLO 默认 0.1
    "scale": 0.5,      # YOLO 默认 0.5
    "shear": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,     # YOLO 默认 0.5
}


def resolve_augment(args: argparse.Namespace) -> dict[str, float] | None:
    """--augment 启用预设,显式指定的单项参数优先生效;全部为 0 时返回 None。"""
    explicit = {
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
    }
    if not args.augment and all(value is None for value in explicit.values()):
        return None
    resolved = {
        key: value if value is not None else (AUGMENT_PRESET[key] if args.augment else 0.0)
        for key, value in explicit.items()
    }
    resolved = {key: value for key, value in resolved.items() if value}
    return resolved or None


def next_csv_path(runs_dir: Path, name: str) -> Path:
    """同名训练自动版本化: {name}.csv -> {name}-1.csv -> {name}-2.csv ..."""
    path = runs_dir / f"{name}-result.csv"
    suffix = 1
    while path.exists():
        path = runs_dir / f"{name}-result-{suffix}.csv"
        suffix += 1
    return path


def next_ckpt_dir(runs_dir: Path) -> Path:
    """checkpoint 目录自动版本化: checkpoints -> checkpoints-1 -> checkpoints-2 ..."""
    directory = runs_dir / "checkpoints"
    suffix = 1
    while directory.exists():
        directory = runs_dir / f"checkpoints-{suffix}"
        suffix += 1
    return directory




def train_one_epoch(
    model: EyeStateClassifier,
    loader: DataLoader,     # 数据加载器
    criterion: nn.Module,   # 损失函数
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    progress: ProgressBar | None = None,
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    sample_count = 0
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for images, metadata, labels in loader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, metadata).squeeze(1)
        loss_labels = labels
        if label_smoothing > 0:
            # 0/1 标签 -> eps/2 / 1-eps/2
            loss_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        loss = criterion(logits, loss_labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        sample_count += batch_size
        total_loss += loss.item() * batch_size
        predictions = logits >= 0
        targets = labels >= 0.5

        if progress is not None:
            progress.update(postfix=f", loss {loss.item():.4f}")

        batch_tp, batch_tn, batch_fp, batch_fn = update_confusion_matrix(predictions, targets)
        tp += batch_tp
        fp += batch_fp
        tn += batch_tn
        fn += batch_fn
        

    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
        
    metrics = calculate_metrics(tp, tn, fp, fn)
    metrics["train_loss"] = avg_loss

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the eye state classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "mrl_split",
        help="Root containing flat split folders and CSVs (train/, val/, train.csv, val.csv)",
    )
    ## 训练名称
    parser.add_argument(
        "--name",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the best validation checkpoint (默认按 name 自动版本化到 runs/checkpoints[-N]/best.pt)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261105)
    ## 网络架构开关
    parser.add_argument(
        "--img-size",
        type=int,
        default=74,
        help="输入图像分辨率(默认 74,建议 112 配合 SE/多尺度)",
    )
    parser.add_argument(
        "--se",
        action="store_true",
        help="启用 SE 通道注意力(插在 layer3/layer4 之后)",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="启用多尺度特征融合(layer2/3/4 GAP 拼接,896 维)",
    )
    parser.add_argument(
        "--no-spp",
        action="store_false",
        dest="spp",
        help="关闭 SPP 空间金字塔池化,回退为单一 GAP(512 维);默认启用 SPP(1/2/4 bin,10752 维,支持任意输入分辨率)",
    )
    ## 消融实验开关:不加 metadata 拼接
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="消融实验:分类头不拼接 9 维 metadata 特征(默认拼接)",
    )
    ## 额外训练配置
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="标签平滑系数(0 关闭,常用 0.05~0.1)",
    )
    parser.add_argument(
        "--cosine",
        action="store_true",
        help="使用 cosine 退火学习率调度(按 epoch 衰减到 0)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="早停耐心值:val_loss 连续 N 轮无改善则停止(0 关闭)",
    )
    ## 数据增强参数(参考 YOLO;未指定 --augment 且未单独设置时不增强)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="启用数据增强,未显式指定的单项参数使用 YOLO 风格预设值",
    )
    parser.add_argument("--hsv-h", type=float, default=None, help="色调抖动(灰度图不适用,仅记录)")
    parser.add_argument("--hsv-s", type=float, default=None, help="饱和度抖动;灰度图映射为对比度抖动")
    parser.add_argument("--hsv-v", type=float, default=None, help="明度抖动;映射为亮度抖动")
    parser.add_argument("--degrees", type=float, default=None, help="随机旋转角度范围")
    parser.add_argument("--translate", type=float, default=None, help="随机平移比例(相对图像尺寸)")
    parser.add_argument("--scale", type=float, default=None, help="随机缩放比例")
    parser.add_argument("--shear", type=float, default=None, help="随机剪切角度")
    parser.add_argument("--flipud", type=float, default=None, help="上下翻转概率")
    parser.add_argument("--fliplr", type=float, default=None, help="左右翻转概率")
    parser.add_argument(
        "--targeted-augment",
        action="store_true",
        help="追加针对性增强(暗光/模糊/反光,错样归因驱动);需与 --augment 同时使用,仅作用于训练集",
    )
    parser.add_argument(
        "--glint-only",
        action="store_true",
        help="targeted-augment 子集开关:只保留反光斑变换,去掉已证伪的暗光/模糊;需与 --targeted-augment 同时使用",
    )
    parser.add_argument(
        "--spatial-attn",
        action="store_true",
        help="在 layer2/layer3 后插入 CBAM 风格空间门控(近恒等初始化)",
    )
    parser.add_argument(
        "--film",
        action="store_true",
        help="用 FiLM 将 metadata 注入 layer3/layer4 特征(逐通道仿射调制,恒等初始化);需含 metadata",
    )
    parser.add_argument(
        "--weighted-sampling",
        action="store_true",
        help="按困难条件(glasses=1/reflection=2/lighting=0)对训练集加权过采样",
    )
    parser.add_argument(
        "--oversample-weight",
        type=float,
        default=0.5,
        help="加权过采样中每个困难条件的附加权重(默认 0.5,即三条件全中 2.5x)",
    )
    parser.add_argument(
        "--soft-label-csv",
        type=str,
        default=None,
        help="可疑标签 CSV(detect_label_noise.py 输出),命中的样本用软标签训练",
    )
    parser.add_argument(
        "--soft-label-value",
        type=float,
        default=0.2,
        help="软标签软化值 v: 标签 1 重写为 1-v、0 重写为 v,默认 0.2 即 1->0.8 / 0->0.2",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Train without ImageNet pretrained ResNet18 weights",
    )
    
    # 数据增强参数
    
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if args.patience < 0:
        parser.error("--patience cannot be negative")
    if args.img_size < 32:
        parser.error("--img-size must be at least 32")
    if args.targeted_augment and not args.augment:
        parser.error("--targeted-augment 需要与 --augment 同时使用")
    if args.glint_only and not args.targeted_augment:
        parser.error("--glint-only 需要与 --targeted-augment 同时使用")
    if args.film and args.no_metadata:
        parser.error("--film 需要 metadata 输入(不能与 --no-metadata 同时使用)")
    if args.spp and args.multi_scale:
        parser.error("--spp 与 --multi-scale 互斥(SPP 直接作用于 layer4 输出)")
    if not 0 < args.oversample_weight <= 2:
        parser.error("--oversample-weight must be in (0, 2]")
    if not 0 < args.soft_label_value < 1:
        parser.error("--soft-label-value must be in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augment_params = resolve_augment(args)
    if augment_params:
        print("Augmentation enabled: " + ", ".join(f"{k}={v}" for k, v in augment_params.items()))
    if args.targeted_augment:
        if args.glint_only:
            print("Targeted augmentation enabled: glint only(p=0.3), train only")
        else:
            print("Targeted augmentation enabled: low-light(p=0.5) + blur(p=0.3) + glint(p=0.3), train only")
    print(
        "Architecture: "
        + ", ".join([
            f"img_size={args.img_size}",
            f"se={args.se}",
            f"multi_scale={args.multi_scale}",
            f"spp={args.spp}",
            f"metadata={not args.no_metadata}",
        ])
    )
    soft_labels = None
    if args.soft_label_csv:
        soft_csv = Path(args.soft_label_csv)
        if not soft_csv.is_file():
            raise SystemExit(f"--soft-label-csv 文件不存在: {soft_csv}")
        soft_labels = {}
        with soft_csv.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                hard = float(r["label"])
                soft_labels[Path(r["image_path"]).name] = (
                    1.0 - args.soft_label_value if hard >= 0.5 else args.soft_label_value
                )
        print(
            f"Soft labels enabled: {len(soft_labels)} samples from {soft_csv} "
            f"(1.0->{1.0 - args.soft_label_value:.2f}, 0.0->{args.soft_label_value:.2f})"
        )

    train_dataset = EyeStateDataset(
        args.data_dir / "train", args.data_dir / "train.csv",
        augment=augment_params, image_size=args.img_size,
        targeted_augment=args.targeted_augment, glint_only=args.glint_only,
        soft_labels=soft_labels,
    )
    val_dataset = EyeStateDataset(
        args.data_dir / "val", args.data_dir / "val.csv",
        image_size=args.img_size,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = None
    if args.weighted_sampling:
        weights = torch.tensor(train_dataset.condition_weights(args.oversample_weight))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
        print(
            "Weighted sampling enabled: "
            f"per-condition=+{args.oversample_weight}, "
            f"effective epoch size={weights.sum().item() / len(weights):.2f}x"
        )
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = EyeStateClassifier(
        pretrained=args.pretrained,
        use_metadata=not args.no_metadata,
        use_se=args.se,
        multi_scale=args.multi_scale,
        spatial_attn=args.spatial_attn,
        film=args.film,
        use_spp=args.spp,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.cosine else None
    )

    # 同名训练自动版本化: results.csv -> results-1.csv, checkpoints -> checkpoints-1
    runs_dir = PROJECT_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = next_csv_path(runs_dir, args.name)
    output_path = args.output if args.output is not None else next_ckpt_dir(runs_dir) / "best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["epoch", "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1",
         "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"]
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    print(
        f"Device: {device}; train images: {len(train_dataset)}; "
        f"validation images: {len(val_dataset)}"
    )
    print(f"Metrics csv: {csv_path}")
    print(f"Checkpoint : {output_path}")

    for epoch in range(1, args.epochs + 1):
        gpu_mem = ""
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            gpu_mem = f"{torch.cuda.max_memory_allocated() / 2**30:6.2f}G"

        # YOLO 风格表头与进度条
        print(f"{'Epoch':>10}{'GPU_mem':>10}{'loss':>12}{'Instances':>12}{'Size':>8}")
        train_bar = ProgressBar(len(train_loader), desc=f"{epoch:>6}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, criterion, device, optimizer, train_bar,
            label_smoothing=args.label_smoothing,
        )
        train_bar.close()

        print(f"{'':>10}{'Class':>10}{'Images':>10}{'Loss':>12}{'Acc':>9}{'P':>9}{'R':>9}{'F1':>9}")
        val_bar = ProgressBar(len(val_loader), desc=f"{'val':>10}")
        val_metrics = evaluate(model, val_loader, criterion, device, val_bar)
        val_bar.close()
        print(
            f"{'all':>20}{len(val_dataset):>10}"
            f"{val_metrics['eval_loss']:>12.4f}"
            f"{val_metrics['accuracy']:>9.4f}"
            f"{val_metrics['precision']:>9.4f}"
            f"{val_metrics['recall']:>9.4f}"
            f"{val_metrics['f1']:>9.4f}"
        )
        if gpu_mem:
            print(f"GPU peak memory this epoch: {gpu_mem}")
        
        train_loss = train_metrics["train_loss"]
        train_accuracy = train_metrics["accuracy"]
        train_precision = train_metrics["precision"]
        train_recall = train_metrics["recall"]
        train_f1 = train_metrics["f1"]
        val_loss = val_metrics["eval_loss"]
        val_accuracy = val_metrics["accuracy"]
        val_precision = val_metrics["precision"]
        val_recall = val_metrics["recall"]
        val_f1 = val_metrics["f1"]
        
        # 每轮追加写入版本化的 csv
        csv_writer.writerow(
            [epoch, train_loss, train_accuracy, train_precision, train_recall, train_f1,
             val_loss, val_accuracy, val_precision, val_recall, val_f1]
        )
        csv_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "use_metadata": not args.no_metadata,
                    "use_se": args.se,
                    "multi_scale": args.multi_scale,
                    "spatial_attn": args.spatial_attn,
                    "film": args.film,
                    "spp": args.spp,
                    "image_size": args.img_size,
                },
                output_path,
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping: val_loss has not improved for "
                    f"{epochs_without_improvement} epochs (best {best_val_loss:.4f})"
                )
                break

        if scheduler is not None:
            scheduler.step()

    csv_file.close()
    print(f"Training finished. Metrics: {csv_path}; best checkpoint: {output_path}")


if __name__ == "__main__":
    main()
