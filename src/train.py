import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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




def train_one_epoch(
    model: EyeStateClassifier,
    loader: DataLoader,     # 数据加载器
    criterion: nn.Module,   # 损失函数
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    progress: ProgressBar | None = None,
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
        loss = criterion(logits, labels)
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
        default=PROJECT_ROOT / "runs" / "checkpoints" / "best.pt",
        help="Path for the best validation checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261105)
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
    train_dataset = EyeStateDataset(args.data_dir / "train", args.data_dir / "train.csv", augment=augment_params)
    val_dataset = EyeStateDataset(args.data_dir / "val", args.data_dir / "val.csv")
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = EyeStateClassifier(pretrained=args.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    print(
        f"Device: {device}; train images: {len(train_dataset)}; "
        f"validation images: {len(val_dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        gpu_mem = ""
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            gpu_mem = f"{torch.cuda.max_memory_allocated() / 2**30:6.2f}G"

        # YOLO 风格表头与进度条
        print(f"{'Epoch':>10}{'GPU_mem':>10}{'loss':>12}{'Instances':>12}{'Size':>8}")
        train_bar = ProgressBar(len(train_loader), desc=f"{epoch:>6}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, criterion, device, optimizer, train_bar)
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
        
        # 放进csv文件中,使用训练名称,存放在runs文件夹中
        runs_dir = PROJECT_ROOT / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        with open(runs_dir /"results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            # 表头
            if f.tell() == 0:
                writer.writerow(["epoch", "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"])
            writer.writerow([epoch, train_loss, train_accuracy, train_precision, train_recall, train_f1, val_loss, val_accuracy, val_precision, val_recall, val_f1])
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, train_precision={train_precision:.4f}, train_recall={train_recall:.4f}, train_f1={train_f1:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}, val_precision={val_precision:.4f}, val_recall={val_recall:.4f}, val_f1={val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                args.output,
            )
            # print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
