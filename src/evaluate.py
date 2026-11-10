import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import EyeStateClassifier
from metrics import calculate_metrics
from progress import ProgressBar


def evaluate(
    model: EyeStateClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    progress: ProgressBar | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with torch.inference_mode():
        for images, metadata, labels in loader:
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images, metadata).squeeze(1)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            if progress is not None:
                progress.update(postfix=f", loss {loss.item():.4f}")
            
            
            predictions = logits >= 0
            targets = labels >= 0.5
            tp += (predictions & targets).sum().item()
            fp += (predictions & ~targets).sum().item()
            tn += (~predictions & ~targets).sum().item()
            fn += (~predictions & targets).sum().item()
            
    avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
    metrics = calculate_metrics(tp, tn, fp, fn)
    metrics["eval_loss"] = avg_loss
    return metrics

    