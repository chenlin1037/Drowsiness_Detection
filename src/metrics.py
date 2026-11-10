
import torch


def calculate_metrics(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> dict[str, float]:
    """
    Calculate binary classification metrics.

    Args:
        tp: True Positive
        tn: True Negative
        fp: False Positive
        fn: False Negative

    Returns:
        Dictionary containing accuracy, precision,
        recall, specificity and f1.
    """

    total = tp + tn + fp + fn

    # 准确率
    accuracy = (
        (tp + tn) / total
        if total > 0 else 0.0
    )

    # 精确率
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0.0
    )

    # 召回率
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    # 特异度
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0.0
    )

    # F1 分数
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def update_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int, int, int]:
    """
    Calculate TP, TN, FP and FN for one batch.

    Args:
        predictions: Boolean tensor of predicted classes.
        targets: Boolean tensor of ground-truth classes.

    Returns:
        Tuple containing tp, tn, fp, fn.
    """

    tp = (predictions & targets).sum().item()

    tn = (~predictions & ~targets).sum().item()

    fp = (predictions & ~targets).sum().item()

    fn = (~predictions & targets).sum().item()

    return tp, tn, fp, fn
    