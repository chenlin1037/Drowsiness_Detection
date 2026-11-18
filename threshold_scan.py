"""判决阈值扫描:对指定 checkpoint 在 test 集上输出 P/R/F1 随阈值变化曲线。

用法: python threshold_scan.py runs/checkpoints-12/best.pt [runs/checkpoints-14/best.pt ...]
输出: 终端关键阈值摘要 + runs/error_analysis/<ckpt名>_threshold_scan.csv(全阈值明细)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataset import EyeStateDataset
from model import EyeStateClassifier, adapt_state_dict


def collect_logits(ckpt_path: str, device) -> tuple[np.ndarray, np.ndarray]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = EyeStateClassifier(
        pretrained=False,
        use_metadata=ck.get("use_metadata", True),
        use_se=ck.get("use_se", False),
        multi_scale=ck.get("multi_scale", False),
        spatial_attn=ck.get("spatial_attn", False),
        film=ck.get("film", False),
    ).to(device)
    model.load_state_dict(adapt_state_dict(ck["model_state_dict"]))
    model.eval()

    root = Path("/store2/cx/Drowsiness_Detection/dataset/mrl_split")
    dataset = EyeStateDataset(root / "test", root / "test.csv",
                              image_size=ck.get("image_size", 74))
    loader = DataLoader(dataset, batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=True)
    logits, labels = [], []
    with torch.inference_mode():
        for images, meta, lbl in loader:
            out = model(images.to(device), meta.to(device)).squeeze(1)
            logits.extend(out.cpu().tolist())
            labels.extend(lbl.long().tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(logits), np.array(labels)


def prf_at(logits: np.ndarray, labels: np.ndarray, t: float) -> tuple[float, float, float, int, int]:
    pred = logits >= t
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, fp, fn


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("/store2/cx/Drowsiness_Detection/dataset/mrl_split")
    n_test = sum(1 for _ in open(root / "test.csv")) - 1

    for ckpt_path in sys.argv[1:]:
        logits, labels = collect_logits(ckpt_path, device)
        assert len(labels) == n_test
        stem = Path(ckpt_path).parent.name

        thresholds = np.unique(np.concatenate([
            np.arange(-4.0, 4.01, 0.05),
            [0.0, logits[np.argsort(logits)[int(0.995 * len(logits))]]],
        ]))
        rows = [prf_at(logits, labels, float(t)) + (float(t),) for t in thresholds]

        out_csv = Path(f"runs/error_analysis/{stem}_threshold_scan.csv")
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "precision", "recall", "f1", "fp", "fn"])
            for p, r, f1, fp, fn, t in rows:
                w.writerow([f"{t:.3f}", f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", fp, fn])

        best_f1 = max(rows, key=lambda x: x[2])
        # P/R 交点附近: |P-R| 最小
        balanced = min(rows, key=lambda x: abs(x[0] - x[1]))
        recall93 = max((x for x in rows if x[1] >= 0.995), key=lambda x: x[0], default=None)

        print(f"\n===== {ckpt_path} =====")
        print(f"默认阈值 0.00: P={prf_at(logits, labels, 0.0)[0]:.4f} R={prf_at(logits, labels, 0.0)[1]:.4f} F1={prf_at(logits, labels, 0.0)[2]:.4f}")
        print(f"最优 F1     t={best_f1[5]:+.2f}: P={best_f1[0]:.4f} R={best_f1[1]:.4f} F1={best_f1[2]:.4f} (FP={best_f1[3]} FN={best_f1[4]})")
        print(f"P=R 均衡点  t={balanced[5]:+.2f}: P={balanced[0]:.4f} R={balanced[1]:.4f} F1={balanced[2]:.4f} (FP={balanced[3]} FN={balanced[4]})")
        if recall93:
            print(f"R>=0.995 内最高P t={recall93[5]:+.2f}: P={recall93[0]:.4f} R={recall93[1]:.4f} F1={recall93[2]:.4f} (FP={recall93[3]} FN={recall93[4]})")
        print(f"全阈值明细: {out_csv}")
