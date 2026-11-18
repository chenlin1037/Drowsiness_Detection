"""在 test 集上评估 checkpoint:总体指标 + 分 subject 明细。"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataset import EyeStateDataset
from model import EyeStateClassifier, adapt_state_dict


def evaluate_checkpoint(ckpt_path: str, data_dir: Path, csv_path: Path, batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    dataset = EyeStateDataset(data_dir, csv_path, image_size=ck.get("image_size", 74))
    subjects = [Path(k).name.split("_")[0] for k in dataset.image_keys]
    glasses = None
    with csv_path.open(newline="", encoding="utf-8") as f:
        glasses = [row["glasses"] for row in csv.DictReader(f)]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    preds, labels = [], []
    with torch.inference_mode():
        for images, meta, lbl in loader:
            logits = model(images.to(device), meta.to(device)).squeeze(1)
            preds.extend((logits >= 0).long().cpu().tolist())
            labels.extend(lbl.long().tolist())

    n = len(labels)
    correct = [int(p == l) for p, l in zip(preds, labels)]
    acc = sum(correct) / n
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = n - tp - fp - fn
    print(f"\n=== {ckpt_path} (img={ck.get('image_size', 74)}, se={ck.get('use_se', False)}, "
          f"ms={ck.get('multi_scale', False)}, meta={ck.get('use_metadata', True)}) ===")
    print(f"overall: acc={acc:.4f}  P={tp / (tp + fp):.4f}  R={tp / (tp + fn):.4f}  "
          f"F1={2 * tp / (2 * tp + fp + fn):.4f}  (tp={tp} fp={fp} fn={fn} tn={tn})")

    per_subj = defaultdict(lambda: [0, 0])
    for s, c in zip(subjects, correct):
        per_subj[s][0] += c
        per_subj[s][1] += 1
    print("per-subject:")
    for s in sorted(per_subj):
        c, t = per_subj[s]
        print(f"  {s}: acc={c / t:.4f}  n={t}")
    # 戴眼镜子集
    gl = defaultdict(lambda: [0, 0])
    for g, c in zip(glasses, correct):
        gl[g][0] += c
        gl[g][1] += 1
    for g in sorted(gl):
        c, t = gl[g]
        print(f"glasses={g}: acc={c / t:.4f}  n={t}")


if __name__ == "__main__":
    root = Path("/store2/cx/Drowsiness_Detection/dataset/mrl_split")
    for ckpt in sys.argv[1:]:
        evaluate_checkpoint(ckpt, root / "test", root / "test.csv")
