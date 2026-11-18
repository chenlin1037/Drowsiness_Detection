"""错样归因分析:对指定 checkpoint 在 test 集上导出 FP/FN,并按 metadata 交叉统计。

用法: python analyze_errors.py runs/checkpoints-3/best.pt
输出: 终端统计报表 + runs/error_analysis/<ckpt名>_errors.csv(错样明细)
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataset import EyeStateDataset

META_FIELDS = ("sex", "glasses", "reflection", "lighting", "sensor")


def load_csv_rows(csv_path: Path) -> dict[str, dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {Path(r["image_path"]).name: r for r in csv.DictReader(f)}


def cross_tab(rows: list[dict], field: str, tag: str) -> None:
    """按字段取值统计错误数与错误率,并给出相对总错误率的提升倍数。"""
    total_err = len(rows)
    if total_err == 0:
        return
    overall = total_err / 12807
    groups: dict[str, dict] = defaultdict(lambda: {"err": 0, "n": 0})
    for r in rows:
        g = groups[r[field]]
        g["err"] += 1
        g["n"] += 1
    # 每组总样本数(含正确样本)由 csv 全量提供
    print(f"\n--- {tag} × {field} ---")
    print(f"{'取值':<8}{'错误数':>8}{'该组错误率':>12}{'组内样本占比':>14}{'风险倍数':>10}")
    for value in sorted(groups):
        g = groups[value]
        n_total = field_totals[field][value]
        rate = g["err"] / n_total
        share = n_total / 12807
        lift = rate / overall
        print(f"{value:<8}{g['err']:>8}{rate:>12.4f}{share:>14.3f}{lift:>10.2f}x")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("/store2/cx/Drowsiness_Detection/dataset/mrl_split")
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/checkpoints-3/best.pt"

    from model import EyeStateClassifier, adapt_state_dict

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

    dataset = EyeStateDataset(root / "test", root / "test.csv", image_size=ck.get("image_size", 74))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    preds, labels, logits_all = [], [], []
    with torch.inference_mode():
        for images, meta, lbl in loader:
            logits = model(images.to(device), meta.to(device)).squeeze(1)
            logits_all.extend(logits.cpu().tolist())
            preds.extend((logits >= 0).long().cpu().tolist())
            labels.extend(lbl.long().tolist())

    csv_rows = load_csv_rows(root / "test.csv")
    assert len(csv_rows) == len(dataset), f"CSV 与数据集数量不一致: {len(csv_rows)} vs {len(dataset)}"

    fp, fn = [], []
    for i, (p, l, logit) in enumerate(zip(preds, labels, logits_all)):
        key = Path(dataset.image_keys[i]).name
        row = dict(csv_rows[key])
        row["logit"] = f"{logit:.4f}"
        if p == 1 and l == 0:
            row["error_type"] = "FP"
            fp.append(row)
        elif p == 0 and l == 1:
            row["error_type"] = "FN"
            fn.append(row)

    n = len(labels)
    print(f"checkpoint: {ckpt_path}  (img={ck.get('image_size')}, se={ck.get('use_se')}, ms={ck.get('multi_scale')})")
    print(f"test n={n}  FP={len(fp)}  FN={len(fn)}  总错误率={(len(fp)+len(fn))/n:.4f}")

    # 全量字段取值计数(用于错误率分母)
    field_totals: dict[str, dict[str, int]] = {f: defaultdict(int) for f in META_FIELDS}
    subj_totals: dict[str, int] = defaultdict(int)
    for row in csv_rows.values():
        for f in META_FIELDS:
            field_totals[f][row[f]] += 1
        subj_totals[Path(row["image_path"]).name.split("_")[0]] += 1

    for tag, rows in (("FP(误报)", fp), ("FN(漏报)", fn)):
        print(f"\n===== {tag}: {len(rows)} 个 =====")
        # 按 subject
        subj_err: dict[str, int] = defaultdict(int)
        for r in rows:
            subj_err[Path(r["image_path"]).name.split("_")[0]] += 1
        print(f"--- {tag} × subject ---")
        for s in sorted(subj_totals):
            if s in subj_err:
                print(f"  {s}: err={subj_err[s]:>4} / n={subj_totals[s]:>5}  err_rate={subj_err[s]/subj_totals[s]:.4f}")
        for f in META_FIELDS:
            cross_tab(rows, f, tag)
        if rows and tag.startswith("FP"):
            logits_fp = [float(r["logit"]) for r in rows]
            logits_fp.sort(reverse=True)
            print(f"\nFP logit 分布: top5={logits_fp[:5]}  中位={logits_fp[len(logits_fp)//2]:.3f}  min={logits_fp[-1]:.3f}")

    # 导出错样明细
    out_dir = Path("runs/error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_path).parent.name
    out_csv = out_dir / f"{stem}_errors.csv"
    fields_out = ["error_type", "image_path", "subject_id", *META_FIELDS, "eye_state", "logit"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_out, extrasaction="ignore")
        w.writeheader()
        for r in fp + fn:
            r["subject_id"] = Path(r["image_path"]).name.split("_")[0]
            w.writerow(r)
    print(f"\n错样明细已保存: {out_csv}")
