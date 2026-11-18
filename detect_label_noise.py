"""训练/验证集标签噪声审计:用多个异构 checkpoint 联合投票,找出模型高度确信反对标签的样本。

判定规则: >= K 个模型在 |logit| >= CONF 阈值下预测与标注相反 => 可疑标签。
用法: python detect_label_noise.py [--split train] [--conf 2.0] [--votes 3]
输出: runs/error_analysis/<split>_label_noise.csv + 可视化拼图 + 终端统计
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataset import EyeStateDataset
from model import EyeStateClassifier, adapt_state_dict

CKPTS = [
    "runs/checkpoints-3/best.pt",   # #4 112+augment(冠军)
    "runs/checkpoints-7/best.pt",   # #8 112+se+augment
    "runs/checkpoints-8/best.pt",   # #9 112+ms+augment
    "runs/checkpoints-9/best.pt",   # #10 112+no-metadata
]


def load_model(ckpt_path: str, device):
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
    return model, ck.get("image_size", 74)


def infer_logits(model, image_size, split_dir, csv_path, keys, device):
    ds = EyeStateDataset(split_dir, csv_path, image_size=image_size)
    # 与主数据集逐样本对齐:复用同一 csv,顺序一致(shuffle=False)
    assert ds.image_keys == keys, "数据集顺序不一致"
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=True)
    logits = []
    with torch.inference_mode():
        for images, meta, _ in loader:
            out = model(images.to(device), meta.to(device)).squeeze(1)
            logits.extend(out.cpu().tolist())
    return np.array(logits)


def make_mosaic(rows, keys, out_path, n=30):
    cell = 112
    cols, rws = 6, 5
    mosaic = Image.new("L", (cols * cell, rws * (cell + 14)), 0)
    d = ImageDraw.Draw(mosaic)
    for i, r in enumerate(rows[:n]):
        im = Image.open(Path(r["image_path"])).convert("L").resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 14)
        mosaic.paste(im, (x, y))
        d.text((x + 2, y + cell), f"{Path(r['image_path']).name[:14]} v{r['votes']}", fill=255)
    mosaic.save(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--conf", type=float, default=2.0, help="单模型确信阈值 |logit|")
    ap.add_argument("--votes", type=int, default=3, help="判定可疑所需的最少反对票数")
    ap.add_argument("--cache", action="store_true", help="缓存 logits 到 npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path("/store2/cx/Drowsiness_Detection/dataset/mrl_split")
    dataset = EyeStateDataset(root / args.split, root / f"{args.split}.csv")

    cache_file = Path(f"runs/error_analysis/logits_{args.split}.npz")
    if args.cache and cache_file.exists():
        data = np.load(cache_file)
        all_logits, names = data["logits"], data["names"].tolist()
    else:
        all_logits, names = [], []
        for p in CKPTS:
            model, img_size = load_model(p, device)
            lg = infer_logits(model, img_size, root / args.split,
                              root / f"{args.split}.csv", dataset.image_keys, device)
            all_logits.append(lg)
            print(f"done: {p} (img={img_size})")
            del model
            torch.cuda.empty_cache()
        all_logits = np.stack(all_logits, axis=1)  # [N, M]
        names = dataset.image_keys
        if args.cache:
            np.savez_compressed(cache_file, logits=all_logits,
                                names=np.array(names))
    logits = all_logits
    M = logits.shape[1]

    labels = np.array([s[2] for s in dataset.samples])
    keys = [Path(k).name for k in dataset.image_keys]
    with (root / f"{args.split}.csv").open(newline="", encoding="utf-8") as f:
        meta_rows = {Path(r["image_path"]).name: r for r in csv.DictReader(f)}

    pred = (logits >= 0).astype(int)
    confident = (np.abs(logits) >= args.conf)
    disagree = (pred != labels[:, None].astype(int)) & confident
    votes = disagree.sum(axis=1)

    suspect_idx = np.where(votes >= args.votes)[0]
    order = np.argsort(-votes[suspect_idx])
    suspect_idx = suspect_idx[order]

    n = len(labels)
    print(f"\nsplit={args.split}  n={n}  模型数={M}  阈值|logit|>={args.conf}  判定票数>={args.votes}")
    print(f"可疑标签样本: {len(suspect_idx)} ({len(suspect_idx)/n:.4%})")
    # 方向统计
    pred_major = (logits.mean(axis=1) >= 0).astype(int)
    fp_dir = int(((pred_major[suspect_idx] == 1) & (labels[suspect_idx] == 0)).sum())
    fn_dir = int(((pred_major[suspect_idx] == 0) & (labels[suspect_idx] == 1)).sum())
    print(f"  其中 标注=睁眼但模型判闭眼(FP向): {fp_dir}   标注=闭眼但模型判睁眼(FN向): {fn_dir}")

    # 按 subject / metadata 细分
    subj = defaultdict(int)
    fields = ("glasses", "reflection", "lighting", "sensor")
    field_stat = {f: defaultdict(int) for f in fields}
    for i in suspect_idx:
        r = meta_rows[keys[i]]
        subj[Path(keys[i]).name.split("_")[0]] += 1
        for f in fields:
            field_stat[f][r[f]] += 1
    print("  按 subject:", dict(sorted(subj.items())))
    for f in fields:
        print(f"  按 {f}:", dict(sorted(field_stat[f].items())))

    # 导出 + 拼图
    out_dir = Path("runs/error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.split}_label_noise.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "votes", *[f"logit_m{j}" for j in range(M)]])
        for i in suspect_idx:
            w.writerow([dataset.split_dir / keys[i], int(labels[i]), int(votes[i]),
                        *[f"{v:.4f}" for v in logits[i]]])
    print(f"明细已保存: {out_csv}")
    rows = [{"image_path": str(dataset.split_dir / keys[i]), "votes": int(votes[i])}
            for i in suspect_idx]
    mosaic_path = out_dir / f"{args.split}_label_noise_mosaic.png"
    make_mosaic(rows, keys, mosaic_path)
    print(f"拼图已保存: {mosaic_path}")
