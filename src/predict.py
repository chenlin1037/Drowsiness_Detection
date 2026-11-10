import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import EyeStateDataset
from model import EyeStateClassifier


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Predict eye states for a dataset split")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "dataset" / "mrl_split" / "test",
        help="Flat split directory containing the images",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root / "dataset" / "mrl_split" / "test.csv",
        help="Split CSV providing image paths and attribute metadata",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "checkpoints" / "eye_state_classifier.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dataset" / "mrl_split" / "test_predictions.csv",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.csv.is_file():
        raise FileNotFoundError(f"Split CSV not found: {args.csv}")

    dataset = EyeStateDataset(args.data_dir, args.csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = EyeStateClassifier(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    predictions: list[int] = []
    with torch.inference_mode():
        for images, metadata, _ in loader:
            logits = model(
                images.to(device, non_blocking=True),
                metadata.to(device, non_blocking=True),
            ).squeeze(1)
            predictions.extend((logits >= 0).to(torch.int64).cpu().tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "predicted_eye_state"])
        writer.writeheader()
        for image_path, prediction in zip(dataset.image_keys, predictions):
            writer.writerow(
                {
                    "image_path": image_path,
                    "predicted_eye_state": prediction,
                }
            )
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
