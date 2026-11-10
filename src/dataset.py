import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = 74
ALLOWED_VALUES = {
    "sex": ("0", "1"),
    "glasses": ("0", "1"),
    "eye_state": ("0", "1"),
    "reflection": ("0", "1", "2"),
    "lighting": ("0", "1"),
    "sensor": ("01", "02", "03"),
}
REQUIRED_COLUMNS = ("image_path", "subject_id", *ALLOWED_VALUES)

AUGMENT_KEYS = ("hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "flipud", "fliplr")


def build_transform(augment: dict[str, float] | None = None) -> transforms.Compose:
    """Build the image transform; enable YOLO-style augmentation when given.

    灰度图下 hsv_h 无意义被忽略;hsv_s 映射为对比度抖动,hsv_v 映射为亮度抖动。
    """
    ops: list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if augment:
        degrees = float(augment.get("degrees", 0.0) or 0.0)
        translate = float(augment.get("translate", 0.0) or 0.0)
        scale = float(augment.get("scale", 0.0) or 0.0)
        shear = float(augment.get("shear", 0.0) or 0.0)
        if any((degrees, translate, scale, shear)):
            ops.append(
                transforms.RandomAffine(
                    degrees=degrees,
                    translate=(translate, translate) if translate else None,
                    scale=(1.0 - scale, 1.0 + scale) if scale else None,
                    shear=(shear, shear) if shear else None,
                )
            )
        jitter = {}
        if float(augment.get("hsv_v", 0.0) or 0.0):
            jitter["brightness"] = float(augment["hsv_v"])
        if float(augment.get("hsv_s", 0.0) or 0.0):
            jitter["contrast"] = float(augment["hsv_s"])
        if jitter:
            ops.append(transforms.ColorJitter(**jitter))
        if float(augment.get("fliplr", 0.0) or 0.0):
            ops.append(transforms.RandomHorizontalFlip(p=float(augment["fliplr"])))
        if float(augment.get("flipud", 0.0) or 0.0):
            ops.append(transforms.RandomVerticalFlip(p=float(augment["flipud"])))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.449,), std=(0.226,)),
        ]
    )
    return transforms.Compose(ops)


class EyeStateDataset(Dataset):
    """Eye-state dataset driven by a flat image folder and its split CSV.

    The CSV is produced by ``mrl_dataset.py prepare``; ``image_path`` values
    are relative to the split-output root (e.g. ``train/xxx.png``) and labels
    come from the ``eye_state`` column instead of folder names.
    """

    def __init__(
        self,
        split_dir: Path,
        csv_path: Path,
        augment: dict[str, float] | None = None,
    ) -> None:
        self.split_dir = split_dir
        self.transform = build_transform(augment)

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{csv_path}: missing required columns {sorted(missing)}"
                )
            rows = list(reader)
        if not rows:
            raise ValueError(f"{csv_path}: no records found")

        seen: set[str] = set()
        self.samples: list[tuple[Path, list[float], float]] = []
        self.image_keys: list[str] = []
        for row in rows:
            for name, allowed in ALLOWED_VALUES.items():
                value = row[name]
                if value not in allowed:
                    raise ValueError(
                        f"{csv_path}: invalid {name} value {value!r} for "
                        f"{row['image_path']}"
                    )
            image_key = Path(row["image_path"]).as_posix()
            if image_key in seen:
                raise ValueError(f"{csv_path}: duplicate image path {image_key}")
            seen.add(image_key)

            image_path = split_dir / Path(image_key).name
            if not image_path.is_file():
                raise FileNotFoundError(f"Image does not exist: {image_path}")

            metadata = [float(row["sex"]), float(row["glasses"])]
            metadata.extend(
                float(row["reflection"] == category)
                for category in ALLOWED_VALUES["reflection"]
            )
            metadata.append(float(row["lighting"]))
            metadata.extend(
                float(row["sensor"] == category)
                for category in ALLOWED_VALUES["sensor"]
            )
            self.samples.append((image_path, metadata, float(row["eye_state"])))
            self.image_keys.append(image_key)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, metadata, label = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("L"))
        return (
            image_tensor,
            torch.tensor(metadata, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )
