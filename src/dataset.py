import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter
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


class RandomLowLight:
    """随机伽马压暗 + 对比度衰减,模拟弱光(lighting=0)条件。"""

    def __init__(
        self,
        p: float = 0.5,
        gamma_range: tuple[float, float] = (0.45, 0.85),
        contrast_range: tuple[float, float] = (0.7, 0.95),
    ):
        self.p = p
        self.gamma_range = gamma_range
        self.contrast_range = contrast_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        gamma = random.uniform(*self.gamma_range)
        img = img.point(lambda v: int(255.0 * (v / 255.0) ** gamma))
        return ImageEnhance.Contrast(img).enhance(random.uniform(*self.contrast_range))


class RandomBlur:
    """轻微高斯/运动模糊,模拟低照度下的运动模糊帧。"""

    def __init__(self, p: float = 0.3, radius_range: tuple[float, float] = (0.4, 1.2)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        if random.random() < 0.5:  # 对角方向运动模糊
            kernel = random.choice(
                ([1, 0, 0, 0, 2, 0, 0, 0, 1], [0, 0, 1, 0, 2, 0, 1, 0, 0])
            )
            return img.filter(ImageFilter.Kernel((3, 3), kernel, scale=3))
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius_range)))


class RandomGlint:
    """随机叠加小高光斑,模拟镜片反光点(reflection=2)。"""

    def __init__(
        self,
        p: float = 0.3,
        radius_range: tuple[float, float] = (2.0, 5.0),
        value_range: tuple[int, int] = (180, 255),
    ):
        self.p = p
        self.radius_range = radius_range
        self.value_range = value_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        overlay = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(overlay)
        for _ in range(random.randint(1, 2)):
            cx = random.uniform(w * 0.2, w * 0.8)
            cy = random.uniform(h * 0.2, h * 0.8)
            r = random.uniform(*self.radius_range)
            draw.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                fill=random.randint(*self.value_range),
            )
        overlay = overlay.filter(ImageFilter.GaussianBlur(1))
        return ImageChops.screen(img, overlay)


def build_transform(
    augment: dict[str, float] | None = None,
    image_size: int = IMAGE_SIZE,
    targeted_augment: bool = False,
    glint_only: bool = False,
) -> transforms.Compose:
    """Build the image transform; enable YOLO-style augmentation when given.

    灰度图下 hsv_h 无意义被忽略;hsv_s 映射为对比度抖动,hsv_v 映射为亮度抖动。
    targeted_augment: 追加暗光/模糊/反光三种针对性变换(错样归因驱动,仅训练集使用)。
    glint_only: targeted_augment 的子集开关,只保留反光斑变换(暗光/模糊已被消融证伪)。
    """
    ops: list = [transforms.Resize((image_size, image_size))]
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
        if targeted_augment:
            if glint_only:
                ops.append(RandomGlint())
            else:
                ops.extend([RandomLowLight(), RandomBlur(), RandomGlint()])
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
        image_size: int = IMAGE_SIZE,
        targeted_augment: bool = False,
        glint_only: bool = False,
        soft_labels: dict[str, float] | None = None,
    ) -> None:
        self.split_dir = split_dir
        self.transform = build_transform(augment, image_size, targeted_augment, glint_only)

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
            # 软标签覆盖:按文件名命中可疑标签表,仅在训练集使用
            label = float(row["eye_state"])
            if soft_labels is not None:
                label = soft_labels.get(Path(image_key).name, label)
            self.samples.append((image_path, metadata, label))
            self.image_keys.append(image_key)

    def __len__(self) -> int:
        return len(self.samples)

    def condition_weights(
        self,
        per_condition: float = 0.5,
    ) -> list[float]:
        """按 metadata 困难条件计算过采样权重(错样归因驱动)。

        对 glasses=1 / reflection=2 / lighting=0 三个困难条件各加 per_condition 权重,
        同时命中多个条件的样本(如 test 上错误率最高的三条件联合)获得最高权重。
        metadata 布局: [sex, glasses, refl==0, refl==1, refl==2, lighting, sen01, sen02, sen03]
        """
        weights = []
        for _, metadata, _ in self.samples:
            w = 1.0
            if metadata[1] == 1.0:
                w += per_condition
            if metadata[4] == 1.0:
                w += per_condition
            if metadata[5] == 1.0:
                w += per_condition
            weights.append(w)
        return weights


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, metadata, label = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("L"))
        return (
            image_tensor,
            torch.tensor(metadata, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )
