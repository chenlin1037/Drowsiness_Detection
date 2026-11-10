#!/usr/bin/env python3
"""Prepare subject-disjoint MRL eye splits and evaluate held-out predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ATTRIBUTES = {
    "eye_state": ("0", "1"),
    "glasses": ("0", "1"),
    "lighting": ("0", "1"),
    "reflection": ("0", "1", "2"),
    "sensor": ("01", "02", "03"),
}
ATTRIBUTE_LABELS = {
    "sex": {"0": "male", "1": "female"},
    "eye_state": {"0": "closed", "1": "open"},
    "glasses": {"0": "no", "1": "yes"},
    "lighting": {"0": "poor", "1": "good"},
    "reflection": {"0": "none", "1": "slight", "2": "strong"},
}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_CANDIDATES = 10_000
DEFAULT_SEED = 20261105


@dataclass(frozen=True)
class ImageRecord:
    image_path: str
    subject_id: str
    sex: str
    glasses: str
    eye_state: str
    reflection: str
    lighting: str
    sensor: str

    def attribute(self, name: str) -> str:
        return str(getattr(self, name))


def parse_image(path: Path, dataset_root: Path) -> ImageRecord:
    fields = path.stem.split("_")
    if len(fields) != 8:
        raise ValueError(f"{path}: expected 8 underscore-separated fields, got {len(fields)}")
    subject_id, _image_id, sex, glasses, eye_state, reflection, lighting, sensor = fields
    allowed = {
        "sex": {"0", "1"},
        "glasses": {"0", "1"},
        "eye_state": {"0", "1"},
        "reflection": {"0", "1", "2"},
        "lighting": {"0", "1"},
        "sensor": {"01", "02", "03"},
    }
    values = {
        "sex": sex,
        "glasses": glasses,
        "eye_state": eye_state,
        "reflection": reflection,
        "lighting": lighting,
        "sensor": sensor,
    }
    for field, value in values.items():
        if value not in allowed[field]:
            raise ValueError(f"{path}: invalid {field} value {value!r}")

    expected_folder = {"0": "Close-Eyes", "1": "Open-Eyes"}[eye_state]
    if path.parent.name != expected_folder:
        raise ValueError(
            f"{path}: folder label {path.parent.name!r} conflicts with "
            f"filename eye_state={eye_state}"
        )

    return ImageRecord(
        image_path=path.relative_to(dataset_root.parent).as_posix(),
        subject_id=subject_id,
        **values,
    )


def load_images(dataset_root: Path) -> list[ImageRecord]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    records = [
        parse_image(path, dataset_root)
        for folder in ("Close-Eyes", "Open-Eyes")
        for path in sorted((dataset_root / folder).glob("*.png"))
    ]
    if not records:
        raise ValueError(f"No PNG images found under {dataset_root}")
    return sorted(records, key=lambda record: record.image_path)


def allocated_split_sizes(subject_count: int) -> dict[str, int]:
    raw = {name: subject_count * ratio for name, ratio in SPLIT_RATIOS.items()}
    sizes = {name: int(value) for name, value in raw.items()}
    remainder = subject_count - sum(sizes.values())
    tie_order = {"train": 0, "test": 1, "val": 2}
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw[name] - sizes[name]), tie_order[name]),
    )
    for name in order[:remainder]:
        sizes[name] += 1
    return sizes


def value_counts(records: list[ImageRecord], attribute: str) -> Counter[str]:
    return Counter(record.attribute(attribute) for record in records)


def score_assignment(
    assignment: dict[str, str],
    subject_counts: dict[str, dict[str, Counter[str]]],
    subject_image_counts: dict[str, int],
    total_counts: dict[str, Counter[str]],
    total_subject_coverage: dict[str, Counter[str]],
    total_images: int,
) -> float:
    total_weight = 0.0
    distribution_score = 0.0
    image_share_error = 0.0
    for split in SPLIT_NAMES:
        split_subjects = [
            subject for subject, value in assignment.items() if value == split
        ]
        split_images = sum(subject_image_counts[subject] for subject in split_subjects)
        split_share = split_images / total_images
        image_share_error += abs(split_share - SPLIT_RATIOS[split])
        for attribute, values in ATTRIBUTES.items():
            weight = 5.0 if attribute == "eye_state" else 1.0
            split_counts = Counter(
                {
                    value: sum(
                        subject_counts[subject][attribute][value]
                        for subject in split_subjects
                    )
                    for value in values
                }
            )
            total_variation = 0.5 * sum(
                abs(
                    split_counts[value] / max(1, split_images)
                    - total_counts[attribute][value] / total_images
                )
                for value in values
            )
            distribution_score += SPLIT_RATIOS[split] * weight * total_variation
            covered_subject_counts = Counter(
                {
                    value: sum(
                        subject_counts[subject][attribute][value] > 0
                        for subject in split_subjects
                    )
                    for value in values
                }
            )
            coverage_total_variation = 0.5 * sum(
                abs(
                    covered_subject_counts[value] / max(1, len(split_subjects))
                    - total_subject_coverage[attribute][value]
                    / max(1, len(subject_counts))
                )
                for value in values
            )
            coverage_weight = 0.75
            distribution_score += (
                SPLIT_RATIOS[split] * coverage_weight * coverage_total_variation
            )
            total_weight += SPLIT_RATIOS[split] * (weight + coverage_weight)
    return distribution_score / total_weight + 1.5 * image_share_error


def choose_split(
    records: list[ImageRecord], candidates: int, seed: int
) -> tuple[dict[str, str], float, int]:
    by_subject: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_subject[record.subject_id].append(record)
    subjects = sorted(by_subject)
    sizes = allocated_split_sizes(len(subjects))
    if any(size == 0 for size in sizes.values()):
        raise ValueError("At least one subject is required in every split")

    total_counts = {
        attribute: value_counts(records, attribute) for attribute in ATTRIBUTES
    }
    subject_counts = {
        subject: {
            attribute: value_counts(group, attribute) for attribute in ATTRIBUTES
        }
        for subject, group in by_subject.items()
    }
    subject_image_counts = {
        subject: len(group) for subject, group in by_subject.items()
    }
    total_subject_coverage = {
        attribute: Counter(
            {
                value: sum(
                    subject_counts[subject][attribute][value] > 0
                    for subject in subjects
                )
                for value in values
            }
        )
        for attribute, values in ATTRIBUTES.items()
    }
    rng = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    best_candidate = -1

    for candidate in range(candidates):
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        assignment = {
            subject: split
            for split, start, end in (
                ("train", 0, sizes["train"]),
                ("val", sizes["train"], sizes["train"] + sizes["val"]),
                ("test", sizes["train"] + sizes["val"], len(subjects)),
            )
            for subject in shuffled[start:end]
        }
        if any(
            not all(
                sum(
                    subject_counts[subject]["eye_state"][value]
                    for subject in subjects
                    if assignment[subject] == split
                )
                for value in ("0", "1")
            )
            for split in SPLIT_NAMES
        ):
            continue
        if any(
            not any(
                subject_counts[subject][attribute][value] > 0
                for subject in subjects
                if assignment[subject] == split
            )
            for split in ("train", "test")
            for attribute, values in ATTRIBUTES.items()
            for value in values
            if total_subject_coverage[attribute][value] >= 2
        ):
            continue
        score = score_assignment(
            assignment,
            subject_counts,
            subject_image_counts,
            total_counts,
            total_subject_coverage,
            len(records),
        )
        if score < best_score:
            best_assignment = assignment
            best_score = score
            best_candidate = candidate

    if best_assignment is None:
        raise RuntimeError(
            f"No candidate split contained both eye states in every split "
            f"after {candidates} attempts; increase --candidates."
        )
    return best_assignment, best_score, best_candidate


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_images(
    output_dir: Path,
    dataset_root: Path,
    copy_plan: list[tuple[str, str]],
) -> None:
    expected_paths = {Path(destination) for _, destination in copy_plan}
    split_dirs = [output_dir / split for split in SPLIT_NAMES]
    existing_paths = {
        path.relative_to(output_dir)
        for split_dir in split_dirs
        if split_dir.is_dir()
        for path in split_dir.rglob("*")
        if path.is_file()
    }
    unexpected_paths = sorted(existing_paths - expected_paths)
    if unexpected_paths:
        examples = ", ".join(str(path) for path in unexpected_paths[:5])
        raise FileExistsError(
            "Existing split folders contain files not present in the generated "
            f"split ({examples}). Images are now copied flat into each split "
            "folder; move or remove the old train/val/test folders before "
            "rebuilding the split."
        )

    for source_path, destination_path in copy_plan:
        source = dataset_root.parent / source_path
        destination = output_dir / destination_path
        if not source.is_file():
            raise FileNotFoundError(f"Source image does not exist: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def prepare(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    records = load_images(dataset_root)
    by_subject: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_subject[record.subject_id].append(record)
    assignment, best_score, best_candidate = choose_split(
        records, args.candidates, args.seed
    )

    csv_fields = [
        "image_path",
        "subject_id",
        "sex",
        "glasses",
        "eye_state",
        "reflection",
        "lighting",
        "sensor",
    ]
    metadata_rows: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    copy_plan: list[tuple[str, str]] = []
    for record in records:
        split = assignment[record.subject_id]
        attributes = {
            "subject_id": record.subject_id,
            "sex": record.sex,
            "glasses": record.glasses,
            "eye_state": record.eye_state,
            "reflection": record.reflection,
            "lighting": record.lighting,
            "sensor": record.sensor,
        }
        metadata_rows.append({"image_path": record.image_path, **attributes})
        destination = f"{split}/{Path(record.image_path).name}"
        split_rows[split].append({"image_path": destination, **attributes})
        copy_plan.append((record.image_path, destination))

    output_dir.mkdir(parents=True, exist_ok=True)
    materialize_images(output_dir, dataset_root, copy_plan)
    write_csv(output_dir / "metadata.csv", csv_fields, metadata_rows)
    for split in SPLIT_NAMES:
        write_csv(output_dir / f"{split}.csv", csv_fields, split_rows[split])

    print(
        f"Prepared {len(records)} images from {len(by_subject)} subjects in "
        f"{output_dir}; selected candidate {best_candidate} "
        f"(score={best_score:.6f})."
    )
    for split in SPLIT_NAMES:
        image_count = len(split_rows[split])
        subject_count = sum(assignment[s] == split for s in by_subject)
        print(
            f"  {split}: {subject_count} subjects, "
            f"{image_count} images ({image_count / len(records):.1%})"
        )


def binary_metrics(
    actual: list[str], predicted: list[str], subject_ids: list[str]
) -> dict[str, Any]:
    confusion = {
        "closed": {"closed": 0, "open": 0},
        "open": {"closed": 0, "open": 0},
    }
    for truth, guess in zip(actual, predicted):
        confusion[{"0": "closed", "1": "open"}[truth]][
            {"0": "closed", "1": "open"}[guess]
        ] += 1

    class_metrics = {}
    for label, positive in (("closed", "0"), ("open", "1")):
        tp = sum(t == positive and p == positive for t, p in zip(actual, predicted))
        fp = sum(t != positive and p == positive for t, p in zip(actual, predicted))
        fn = sum(t == positive and p != positive for t, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(truth == positive for truth in actual),
        }
    macro = {
        metric: sum(class_metrics[label][metric] for label in class_metrics) / 2
        for metric in ("precision", "recall", "f1")
    }
    return {
        "image_count": len(actual),
        "subject_count": len(set(subject_ids)),
        "accuracy": (
            sum(truth == guess for truth, guess in zip(actual, predicted)) / len(actual)
            if actual
            else None
        ),
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "open_precision": class_metrics["open"]["precision"],
        "open_recall": class_metrics["open"]["recall"],
        "open_f1": class_metrics["open"]["f1"],
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
    }


def evaluate(args: argparse.Namespace) -> None:
    split_csv_path = args.manifest.resolve()
    required_columns = {
        "image_path",
        "subject_id",
        "sex",
        "glasses",
        "eye_state",
        "reflection",
        "lighting",
        "sensor",
    }
    with split_csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = required_columns - set(reader.fieldnames or [])
        rows = list(reader)
    if missing_columns:
        raise ValueError(
            f"{split_csv_path}: missing required columns {sorted(missing_columns)}"
        )
    if not rows:
        raise ValueError(f"No records in split CSV: {split_csv_path}")
    expected = {row["image_path"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Split CSV contains duplicate image paths")

    with args.predictions.resolve().open(newline="", encoding="utf-8") as file:
        prediction_rows = list(csv.DictReader(file))
    predictions: dict[str, str] = {}
    for row in prediction_rows:
        image_path = row.get("image_path", "")
        label = row.get("predicted_eye_state", "")
        if image_path in predictions:
            raise ValueError(f"Duplicate prediction for {image_path}")
        if label not in {"0", "1"}:
            raise ValueError(
                f"Invalid predicted_eye_state {label!r} for {image_path}; expected 0 or 1"
            )
        predictions[image_path] = label
    missing = sorted(set(expected) - set(predictions))
    extra = sorted(set(predictions) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Predictions must cover exactly the {split_csv_path.stem} split: "
            f"{len(missing)} missing, {len(extra)} unexpected"
        )

    actual = [row["eye_state"] for row in rows]
    predicted = [predictions[row["image_path"]] for row in rows]
    subject_ids = [row["subject_id"] for row in rows]
    by_attribute: dict[str, list[dict[str, Any]]] = {}
    for attribute in ATTRIBUTES:
        entries = []
        for value in ATTRIBUTES[attribute]:
            indices = [
                i for i, row in enumerate(rows) if row[attribute] == value
            ]
            if not indices:
                continue
            label = ATTRIBUTE_LABELS.get(attribute, {}).get(value, value)
            metrics = binary_metrics(
                [actual[i] for i in indices],
                [predicted[i] for i in indices],
                [subject_ids[i] for i in indices],
            )
            entries.append({"value": value, "label": label, **metrics})
        by_attribute[attribute] = entries

    report = {
        "evaluated_split": split_csv_path.stem,
        "positive_class": "open (eye_state=1)",
        "subjects": sorted(set(subject_ids)),
        "prediction_file": args.predictions.resolve().as_posix(),
        "overall": binary_metrics(actual, predicted, subject_ids),
        "by_attribute": by_attribute,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote metrics for {len(rows)} images to {output_path}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="audit metadata and select a subject-disjoint split"
    )
    prepare_parser.add_argument(
        "--dataset-root", type=Path, default=Path("dataset")
    )
    prepare_parser.add_argument(
        "--output-dir", type=Path, default=Path("dataset/mrl_split")
    )
    prepare_parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_parser.set_defaults(func=prepare)

    eval_parser = subparsers.add_parser(
        "evaluate", help="report metrics for predictions on the test split"
    )
    eval_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/mrl_split/test.csv"),
        help="Split CSV (train/val/test.csv) with image_path and attribute columns",
    )
    eval_parser.add_argument("--predictions", type=Path, required=True)
    eval_parser.add_argument(
        "--output", type=Path, default=Path("dataset/mrl_split/test_metrics.json")
    )
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "candidates", 1) < 1:
        parser.error("--candidates must be at least 1")
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
