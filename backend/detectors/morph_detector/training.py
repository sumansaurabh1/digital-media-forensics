"""Train a dedicated bona-fide-versus-morph image classifier."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


DATA_ROOT = Path(r"D:\ai-forensics-datasets")
MORPH_ROOT = DATA_ROOT / "DiM-FRLL-Morphs" / "morphs"
REAL_ROOT = DATA_ROOT / "FRLL" / "neutral_front"
BACKBONE = Path("models/capcheck-unified")
OUTPUT = Path("models/morph-detector-v2")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Config:
    seed: int = 0
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_ratio: float = 0.2
    num_workers: int = 0


class ImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB").copy(), label


def _valid_images(paths):
    valid = []

    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append(path)
        except (OSError, UnidentifiedImageError):
            pass

    return valid


def discover():
    if not MORPH_ROOT.exists():
        raise RuntimeError(f"Morph dataset not found: {MORPH_ROOT}")

    if not REAL_ROOT.exists():
        raise RuntimeError(f"FRLL dataset not found: {REAL_ROOT}")

    groups = {
        group.name: _valid_images(
            sorted(
                path
                for path in group.rglob("*")
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
        )
        for group in sorted(MORPH_ROOT.iterdir())
        if group.is_dir()
    }

    reals = _valid_images(sorted(REAL_ROOT.glob("*_03.jpg")))

    if not groups or not reals or any(not paths for paths in groups.values()):
        raise RuntimeError(
            "Could not establish valid morph and bona-fide image classes."
        )

    return groups, reals


def _split_groups(groups, reals, config):
    pair_names = sorted(
        {
            path.stem
            for paths in groups.values()
            for path in paths
        }
    )

    identities = sorted(
        {
            path.stem.split("_", 1)[0]
            for path in reals
        }
    )

    rng = random.Random(config.seed)
    rng.shuffle(pair_names)
    rng.shuffle(identities)

    validation_pairs = set(
        pair_names[
            :max(1, round(len(pair_names) * config.validation_ratio))
        ]
    )

    validation_ids = set(
        identities[
            :max(1, round(len(identities) * config.validation_ratio))
        ]
    )

    morph_train = [
        path
        for paths in groups.values()
        for path in paths
        if path.stem not in validation_pairs
    ]

    morph_validation = [
        path
        for paths in groups.values()
        for path in paths
        if path.stem in validation_pairs
    ]

    real_train = [
        path
        for path in reals
        if path.stem.split("_", 1)[0] not in validation_ids
    ]

    real_validation = [
        path
        for path in reals
        if path.stem.split("_", 1)[0] in validation_ids
    ]

    if not all(
        (morph_train, morph_validation, real_train, real_validation)
    ):
        raise RuntimeError(
            "Grouped split did not retain both classes."
        )

    return (
        morph_train,
        morph_validation,
        real_train,
        real_validation,
    )


def _collate(processor, batch):
    images, labels = zip(*batch)

    inputs = processor(
        images=list(images),
        return_tensors="pt",
    )

    return inputs, torch.tensor(labels, dtype=torch.long)


def train(config=Config()):
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {OUTPUT}"
        )

    groups, reals = discover()

    morph_count = sum(len(paths) for paths in groups.values())

    print(f"morph_dataset={MORPH_ROOT}", flush=True)
    print(f"bona_fide_dataset={REAL_ROOT}", flush=True)

    print(
        "morph_groups="
        + json.dumps(
            {name: len(paths) for name, paths in groups.items()}
        ),
        flush=True,
    )

    print(
        f"class_counts={{'bona_fide': {len(reals)}, "
        f"'morph': {morph_count}}}",
        flush=True,
    )

    (
        morph_train,
        morph_validation,
        real_train,
        real_validation,
    ) = _split_groups(groups, reals, config)

    print(
        f"train_counts={{'bona_fide': {len(real_train)}, "
        f"'morph': {len(morph_train)}}}",
        flush=True,
    )

    print(
        f"validation_counts={{'bona_fide': {len(real_validation)}, "
        f"'morph': {len(morph_validation)}}}",
        flush=True,
    )

    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device={device}", flush=True)

    processor = AutoImageProcessor.from_pretrained(
        BACKBONE,
        local_files_only=True,
    )

    model = AutoModelForImageClassification.from_pretrained(
        BACKBONE,
        local_files_only=True,
    )

    classifier = getattr(model, "classifier", None)

    if not isinstance(classifier, nn.Linear):
        raise RuntimeError(
            "Backbone does not expose a replaceable linear classifier."
        )

    model.classifier = nn.Linear(
        classifier.in_features,
        2,
    )

    model.config.num_labels = 2
    model.config.id2label = {
        0: "bona_fide",
        1: "morph",
    }
    model.config.label2id = {
        "bona_fide": 0,
        "morph": 1,
    }

    for parameter in model.base_model.parameters():
        parameter.requires_grad = False

    model.to(device)

    train_samples = (
        [(path, 0) for path in real_train]
        + [(path, 1) for path in morph_train]
    )

    validation_samples = (
        [(path, 0) for path in real_validation]
        + [(path, 1) for path in morph_validation]
    )

    generator = torch.Generator().manual_seed(config.seed)

    collate = lambda batch: _collate(processor, batch)

    train_loader = DataLoader(
        ImageDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        ImageDataset(validation_samples),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    counts = Counter(
        label
        for _, label in train_samples
    )

    total = len(train_samples)

    weights = torch.tensor(
        [
            total / (2 * counts[0]),
            total / (2 * counts[1]),
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"class_weights={weights.detach().cpu().tolist()}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=config.learning_rate,
    )

    best = None

    for epoch in range(1, config.epochs + 1):
        model.train()

        total_loss = 0.0
        batches = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            labels = labels.to(device)

            logits = model(**inputs).logits

            loss = nn.functional.cross_entropy(
                logits,
                labels,
                weight=weights,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_loss = total_loss / max(1, batches)

        model.eval()

        actual = []
        predicted = []

        validation_loss = 0.0
        validation_batches = 0

        with torch.inference_mode():
            for inputs, labels in validation_loader:
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                labels_device = labels.to(device)

                logits = model(**inputs).logits

                loss = nn.functional.cross_entropy(
                    logits,
                    labels_device,
                    weight=weights,
                )

                validation_loss += loss.item()
                validation_batches += 1

                actual.extend(labels.tolist())
                predicted.extend(
                    logits.argmax(dim=-1)
                    .cpu()
                    .tolist()
                )

        validation_loss /= max(1, validation_batches)

        metrics = {
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "accuracy": float(
                accuracy_score(actual, predicted)
            ),
            "precision": float(
                precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                actual,
                predicted,
                labels=[0, 1],
            ).tolist(),
            "per_class": classification_report(
                actual,
                predicted,
                labels=[0, 1],
                target_names=[
                    "bona_fide",
                    "morph",
                ],
                output_dict=True,
                zero_division=0,
            ),
        }

        print(
            f"epoch={epoch} "
            f"train_loss={metrics['train_loss']:.6f} "
            f"validation_loss={metrics['validation_loss']:.6f} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}",
            flush=True,
        )

        print(
            f"confusion_matrix={metrics['confusion_matrix']}",
            flush=True,
        )

        if best is None or metrics["f1"] > best["metrics"]["f1"]:
            best = {
                "epoch": epoch,
                "metrics": metrics,
            }

            OUTPUT.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(OUTPUT)
            processor.save_pretrained(OUTPUT)

            print(
                f"best_checkpoint_saved={OUTPUT}",
                flush=True,
            )

    metadata = {
        "dataset_paths": {
            "morph": str(MORPH_ROOT),
            "bona_fide": str(REAL_ROOT),
        },
        "class_mapping": {
            "0": "bona_fide",
            "1": "morph",
        },
        "class_counts": {
            "bona_fide": len(reals),
            "morph": morph_count,
            "morph_generation_groups": {
                name: len(paths)
                for name, paths in groups.items()
            },
        },
        "train_counts": {
            "bona_fide": len(real_train),
            "morph": len(morph_train),
        },
        "validation_counts": {
            "bona_fide": len(real_validation),
            "morph": len(morph_validation),
        },
        "seed": config.seed,
        "device": str(device),
        "model_backbone": str(BACKBONE),
        "frozen_backbone": True,
        "training_configuration": asdict(config),
        "validation_metrics": best["metrics"],
        "best_epoch": best["epoch"],
        "checkpoint_path": str(OUTPUT),
    }

    (OUTPUT / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"best_epoch={best['epoch']}",
        flush=True,
    )

    print(
        f"best_f1={best['metrics']['f1']:.4f}",
        flush=True,
    )

    print(
        f"checkpoint={OUTPUT}",
        flush=True,
    )

    return metadata


if __name__ == "__main__":
    print(
        json.dumps(
            train(),
            indent=2,
        )
    )
