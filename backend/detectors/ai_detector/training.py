"""Offline NTIRE ZIP-dataset loading and CapCheck fine-tuning helpers."""

from __future__ import annotations

import csv
import json
import os
import random
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.detectors.ai_detector.capcheck_detector import CapCheckDetector


_IMAGE_SUFFIX = ".jpg"


@dataclass(frozen=True)
class NTIRESample:
    shard_path: Path
    image_member: str
    label: int


class NTIREZipDataset(Dataset[tuple[Image.Image, int]]):
    """Index NTIRE labels while leaving images compressed until requested.

    Each process opens a shard lazily on first use and reuses that handle. The
    cache is excluded from pickling, so Windows DataLoader workers create their
    own independent handles. Call ``close()`` when the dataset is no longer used.
    """

    def __init__(self, root: str | Path, max_samples: int | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"NTIRE dataset root does not exist: {self.root}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")

        shards = sorted(self.root.glob("shard_*.zip"), key=lambda path: path.name)
        if not shards:
            raise ValueError(f"No shard_*.zip files found in: {self.root}")

        samples: list[NTIRESample] = []
        for shard_path in shards:
            samples.extend(self._read_shard(shard_path, max_samples, len(samples)))
            if max_samples is not None and len(samples) >= max_samples:
                break
        if not samples:
            raise ValueError("NTIRE dataset contains no labeled images.")
        self.samples = tuple(samples[:max_samples])
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self._archive_pid = os.getpid()

    @staticmethod
    def _read_shard(shard_path: Path, max_samples: int | None, existing: int) -> list[NTIRESample]:
        prefix = f"{shard_path.stem}/"
        labels_member = f"{prefix}labels.csv"
        samples: list[NTIRESample] = []
        try:
            with zipfile.ZipFile(shard_path) as archive:
                names = set(archive.namelist())
                if labels_member not in names:
                    raise ValueError(f"Missing labels.csv in shard: {shard_path}")
                with archive.open(labels_member) as labels_file:
                    rows = csv.DictReader((line.decode("utf-8") for line in labels_file))
                    if not rows.fieldnames or not {"image_name", "label"}.issubset(rows.fieldnames):
                        raise ValueError(f"Malformed labels.csv in shard: {shard_path}")
                    for row_number, row in enumerate(rows, start=2):
                        image_name = row.get("image_name")
                        label_text = row.get("label")
                        if not image_name or label_text not in {"0", "1"}:
                            raise ValueError(f"Invalid label row {row_number} in shard: {shard_path}")
                        image_member = f"{prefix}images/{image_name}"
                        if Path(image_name).suffix.lower() != _IMAGE_SUFFIX or image_member not in names:
                            raise ValueError(f"Missing or unsupported image '{image_name}' in shard: {shard_path}")
                        samples.append(NTIRESample(shard_path, image_member, int(label_text)))
                        if max_samples is not None and existing + len(samples) >= max_samples:
                            break
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid ZIP shard: {shard_path}") from exc
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        sample = self.samples[index]
        archive = self._archive_for(sample.shard_path)
        try:
            with archive.open(sample.image_member) as image_file:
                with Image.open(BytesIO(image_file.read())) as image:
                    return image.convert("RGB").copy(), sample.label
        except KeyError as exc:
            raise ValueError(f"Referenced image is unavailable: {sample.image_member}") from exc

    def _archive_for(self, shard_path: Path) -> zipfile.ZipFile:
        if self._archive_pid != os.getpid():
            self.close()
            self._archive_pid = os.getpid()
        archive = self._archives.get(shard_path)
        if archive is None or archive.fp is None:
            archive = zipfile.ZipFile(shard_path)
            self._archives[shard_path] = archive
        return archive

    def close(self) -> None:
        """Close cached archives owned by this dataset process."""
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._archives = {}
        self._archive_pid = os.getpid()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def split_ntire_dataset(
    dataset: NTIREZipDataset, validation_ratio: float = 0.2, seed: int = 0
) -> tuple[Subset[NTIREZipDataset], Subset[NTIREZipDataset]]:
    """Return deterministic, non-overlapping stratified train/validation subsets."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    by_label = {0: [], 1: []}
    for index, sample in enumerate(dataset.samples):
        by_label[sample.label].append(index)

    randomizer = random.Random(seed)
    validation: list[int] = []
    for indices in by_label.values():
        shuffled = indices.copy()
        randomizer.shuffle(shuffled)
        count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_ratio))) if len(shuffled) > 1 else 0
        validation.extend(shuffled[:count])
    if not validation and len(dataset) > 1:
        validation.append(len(dataset) - 1)
    validation_set = set(validation)
    train = [index for index in range(len(dataset)) if index not in validation_set]
    if not train or not validation:
        raise ValueError("Dataset is too small to form both training and validation splits.")
    return Subset(dataset, train), Subset(dataset, sorted(validation))


def collate_images(batch: Sequence[tuple[Image.Image, int]], processor: Any) -> dict[str, torch.Tensor]:
    """Apply the CapCheck processor to a dataset batch."""
    images, labels = zip(*batch)
    encoded = processor(images=list(images), return_tensors="pt")
    if "pixel_values" not in encoded:
        raise ValueError("Image processor did not return pixel_values.")
    return {**encoded, "labels": torch.tensor(labels, dtype=torch.long)}


def resolve_class_mapping(model: Any) -> dict[int, int]:
    """Map NTIRE labels (human=0, AI=1) to the model's configured indices."""
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not isinstance(id2label, dict):
        raise ValueError("Model has no id2label mapping.")
    mapping: dict[int, int] = {}
    for index, name in id2label.items():
        normalized = str(name).lower().replace("_", "-").replace(" ", "-")
        if "human" in normalized:
            mapping[0] = int(index)
        elif "ai" in normalized or "artificial" in normalized:
            mapping[1] = int(index)
    if set(mapping) != {0, 1}:
        raise ValueError(f"Incompatible model label mapping: {id2label!r}")
    return mapping


def load_capcheck_training_components() -> tuple[Any, Any, dict[int, int]]:
    """Load the existing CapCheck architecture and processor from local cache only."""
    name = CapCheckDetector.MODEL_NAME
    processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(name, local_files_only=True)
    return processor, model, resolve_class_mapping(model)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-5
    validation_ratio: float = 0.2
    seed: int = 0
    num_workers: int = 0
    checkpoint_dir: Path = Path(tempfile.gettempdir()) / "digital-media-forensics-checkpoint"


def fine_tune_ntire(
    dataset: NTIREZipDataset, processor: Any, model: Any, class_mapping: dict[int, int], config: TrainingConfig
) -> dict[str, Any]:
    """Fine-tune CapCheck on NTIRE labels and save a reloadable checkpoint."""
    if config.epochs <= 0 or config.batch_size <= 0 or config.learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    if set(class_mapping) != {0, 1}:
        raise ValueError("Class mapping must contain NTIRE labels 0 and 1.")
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, validation_set = split_ntire_dataset(dataset, config.validation_ratio, config.seed)
    collate = partial(collate_images, processor=processor)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator,
                              num_workers=config.num_workers, collate_fn=collate, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size, shuffle=False,
                                   num_workers=config.num_workers, collate_fn=collate, pin_memory=device.type == "cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    inverse_mapping = {model_label: ntire_label for ntire_label, model_label in class_mapping.items()}

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, train_count = 0.0, 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            targets = torch.tensor([class_mapping[int(label)] for label in labels.tolist()], device=device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(**inputs).logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(labels)
            train_count += len(labels)
        validation = _validate(model, validation_loader, class_mapping, inverse_mapping, device)
        history.append({"epoch": float(epoch), "train_loss": train_loss / train_count, **validation})

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    metadata = {
        "base_model": CapCheckDetector.MODEL_NAME,
        "epoch": config.epochs,
        "validation_metrics": history[-1],
        "class_mapping": {str(key): value for key, value in class_mapping.items()},
        "training_configuration": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "seed": config.seed,
    }
    (checkpoint_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"device": device.type, "history": history, "checkpoint_dir": str(checkpoint_dir), "metadata": metadata}


def _validate(model: Any, loader: DataLoader[Any], class_mapping: dict[int, int], inverse_mapping: dict[int, int], device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss, total_count = 0.0, 0
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            targets = torch.tensor([class_mapping[int(label)] for label in labels.tolist()], device=device)
            logits = model(**inputs).logits
            total_loss += float(nn.functional.cross_entropy(logits, targets).item()) * len(labels)
            total_count += len(labels)
            actual.extend(labels.tolist())
            try:
                predicted.extend(inverse_mapping[int(value)] for value in logits.argmax(dim=-1).tolist())
            except KeyError as exc:
                raise ValueError("Model predicted an index outside the verified class mapping.") from exc
    return {
        "validation_loss": total_loss / total_count,
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def reload_checkpoint(checkpoint_dir: str | Path) -> dict[int, int]:
    """Reload a checkpoint and verify processor/model compatibility with one forward pass."""
    path = Path(checkpoint_dir)
    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(path, local_files_only=True)
    mapping = resolve_class_mapping(model)
    inputs = processor(images=Image.new("RGB", (32, 32)), return_tensors="pt")
    with torch.inference_mode():
        model(**inputs).logits
    return mapping
