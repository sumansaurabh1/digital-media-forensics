"""Offline tests for the NTIRE ZIP training dataset helpers."""

from __future__ import annotations

import csv
import pickle
import zipfile
from functools import partial
from io import BytesIO, StringIO
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from backend.detectors.ai_detector.training import (
    NTIREZipDataset,
    collate_images,
    resolve_class_mapping,
    split_ntire_dataset,
)
from backend.detectors.ai_detector import training


def _jpg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def _shard(root: Path, index: int, rows: list[tuple[str, str]], images: set[str] | None = None, labels: bool = True) -> Path:
    path = root / f"shard_{index}.zip"
    prefix = f"shard_{index}"
    with zipfile.ZipFile(path, "w") as archive:
        if labels:
            content = StringIO(newline="")
            writer = csv.writer(content)
            writer.writerow(["index", "image_name", "label"])
            for row_number, (name, label) in enumerate(rows):
                writer.writerow([row_number, name, label])
            archive.writestr(f"{prefix}/labels.csv", content.getvalue())
        for image_name in images if images is not None else {name for name, _ in rows}:
            archive.writestr(f"{prefix}/images/{image_name}", _jpg_bytes())
    return path


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "ntire"
    root.mkdir()
    _shard(root, 1, [("z.jpg", "1"), ("a.jpg", "0")])
    _shard(root, 0, [("b.jpg", "0"), ("c.jpg", "1")])
    return root


def test_discovery_labels_image_lookup_and_deterministic_order(tmp_path: Path) -> None:
    dataset = NTIREZipDataset(_dataset_root(tmp_path))

    assert [(sample.shard_path.name, sample.image_member, sample.label) for sample in dataset.samples] == [
        ("shard_0.zip", "shard_0/images/b.jpg", 0), ("shard_0.zip", "shard_0/images/c.jpg", 1),
        ("shard_1.zip", "shard_1/images/z.jpg", 1), ("shard_1.zip", "shard_1/images/a.jpg", 0),
    ]
    image, label = dataset[0]
    assert image.mode == "RGB" and label == 0


def test_split_is_deterministic_non_overlapping_and_preserves_classes(tmp_path: Path) -> None:
    dataset = NTIREZipDataset(_dataset_root(tmp_path))
    train_a, valid_a = split_ntire_dataset(dataset, validation_ratio=0.5, seed=4)
    train_b, valid_b = split_ntire_dataset(dataset, validation_ratio=0.5, seed=4)

    assert train_a.indices == train_b.indices and valid_a.indices == valid_b.indices
    assert not set(train_a.indices) & set(valid_a.indices)
    assert {dataset.samples[index].label for index in train_a.indices} == {0, 1}
    assert {dataset.samples[index].label for index in valid_a.indices} == {0, 1}


def test_invalid_structures_and_rows_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No shard"):
        NTIREZipDataset(tmp_path)
    root = tmp_path / "missing-labels"
    root.mkdir()
    _shard(root, 0, [], labels=False)
    with pytest.raises(ValueError, match="Missing labels"):
        NTIREZipDataset(root)
    root = tmp_path / "bad-row"
    root.mkdir()
    _shard(root, 0, [("missing.jpg", "2")], images=set())
    with pytest.raises(ValueError, match="Invalid label"):
        NTIREZipDataset(root)
    root = tmp_path / "missing-image"
    root.mkdir()
    _shard(root, 0, [("missing.jpg", "0")], images=set())
    with pytest.raises(ValueError, match="Missing or unsupported"):
        NTIREZipDataset(root)


class _Processor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros((len(images), 3, 2, 2))}


def test_dataloader_compatibility_and_no_shared_zip_resources(tmp_path: Path) -> None:
    dataset = NTIREZipDataset(_dataset_root(tmp_path))
    batch = next(iter(DataLoader(
        dataset, batch_size=2, num_workers=1, collate_fn=partial(collate_images, processor=_Processor())
    )))

    assert batch["pixel_values"].shape == (2, 3, 2, 2)
    assert batch["labels"].tolist() == [0, 1]
    image, label = dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert image.mode == "RGB" and label == 0
    assert not any(isinstance(value, zipfile.ZipFile) for value in vars(restored).values())
    dataset.close()


def test_archives_are_lazy_reused_per_shard_and_closed(tmp_path: Path, monkeypatch) -> None:
    dataset = NTIREZipDataset(_dataset_root(tmp_path))
    original_zipfile = training.zipfile.ZipFile
    opened: list[Path] = []

    def tracked_zipfile(path, *args, **kwargs):
        opened.append(Path(path))
        return original_zipfile(path, *args, **kwargs)

    monkeypatch.setattr(training.zipfile, "ZipFile", tracked_zipfile)
    dataset[0]
    dataset[1]
    dataset[2]

    assert opened == [dataset.samples[0].shard_path, dataset.samples[2].shard_path]
    assert set(dataset._archives) == set(opened)
    dataset.close()
    assert dataset._archives == {}


def test_model_mapping_uses_configured_labels() -> None:
    model = type("Model", (), {"config": type("Config", (), {"id2label": {3: "AI-generated", 8: "human"}})})()
    assert resolve_class_mapping(model) == {0: 8, 1: 3}
