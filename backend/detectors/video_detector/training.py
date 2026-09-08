"""Train a compact video classifier from sampled CapCheck scores."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterator

import joblib
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from backend.core.video_analyzer import probe_video, temporal_features
from backend.detectors.ai_detector.capcheck_detector import CapCheckDetector


DATASET = Path(r"D:\ai-forensics-datasets\RealOrFake\videos.zip")
CHECKPOINT_DIR = Path("models/video-baseline")


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    joblib.dump(state, temporary)
    temporary.replace(path)


def discover_videos(archive_path: str | Path) -> Iterator[tuple[str, int]]:
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            parts = Path(name).parts
            if len(parts) >= 2 and parts[-2] in {"real_256", "fake_256"} and Path(name).suffix.lower() == ".mp4":
                yield name, int(parts[-2] == "fake_256")


def split_videos(samples: list[tuple[str, int]], seed: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    grouped = {0: [], 1: []}
    for sample in samples:
        grouped[sample[1]].append(sample)
    randomizer = random.Random(seed)
    validation: list[tuple[str, int]] = []
    for values in grouped.values():
        randomizer.shuffle(values)
        validation.extend(values[:max(1, round(len(values) * .2))])
    validation_set = set(validation)
    return [sample for sample in samples if sample not in validation_set], validation


def video_features_batch(
    archive_path: str | Path,
    samples: list[tuple[str, int]],
    detector: CapCheckDetector,
    frames: int,
) -> tuple[list[list[float]], list[int]]:
    features, labels, images, counts = [], [], [], []

    with tempfile.TemporaryDirectory(prefix="forensics-training-") as directory, zipfile.ZipFile(archive_path) as archive:
        root = Path(directory)

        for index, (member, label) in enumerate(samples):
            video = root / f"video-{index}.mp4"
            frame_dir = root / f"frames-{index}"
            frame_dir.mkdir()

            try:
                with archive.open(member) as source, video.open("wb") as target:
                    shutil.copyfileobj(source, target)

                duration = float(probe_video(video)["duration"])
                fps = frames / max(duration, .1)
                pattern = str(frame_dir / "frame-%02d.jpg")

                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps}",
                     "-frames:v", str(frames), "-q:v", "2", pattern],
                    capture_output=True,
                    check=True,
                )

                paths = sorted(frame_dir.glob("frame-*.jpg"))
                if not paths:
                    raise ValueError("No frames extracted.")

                batch = [Image.open(path).convert("RGB") for path in paths]
                images.extend(batch)
                counts.append((len(batch), label))
            except Exception as exc:
                print(f"Skipping {member}: {exc}")

        if not images:
            return [], []

        detector._load_model()
        try:
            inputs = detector._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(detector._device) for key, value in inputs.items()}

            with torch.inference_mode():
                probabilities = torch.softmax(detector._model(**inputs).logits, dim=-1)

            scores = [detector._class_scores(row.tolist())[1] for row in probabilities]
        finally:
            for image in images:
                image.close()

        offset = 0
        for count, label in counts:
            features.append(temporal_features(scores[offset:offset + count]))
            labels.append(label)
            offset += count

    return features, labels


def _metrics(model: SGDClassifier, values: list[list[float]], labels: list[int]) -> dict[str, float]:
    probabilities = model.predict_proba(values)[:, 1]
    predictions = (probabilities >= .5).astype(int)
    return {
        "validation_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def train(
    epochs: int = 1,
    frames: int = 12,
    seed: int = 0,
    batch_size: int = 4,
    archive_path: Path = DATASET,
) -> dict[str, Any]:
    samples = list(discover_videos(archive_path))
    if not samples:
        raise ValueError("No real_256/fake_256 MP4 videos found.")

    train_samples, validation_samples = split_videos(samples, seed)
    detector = CapCheckDetector()
    print(f"device={detector._device}, frames_per_video={frames}, batch_size={batch_size}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest = CHECKPOINT_DIR / "latest.joblib"

    state = joblib.load(latest) if latest.exists() else {
        "epoch": 0,
        "history": [],
        "model": SGDClassifier(loss="log_loss", random_state=seed),
    }

    model, history = state["model"], state["history"]
    best_f1 = max((item["f1"] for item in history), default=-1.0)

    try:
        for epoch in range(state["epoch"] + 1, epochs + 1):
            random.Random(seed + epoch).shuffle(train_samples)
            train_values, train_labels = [], []

            total_batches = (len(train_samples) + batch_size - 1) // batch_size

            for start in range(0, len(train_samples), batch_size):
                batch = train_samples[start:start + batch_size]
                values, labels = video_features_batch(archive_path, batch, detector, frames)

                if not labels:
                    continue

                model.partial_fit(
                    values,
                    labels,
                    classes=[0, 1] if not history and not train_values else None,
                )

                train_values.extend(values)
                train_labels.extend(labels)

                batch_no = start // batch_size + 1
                print(f"epoch {epoch}/{epochs} batch {batch_no}/{total_batches} videos={len(train_labels)}", flush=True)

            if not train_labels or len(set(train_labels)) < 2:
                raise ValueError("Insufficient successfully analyzed videos for training.")

            validation_values, validation_labels = [], []

            for start in range(0, len(validation_samples), batch_size):
                values, labels = video_features_batch(
                    archive_path,
                    validation_samples[start:start + batch_size],
                    detector,
                    frames,
                )
                validation_values.extend(values)
                validation_labels.extend(labels)

            if not validation_labels:
                raise ValueError("No validation videos successfully analyzed.")

            train_loss = float(log_loss(train_labels, model.predict_proba(train_values)[:, 1], labels=[0, 1]))
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **_metrics(model, validation_values, validation_labels),
            }

            history.append(metrics)
            checkpoint = {"epoch": epoch, "history": history, "model": model}
            _save_checkpoint(latest, checkpoint)

            if metrics["f1"] >= best_f1:
                _save_checkpoint(CHECKPOINT_DIR / "best.joblib", checkpoint)
                best_f1 = metrics["f1"]

            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"validation_loss={metrics['validation_loss']:.4f} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
            )

    except KeyboardInterrupt:
        if history:
            _save_checkpoint(latest, {"epoch": history[-1]["epoch"], "history": history, "model": model})
        print("Training safely interrupted; latest checkpoint preserved.")

    metadata = {
        "dataset": str(archive_path),
        "frames": frames,
        "batch_size": batch_size,
        "device": str(detector._device),
        "seed": seed,
        "history": history,
    }

    (CHECKPOINT_DIR / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.epochs, args.frames, args.seed, args.batch_size)


if __name__ == "__main__":
    main()