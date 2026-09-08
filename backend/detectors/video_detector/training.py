"""Fast video classifier training with cached CapCheck features."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.detectors.ai_detector.capcheck_detector import CapCheckDetector


DATASET = Path(r"D:\ai-forensics-datasets\RealOrFake\videos.zip")
CHECKPOINT_DIR = Path("models/video-baseline")
CACHE = Path("data/video_features.npz")
FEATURE_VERSION = 2


def discover_videos(archive_path: str | Path) -> Iterator[tuple[str, int]]:
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            parts = Path(name).parts
            if len(parts) >= 2 and parts[-2] in {"real_256", "fake_256"} and Path(name).suffix.lower() == ".mp4":
                yield name, int(parts[-2] == "fake_256")


def split_videos(samples, seed):
    grouped = {0: [], 1: []}
    for sample in samples:
        grouped[sample[1]].append(sample)

    rng = random.Random(seed)
    validation = []

    for values in grouped.values():
        rng.shuffle(values)
        validation.extend(values[:max(1, round(len(values) * .2))])

    validation_set = set(validation)
    return [x for x in samples if x not in validation_set], validation


def make_features(scores):
    x = np.asarray(scores, dtype=np.float32)
    d = np.diff(x)

    return [
        float(x.mean()),
        float(np.median(x)),
        float(x.std()),
        float(x.min()),
        float(x.max()),
        float(x.max() - x.min()),
        float(x[-1] - x[0]),
        float(np.percentile(x, 10)),
        float(np.percentile(x, 25)),
        float(np.percentile(x, 75)),
        float(np.percentile(x, 90)),
        float(np.abs(d).mean()) if len(d) else 0.0,
        float(np.abs(d).max()) if len(d) else 0.0,
        float(x[:max(1, len(x) // 2)].mean()),
        float(x[len(x) // 2:].mean()),
    ]


def extract_features(archive_path, samples, detector, frames, batch_size):
    all_features, labels = [], []
    started = time.time()

    with zipfile.ZipFile(archive_path) as archive:
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            images, counts, batch_labels = [], [], []
            skipped = 0

            with tempfile.TemporaryDirectory(prefix="video-train-") as directory:
                root = Path(directory)

                for i, (member, label) in enumerate(batch):
                    video = root / f"{i}.mp4"
                    frame_dir = root / f"f{i}"
                    frame_dir.mkdir()

                    try:
                        with archive.open(member) as src, video.open("wb") as dst:
                            shutil.copyfileobj(src, dst)

                        probe = subprocess.run(
                            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
                            capture_output=True,
                            text=True,
                            check=True,
                        )

                        duration = max(float(probe.stdout.strip()), 0.1)
                        fps = frames / duration
                        pattern = str(frame_dir / "%02d.jpg")

                        subprocess.run(
                            ["ffmpeg", "-y", "-v", "error", "-i", str(video),
                             "-vf", f"fps={fps}", "-frames:v", str(frames),
                             "-q:v", "2", pattern],
                            capture_output=True,
                            check=True,
                        )

                        paths = sorted(frame_dir.glob("*.jpg"))
                        if not paths:
                            raise ValueError("No frames extracted.")

                        batch_images = [Image.open(p).convert("RGB") for p in paths]
                        images.extend(batch_images)
                        counts.append(len(batch_images))
                        batch_labels.append(label)

                    except Exception as exc:
                        skipped += 1
                        print(f"skip {member}: {exc}", flush=True)

                if images:
                    detector._load_model()

                    inputs = detector._processor(images=images, return_tensors="pt")
                    inputs = {k: v.to(detector._device) for k, v in inputs.items()}

                    with torch.inference_mode():
                        probabilities = torch.softmax(detector._model(**inputs).logits, dim=-1)

                    scores = [detector._class_scores(row.tolist())[1] for row in probabilities]

                    offset = 0
                    for count, label in zip(counts, batch_labels):
                        all_features.append(make_features(scores[offset:offset + count]))
                        labels.append(label)
                        offset += count

                    for image in images:
                        image.close()

            elapsed = (time.time() - started) / 60
            print(
                f"features {min(start + batch_size, len(samples))}/{len(samples)} "
                f"skipped={skipped} elapsed={elapsed:.1f}m",
                flush=True,
            )

    return np.asarray(all_features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def cache_matches(seed, frames, dataset):
    if not CACHE.exists():
        return False

    try:
        data = np.load(CACHE, allow_pickle=False)
        return (
            int(data["version"]) == FEATURE_VERSION
            and int(data["frames"]) == frames
            and int(data["seed"]) == seed
            and str(data["dataset"]) == str(dataset)
        )
    except Exception:
        return False


def train(epochs=1, frames=8, seed=0, batch_size=16, archive_path=DATASET):
    started = time.time()

    samples = list(discover_videos(archive_path))
    train_samples, validation_samples = split_videos(samples, seed)

    detector = CapCheckDetector()

    print(
        f"device={detector._device} videos={len(samples)} "
        f"frames={frames} batch={batch_size}",
        flush=True,
    )

    if cache_matches(seed, frames, archive_path):
        data = np.load(CACHE, allow_pickle=False)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        print(f"Using cached features: {CACHE}", flush=True)
    else:
        X_train, y_train = extract_features(
            archive_path, train_samples, detector, frames, batch_size
        )

        X_val, y_val = extract_features(
            archive_path, validation_samples, detector, frames, batch_size
        )

        CACHE.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            CACHE,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            version=FEATURE_VERSION,
            frames=frames,
            seed=seed,
            dataset=str(archive_path),
        )

        print(f"Saved feature cache: {CACHE}", flush=True)

    if len(set(y_train)) < 2 or len(set(y_val)) < 2:
        raise ValueError("Training/validation data does not contain both classes.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "epoch": 1,
        "train_loss": float(
            log_loss(y_train, model.predict_proba(X_train)[:, 1], labels=[0, 1])
        ),
        "validation_loss": float(
            log_loss(y_val, probabilities, labels=[0, 1])
        ),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
    }

    cm = confusion_matrix(y_val, predictions, labels=[0, 1])

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": 1,
        "history": [metrics],
        "model": model,
        "feature_version": FEATURE_VERSION,
        "frames": frames,
    }

    joblib.dump(state, CHECKPOINT_DIR / "latest.joblib")
    joblib.dump(state, CHECKPOINT_DIR / "best.joblib")

    metadata = {
        "dataset": str(archive_path),
        "frames": frames,
        "batch_size": batch_size,
        "device": str(detector._device),
        "seed": seed,
        "feature_version": FEATURE_VERSION,
        "classifier": "StandardScaler+LogisticRegression",
        "train_videos": int(len(y_train)),
        "validation_videos": int(len(y_val)),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "elapsed_minutes": round((time.time() - started) / 60, 2),
    }

    (CHECKPOINT_DIR / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nRESULT")
    print(json.dumps(metrics, indent=2))
    print(f"confusion_matrix=\n{cm}")
    print(f"total_time={metadata['elapsed_minutes']} minutes")
    print(f"best={CHECKPOINT_DIR / 'best.joblib'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.epochs, args.frames, args.seed, args.batch_size)


if __name__ == "__main__":
    main()