"""Signal-based image manipulation analysis.

Examines an image for spatial inconsistencies (ELA residuals, noise-level
deviations, edge-density anomalies) that may indicate local editing.

The combined score in [0, 1] represents strength of forensic inconsistency
signals, **not** the probability of manipulation.  A future learned detector
can be added as an additional signal without changing the public interface.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage

from backend.core.schemas import ForensicResult

MODULE_NAME = "manipulation"

_ELA_QUALITY = 90       # JPEG recompression quality for ELA
_BLOCK = 32             # block size for local statistics
_MIN_DIM = 16           # minimum image dimension for analysis
_Z_THRESH = 2.0         # z-score threshold for block anomaly
_MIN_REGION_AREA = 256  # minimum connected-component area (px²)
_MAX_REGIONS = 20       # cap on returned suspicious regions

# Heuristic weights for the three independent signals (must sum to 1).
_W_ELA, _W_NOISE, _W_EDGE = 0.40, 0.35, 0.25


def analyze(image_path: str | Path) -> ForensicResult:
    """Examine *image_path* for forensic signals consistent with manipulation.

    Returns a ``ForensicResult`` with ``module="manipulation"``.  The score
    represents strength of detected inconsistency signals, not a probability.
    """
    path = Path(image_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        with Image.open(path) as pil_img:
            pil_img.load()
            is_jpeg = (pil_img.format or "").upper() in ("JPEG", "JPG", "MPO")
            img_rgb = _to_rgb(pil_img)

        h, w = img_rgb.shape[:2]
        if h < _MIN_DIM or w < _MIN_DIM:
            return ForensicResult(
                module=MODULE_NAME, status="success", score=0.0,
                label="image_too_small", confidence="none",
                evidence=[{"type": "note",
                           "message": f"Image ({w}\u00d7{h}) too small for analysis."}],
            )

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Three independent forensic signals, each returning a spatial map
        # and a scalar anomaly-fraction score.
        ela_map, ela_score = _ela_signal(img_rgb, is_jpeg)
        noise_map, noise_score = _noise_signal(img_rgb, gray)
        edge_map, edge_score = _edge_signal(gray)

        # Combined score from the three signals only (no double-counting via
        # region coverage — regions are derived from these same maps).
        combined = float(np.clip(
            _W_ELA * ela_score + _W_NOISE * noise_score + _W_EDGE * edge_score,
            0.0, 1.0,
        ))

        # Spatial localisation: merge normalised maps, threshold, extract BBs.
        regions = _extract_regions(ela_map, noise_map, edge_map)

        evidence: list[dict[str, Any]] = [
            {"type": "signal", "name": "error_level_analysis",
             "score": round(ela_score, 4), "jpeg_source": is_jpeg},
            {"type": "signal", "name": "noise_residual",
             "score": round(noise_score, 4)},
            {"type": "signal", "name": "edge_texture",
             "score": round(edge_score, 4)},
        ]
        if regions:
            evidence.append({"type": "suspicious_regions",
                             "count": len(regions), "regions": regions})

        return ForensicResult(
            module=MODULE_NAME, status="success", model="signal-heuristic-v1",
            score=round(combined, 4),
            label=_label(combined),
            confidence=_confidence(combined, is_jpeg),
            evidence=evidence,
        )
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return ForensicResult(module=MODULE_NAME, status="error",
                              error=f"Manipulation analysis failed: {exc}")
    except Exception as exc:
        return ForensicResult(module=MODULE_NAME, status="error",
                              error=f"Manipulation analysis failed: {exc}")


# -- preprocessing -----------------------------------------------------------

def _to_rgb(img: Image.Image) -> np.ndarray:
    """Convert any Pillow mode to 3-channel RGB uint8 array."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return np.asarray(bg, dtype=np.uint8)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


# -- signal functions --------------------------------------------------------

def _ela_signal(img_rgb: np.ndarray, is_jpeg: bool) -> tuple[np.ndarray, float]:
    """JPEG recompression residual.

    Regions saved at a different quality or pasted from another source show
    higher residuals after controlled recompression.  For non-JPEG sources
    the signal is still computed (edits can survive format conversion) but
    should be interpreted as a generic recompression residual rather than
    evidence of original JPEG history.
    """
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=_ELA_QUALITY)
    buf.seek(0)
    recomp = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)
    ela_map = np.abs(img_rgb.astype(np.float64) - recomp).mean(axis=2)
    return ela_map, _block_anomaly(ela_map, "mean")


def _noise_signal(img_rgb: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Local noise-level deviation.

    Median-filter subtraction isolates the noise layer; blocks with unusual
    noise std-dev (vs. the global distribution) are flagged.
    """
    smoothed = cv2.cvtColor(cv2.medianBlur(img_rgb, 3), cv2.COLOR_RGB2GRAY)
    noise_map = np.abs(gray.astype(np.float64) - smoothed.astype(np.float64))
    return noise_map, _block_anomaly(noise_map, "std")


def _edge_signal(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Edge-density inconsistency via adaptive Canny."""
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    edge_map = edges.astype(np.float64) / 255.0
    return edge_map, _block_anomaly(edge_map, "mean")


# -- block statistics --------------------------------------------------------

def _block_anomaly(signal_map: np.ndarray, stat: str) -> float:
    """Fraction of blocks whose *stat* deviates beyond ``_Z_THRESH``."""
    h, w = signal_map.shape[:2]
    fn = np.std if stat == "std" else np.mean
    vals = [float(fn(signal_map[y:y + _BLOCK, x:x + _BLOCK]))
            for y in range(0, h - _BLOCK + 1, _BLOCK)
            for x in range(0, w - _BLOCK + 1, _BLOCK)]
    if len(vals) < 2:
        return 0.0
    arr = np.array(vals)
    sigma = float(np.std(arr))
    if sigma < 1e-9:
        return 0.0
    z = np.abs(arr - np.mean(arr)) / sigma
    return float(np.sum(z > _Z_THRESH) / len(vals))


# -- region extraction ------------------------------------------------------

def _normalize(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a, dtype=np.float64) if hi - lo < 1e-9 else (a - lo) / (hi - lo)


def _extract_regions(
    ela_map: np.ndarray, noise_map: np.ndarray, edge_map: np.ndarray,
) -> list[dict[str, int]]:
    """Combine normalised anomaly maps → threshold → connected-component BBs."""
    combined = 0.4 * _normalize(ela_map) + 0.35 * _normalize(noise_map) + 0.25 * _normalize(edge_map)
    thresh = float(np.percentile(combined, 95))
    if thresh <= 0:
        return []

    binary = (combined > thresh).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    labelled, n = ndimage.label(binary)
    regions: list[dict[str, int]] = []
    for i in range(1, n + 1):
        ys, xs = np.where(labelled == i)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        rw, rh = x1 - x0 + 1, y1 - y0 + 1
        if rw * rh >= _MIN_REGION_AREA:
            regions.append({"x": x0, "y": y0, "width": rw, "height": rh})

    regions.sort(key=lambda r: r["width"] * r["height"], reverse=True)
    return regions[:_MAX_REGIONS]


# -- scoring helpers ---------------------------------------------------------

def _confidence(score: float, is_jpeg: bool) -> str:
    """Conservative confidence band — capped at 'medium'."""
    if score < 0.15 or not is_jpeg or score < 0.45:
        return "low"
    return "medium"


def _label(score: float) -> str:
    """Signal-strength label (not a manipulation verdict)."""
    if score < 0.15:
        return "minimal_signals"
    if score < 0.35:
        return "weak_signals"
    if score < 0.60:
        return "moderate_signals"
    return "strong_signals"
