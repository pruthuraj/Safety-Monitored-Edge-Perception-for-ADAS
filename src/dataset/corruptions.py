"""Synthetic image corruptions for fault injection (Week 7, EXP-010).

Lightweight OpenCV/NumPy implementations (albumentations deliberately not
used — dependency stack stays frozen). Every corruption:

- is deterministic from (frame_id, corruption, severity) — stochastic ones
  derive their RNG seed from that triple, so reruns are bit-identical;
- preserves shape, dtype (uint8), and BGR channel order, so the YOLO
  inference path is unchanged;
- models a plausible degradation mechanism, NOT physically validated
  weather (documented limitation: synthetic fog is uniform haze, not a
  scattering model).

Severities: low / medium / high, monotonically stronger.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

CORRUPTIONS = ("fog", "motion_blur", "gaussian_noise", "low_light", "dead_pixels")
SEVERITIES = ("low", "medium", "high")

# per-corruption severity parameter tables
_FOG_BLEND = {"low": 0.35, "medium": 0.55, "high": 0.75}
_BLUR_KSIZE = {"low": 5, "medium": 11, "high": 19}
_NOISE_SIGMA = {"low": 8.0, "medium": 18.0, "high": 32.0}
_LIGHT_SCALE = {"low": 0.55, "medium": 0.35, "high": 0.18}
_DEAD_FRACTION = {"low": 0.002, "medium": 0.01, "high": 0.03}


def _rng(frame_id: str, corruption: str, severity: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{frame_id}|{corruption}|{severity}|seed42".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _validate(img: np.ndarray, corruption: str, severity: str) -> None:
    if corruption not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {corruption!r}; expected one of {CORRUPTIONS}")
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}; expected one of {SEVERITIES}")
    if img.dtype != np.uint8 or img.ndim != 3:
        raise ValueError(f"expected uint8 HxWx3 image, got {img.dtype} shape {img.shape}")


def fog(img: np.ndarray, severity: str, frame_id: str = "") -> np.ndarray:
    """Uniform haze: blend toward light gray + mild blur (contrast loss)."""
    t = _FOG_BLEND[severity]
    hazed = img.astype(np.float32) * (1.0 - t) + 235.0 * t
    hazed = cv2.GaussianBlur(hazed, (0, 0), sigmaX=1.0 + 2.0 * t)
    return np.clip(hazed, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, severity: str, frame_id: str = "") -> np.ndarray:
    """Horizontal motion blur (camera/vehicle shake)."""
    k = _BLUR_KSIZE[severity]
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, severity: str, frame_id: str = "") -> np.ndarray:
    """Additive sensor noise, seeded per frame/corruption/severity."""
    rng = _rng(frame_id, "gaussian_noise", severity)
    noise = rng.normal(0.0, _NOISE_SIGMA[severity], img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def low_light(img: np.ndarray, severity: str, frame_id: str = "") -> np.ndarray:
    """Underexposure: linear darkening plus mild gamma lift of shadows lost."""
    scaled = img.astype(np.float32) * _LIGHT_SCALE[severity]
    return np.clip(scaled, 0, 255).astype(np.uint8)


def dead_pixels(img: np.ndarray, severity: str, frame_id: str = "") -> np.ndarray:
    """Random dead (black) and hot (white) pixels, seeded deterministic."""
    rng = _rng(frame_id, "dead_pixels", severity)
    out = img.copy()
    h, w = img.shape[:2]
    n = int(_DEAD_FRACTION[severity] * h * w)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    vals = rng.uniform(size=n) < 0.5  # half dead, half hot
    out[ys[vals], xs[vals]] = 0
    out[ys[~vals], xs[~vals]] = 255
    return out


_FUNCS = {
    "fog": fog,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "low_light": low_light,
    "dead_pixels": dead_pixels,
}


def apply_corruption(img: np.ndarray, corruption: str, severity: str, frame_id: str = "") -> np.ndarray:
    """Dispatch a named corruption; validates inputs, preserves shape/dtype."""
    _validate(img, corruption, severity)
    out = _FUNCS[corruption](img, severity, frame_id)
    assert out.shape == img.shape and out.dtype == img.dtype
    return out
