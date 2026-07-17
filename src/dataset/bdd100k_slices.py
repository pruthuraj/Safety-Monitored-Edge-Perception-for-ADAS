"""Deterministic BDD100K OOD slice builder (SR-02, Week 4, EXP-007).

Selects image slices from the BDD100K val split using frame-level
`attributes.weather` and `attributes.timeofday` from the detection labels
JSON (format: https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md).

Slices (target 500 images each, seed 42; if fewer available, use all and
record the count in the manifest):
  bdd-clear-day  timeofday=daytime AND weather=clear   (transfer control, not pure OOD)
  bdd-night      timeofday=night
  bdd-rain       weather=rainy
  bdd-fog        weather=foggy

Slice files (one image filename per line) are written to configs/splits/ and
committed to git — canonical slice definition, same policy as the KITTI
splits. Never regenerate silently.

Usage:
    python -m src.dataset.bdd100k_slices --root data/raw/bdd100k --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

TARGET_PER_SLICE = 500

# slice name -> attribute constraints (all must hold)
SLICE_RULES: dict[str, dict[str, str]] = {
    "bdd-clear-day": {"timeofday": "daytime", "weather": "clear"},
    "bdd-night": {"timeofday": "night"},
    "bdd-rain": {"weather": "rainy"},
    "bdd-fog": {"weather": "foggy"},
}

# candidate label files, first hit wins (legacy 100k format, then det_20)
LABEL_CANDIDATES = (
    "labels/bdd100k_labels_images_val.json",
    "labels/det_20/det_val.json",
    "labels/det_val.json",
)


def find_labels(root: Path) -> Path:
    for rel in LABEL_CANDIDATES:
        p = root / rel
        if p.exists():
            return p
    raise SystemExit(
        f"no BDD100K val label JSON under {root} (tried: {', '.join(LABEL_CANDIDATES)})"
    )


def matches(frame: dict, rules: dict[str, str]) -> bool:
    attrs = frame.get("attributes") or {}
    return all(attrs.get(k) == v for k, v in rules.items())


def build_slices(
    frames: list[dict], seed: int, target: int = TARGET_PER_SLICE
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Filter frames per slice rules and deterministically sample `target` each.

    Returns (slices, available): sampled image-name lists (sorted) and the
    pre-sampling candidate counts.
    """
    slices: dict[str, list[str]] = {}
    available: dict[str, int] = {}
    for name, rules in SLICE_RULES.items():
        candidates = sorted(f["name"] for f in frames if matches(f, rules))
        available[name] = len(candidates)
        rng = random.Random(seed)
        rng.shuffle(candidates)
        slices[name] = sorted(candidates[:target])
    return slices, available


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_slices(root: Path, out_dir: Path, seed: int, target: int = TARGET_PER_SLICE) -> dict:
    label_path = find_labels(root)
    frames = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(frames, list) or not frames:
        raise SystemExit(f"unexpected label JSON structure in {label_path}")

    slices, available = build_slices(frames, seed, target)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "target_per_slice": target,
        "label_source": str(label_path.relative_to(root)),
        "total_val_frames": len(frames),
        "rules": SLICE_RULES,
        "available": available,
        "counts": {},
        "sha256": {},
    }
    for name, images in slices.items():
        f = out_dir / f"{name}.txt"
        f.write_text("\n".join(images) + "\n")
        manifest["counts"][name] = len(images)
        manifest["sha256"][name] = sha256_of(f)

    (out_dir / "bdd_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/bdd100k"))
    ap.add_argument("--out", type=Path, default=Path("configs/splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=int, default=TARGET_PER_SLICE)
    args = ap.parse_args()

    manifest = write_slices(args.root, args.out, args.seed, args.target)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
