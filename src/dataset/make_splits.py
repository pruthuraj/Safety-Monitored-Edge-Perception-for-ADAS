"""Generate deterministic train/val/test splits for KITTI.

70/15/15 split of all labeled sample IDs, shuffled with a fixed seed.
Split files (one 6-digit ID per line) are written to configs/splits/ and
committed to git — they are the canonical split definition. Never regenerate
silently; if this script must be re-run with different inputs, document why
in docs/experiment_log.md.

Usage:
    python -m src.dataset.make_splits --root data/raw/kitti --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_splits(root: Path, out_dir: Path, seed: int) -> dict:
    label_dir = root / "training" / "label_2"
    ids = sorted(p.stem for p in label_dir.glob("*.txt"))
    if not ids:
        raise SystemExit(f"no label files under {label_dir}")

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * RATIOS[0])
    n_val = round(n * RATIOS[1])
    splits = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }

    # overlap check — hard fail, this is a dataset-hygiene invariant
    assert not set(splits["train"]) & set(splits["val"]), "train/val overlap"
    assert not set(splits["train"]) & set(splits["test"]), "train/test overlap"
    assert not set(splits["val"]) & set(splits["test"]), "val/test overlap"
    assert sum(len(v) for v in splits.values()) == n, "split sizes do not sum to total"

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "total": n, "ratios": RATIOS, "counts": {}, "sha256": {}}
    for name, id_list in splits.items():
        f = out_dir / f"{name}.txt"
        f.write_text("\n".join(id_list) + "\n")
        manifest["counts"][name] = len(id_list)
        manifest["sha256"][name] = sha256_of(f)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/kitti"))
    ap.add_argument("--out", type=Path, default=Path("configs/splits"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = make_splits(args.root, args.out, args.seed)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
