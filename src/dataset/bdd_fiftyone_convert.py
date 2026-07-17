"""Convert a FiftyOne BDD100K export (samples.json) to the official-format
attributes-only label JSON expected by src/dataset/bdd100k_slices.py.

Provenance note (Week 4, EXP-007): the official BDD100K mirror
(dl.cv.ethz.ch) is offline, so the val split is obtained from the Hugging
Face dataset `dgural/bdd100k` — a FiftyOne export of the official 100k val
split (10k images) carrying the official frame-level weather / timeofday /
scene attributes as Classification labels. This converter extracts only
{name, attributes} per frame; detection boxes are NOT converted (Week 4
uses attributes for slice filtering only).

Usage:
    python -m src.dataset.bdd_fiftyone_convert \
        --samples data/raw/bdd100k/_hf_dgural/samples.json \
        --out data/raw/bdd100k/labels/bdd100k_labels_images_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert(samples_path: Path) -> list[dict]:
    data = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) else data
    frames = []
    for s in samples:
        attrs = {}
        for key in ("weather", "timeofday", "scene"):
            cls = s.get(key)
            if isinstance(cls, dict) and "label" in cls:
                attrs[key] = cls["label"]
        frames.append({"name": Path(s["filepath"]).name, "attributes": attrs})
    if not frames:
        raise SystemExit(f"no samples found in {samples_path}")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frames = convert(args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frames, indent=1), encoding="utf-8")
    print(f"wrote {len(frames)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
