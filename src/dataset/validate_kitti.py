"""Validate raw KITTI object-detection data before any split/conversion.

Checks (all must pass, exit code 1 otherwise):
  1. Expected folders exist: training/image_2 (unless --labels-only), training/label_2.
  2. Label files parse; every KITTI type is either mapped or explicitly excluded.
  3. Image/label ID sets match exactly (no missing labels, no orphan labels).
  4. Total sample count reported (expected 7481 for full KITTI).

Usage:
    python -m src.dataset.validate_kitti --root data/raw/kitti [--labels-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.dataset.kitti_classes import CLASS_MAP, EXCLUDED_TYPES, KNOWN_TYPES

EXPECTED_TOTAL = 7481


def validate(root: Path, labels_only: bool = False) -> dict:
    errors: list[str] = []
    label_dir = root / "training" / "label_2"
    image_dir = root / "training" / "image_2"

    if not label_dir.is_dir():
        errors.append(f"missing folder: {label_dir}")
    if not labels_only and not image_dir.is_dir():
        errors.append(f"missing folder: {image_dir}")
    if errors:
        return {"ok": False, "errors": errors}

    label_ids = {p.stem for p in label_dir.glob("*.txt")}
    type_counts: Counter[str] = Counter()
    unmapped: set[str] = set()
    bad_lines: list[str] = []

    for p in sorted(label_dir.glob("*.txt")):
        for i, line in enumerate(p.read_text().splitlines()):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 15:
                bad_lines.append(f"{p.name}:{i + 1} has {len(fields)} fields")
                continue
            obj_type = fields[0]
            type_counts[obj_type] += 1
            if obj_type not in KNOWN_TYPES:
                unmapped.add(obj_type)

    if unmapped:
        errors.append(
            f"unmapped KITTI types (not in CLASS_MAP or EXCLUDED_TYPES): {sorted(unmapped)}"
        )
    if bad_lines:
        errors.append(f"malformed label lines: {bad_lines[:10]}")

    report = {
        "root": str(root),
        "num_labels": len(label_ids),
        "expected_total": EXPECTED_TOTAL,
        "type_counts": dict(sorted(type_counts.items())),
        "mapped_types": sorted(CLASS_MAP),
        "excluded_types": sorted(EXCLUDED_TYPES),
        "labels_only": labels_only,
    }

    if not labels_only:
        image_ids = {p.stem for p in image_dir.glob("*.png")}
        report["num_images"] = len(image_ids)
        missing_labels = sorted(image_ids - label_ids)
        orphan_labels = sorted(label_ids - image_ids)
        if missing_labels:
            errors.append(f"{len(missing_labels)} images without labels, e.g. {missing_labels[:5]}")
        if orphan_labels:
            errors.append(f"{len(orphan_labels)} labels without images, e.g. {orphan_labels[:5]}")

    if len(label_ids) != EXPECTED_TOTAL:
        errors.append(f"expected {EXPECTED_TOTAL} labels, found {len(label_ids)}")

    report["ok"] = not errors
    report["errors"] = errors
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/kitti"))
    ap.add_argument("--labels-only", action="store_true", help="skip image checks (images not downloaded yet)")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    args = ap.parse_args()

    report = validate(args.root, labels_only=args.labels_only)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
