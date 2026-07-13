"""Week 2 dataset-pipeline tests (see 'week 2 plan.md' Test Plan)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.kitti_classes import CLASS_MAP, CLASS_NAMES, EXCLUDED_TYPES, KNOWN_TYPES
from src.dataset.kitti_to_yolo import convert_label
from src.dataset.make_splits import make_splits
from src.dataset.validate_kitti import validate

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "configs" / "splits"
KITTI = REPO / "data" / "raw" / "kitti"

KITTI_LINE = (
    "{t} 0.00 0 -0.20 {l:.2f} {top:.2f} {r:.2f} {b:.2f} 1.89 0.48 1.20 1.84 1.47 8.41 0.01"
)


def write_label(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


# --- class mapping -----------------------------------------------------------

def test_mapping_covers_all_known_kitti_types():
    assert KNOWN_TYPES == {
        "Pedestrian", "Person_sitting", "Car", "Van", "Truck", "Cyclist",
        "Misc", "Tram", "DontCare",
    }
    assert set(CLASS_MAP.values()) == {0, 1, 2}
    assert len(CLASS_NAMES) == 3


# --- conversion --------------------------------------------------------------

def test_convert_normalizes_to_unit_range(tmp_path):
    p = write_label(tmp_path, "000000.txt", [
        KITTI_LINE.format(t="Pedestrian", l=712.4, top=143.0, r=810.7, b=307.9),
    ])
    lines, stats = convert_label(p, img_w=1242, img_h=375)
    assert stats == {"kept": 1, "excluded": 0}
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "0"
    for v in (cx, cy, w, h):
        assert 0.0 < float(v) <= 1.0


def test_convert_drops_excluded_types_by_policy(tmp_path):
    p = write_label(tmp_path, "000001.txt", [
        KITTI_LINE.format(t="DontCare", l=10, top=10, r=50, b=50),
        KITTI_LINE.format(t="Tram", l=10, top=10, r=50, b=50),
        KITTI_LINE.format(t="Car", l=100, top=100, r=300, b=250),
    ])
    lines, stats = convert_label(p, img_w=1242, img_h=375)
    assert stats == {"kept": 1, "excluded": 2}
    assert lines[0].startswith("1 ")  # vehicle


def test_convert_fails_loud_on_unmapped_type(tmp_path):
    p = write_label(tmp_path, "000002.txt", [
        KITTI_LINE.format(t="UnicornRider", l=10, top=10, r=50, b=50),
    ])
    with pytest.raises(ValueError, match="UnicornRider"):
        convert_label(p, img_w=1242, img_h=375)


def test_convert_clamps_out_of_bounds_boxes(tmp_path):
    p = write_label(tmp_path, "000003.txt", [
        KITTI_LINE.format(t="Car", l=-15.5, top=120.0, r=1300.0, b=380.0),
    ])
    lines, _ = convert_label(p, img_w=1242, img_h=375)
    _, cx, cy, w, h = lines[0].split()
    assert 0.0 < float(w) <= 1.0 and 0.0 < float(h) <= 1.0
    assert 0.0 <= float(cx) <= 1.0 and 0.0 <= float(cy) <= 1.0


# --- splits (committed files) -------------------------------------------------

@pytest.mark.skipif(not (SPLITS / "train.txt").exists(), reason="splits not generated yet")
def test_committed_splits_have_no_overlap_and_full_coverage():
    ids = {}
    for name in ("train", "val", "test"):
        ids[name] = set((SPLITS / f"{name}.txt").read_text().split())
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert sum(len(v) for v in ids.values()) == 7481


def test_make_splits_is_deterministic(tmp_path):
    label_dir = tmp_path / "training" / "label_2"
    label_dir.mkdir(parents=True)
    for i in range(100):
        (label_dir / f"{i:06d}.txt").write_text("")
    m1 = make_splits(tmp_path, tmp_path / "s1", seed=42)
    m2 = make_splits(tmp_path, tmp_path / "s2", seed=42)
    assert m1["sha256"] == m2["sha256"]
    assert m1["counts"] == {"train": 70, "val": 15, "test": 15}


# --- validator ----------------------------------------------------------------

def test_validator_fails_on_missing_folders(tmp_path):
    report = validate(tmp_path, labels_only=True)
    assert not report["ok"]
    assert any("missing folder" in e for e in report["errors"])


@pytest.mark.skipif(not (KITTI / "training" / "label_2").exists(), reason="KITTI labels not downloaded")
def test_validator_passes_on_real_labels():
    report = validate(KITTI, labels_only=True)
    assert report["num_labels"] == 7481
    assert not any("unmapped" in e for e in report["errors"])
