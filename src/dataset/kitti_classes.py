"""KITTI -> project class mapping (single source of truth).

Mapping is defined in docs/project_spec.md and docs/dataset_splits.md.
Any KITTI type not in CLASS_MAP or EXCLUDED_TYPES is an error: the dataset
validator must fail rather than silently map or drop it.
"""

# Target classes (YOLO indices)
CLASS_NAMES = ["pedestrian", "vehicle", "cyclist"]

# KITTI type -> target class index
CLASS_MAP = {
    "Pedestrian": 0,
    "Person_sitting": 0,
    "Car": 1,
    "Van": 1,
    "Truck": 1,
    "Cyclist": 2,
}

# KITTI types deliberately excluded (documented in docs/dataset_splits.md)
EXCLUDED_TYPES = {"Misc", "Tram", "DontCare"}

KNOWN_TYPES = set(CLASS_MAP) | EXCLUDED_TYPES
