# Experiment Log

Append-only. One entry per experiment/change. No result may appear in README/paper/CV/demo without an entry here pointing at its command + evidence file.

## Entry template

```markdown
### EXP-NNN — <short title>
- **Date:** YYYY-MM-DD
- **Week:** N
- **What changed:**
- **Why:**
- **Command(s):**
  ```
  <exact command(s), config file, seed>
  ```
- **Environment:** <hardware, driver/CUDA/TensorRT versions if relevant>
- **Result:** <numbers + path to evidence file, e.g. results/baseline_metrics.csv>
- **Limitation:**
- **Next step:**
```

---

### EXP-001 — Week 1 scaffold
- **Date:** 2026-07-13
- **Week:** 1
- **What changed:** Created repo structure, project spec (item + ODD + assumptions + hazards), SR-xx placeholders, traceability skeleton, dataset split notes.
- **Why:** Week 1 deliverable per PLAN.md.
- **Command(s):** n/a (documentation only)
- **Environment:** n/a
- **Result:** `docs/project_spec.md`, `safety/requirements.csv`, `safety/traceability_matrix.csv`, `docs/dataset_splits.md`
- **Limitation:** SR-xx wording is placeholder until Week 3 STPA/HARA. No data downloaded yet.
- **Next step:** Week 2 — KITTI acquisition, deterministic split, YOLOv8n baseline training.

### EXP-002 — KITTI acquisition, validation, deterministic splits
- **Date:** 2026-07-13
- **Week:** 2
- **What changed:** Downloaded KITTI labels (+images in progress) from official AWS mirror; added dataset validator, split generator, KITTI→YOLO converter (3-class mapping), and pipeline tests (9 passing).
- **Why:** Evidence-first baseline requires validated data and committed, deterministic splits before any training.
- **Command(s):**
  ```
  python -m src.dataset.validate_kitti --root data/raw/kitti --labels-only --out results/kitti_validation.json
  python -m src.dataset.make_splits --root data/raw/kitti --seed 42
  python -m pytest tests/ -q
  ```
- **Environment:** Python 3.11.3, torch 2.7.1+cu118 (CUDA available), ultralytics 8.4.93, RTX 3050 Ti Laptop 4 GB, driver 592.27, Windows 11.
- **Result:** 7481/7481 labels valid, zero unmapped types (`results/kitti_validation.json`). Splits 5237/1122/1122 committed under `configs/splits/` with SHA-256 in `manifest.json`. 9/9 tests pass.
- **Limitation:** Image-dependent checks (image/label ID match, YOLO conversion) pending 12 GB image download completion.
- **Next step:** Extract images → full validation → conversion → smoke train (EXP-003).
