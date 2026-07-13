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
