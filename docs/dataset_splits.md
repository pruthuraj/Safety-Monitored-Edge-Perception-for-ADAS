# Dataset Split Notes

Status: Week 4 — KITTI splits committed (EXP-002); kitti-val calibration subsets defined (EXP-006); BDD100K slice policy defined, awaiting local download (EXP-007).

## kitti-val calibration subsets (Week 4, EXP-006)

- **Rule:** `kitti-val` (1122 ids) split 50/50 into `calibration-fit` (561) and
  `calibration-report` (561) by seeded shuffle (seed 42) of the sorted id list —
  implemented in `scripts/evaluate_monitor.py::calib_subsets`, derived at runtime
  from the committed `configs/splits/val.txt` (no separate files; deterministic).
- **Roles:** temperature `T` fitted on `calibration-fit` only; ECE before/after and
  OOD ID-set scoring use `calibration-report` only. `kitti-test` untouched.

## BDD100K slice policy (Week 4, EXP-007)

- **Source:** BDD100K val split — images `data/raw/bdd100k/images/100k/val/`,
  detection labels JSON (frame-level `attributes.weather` / `attributes.timeofday`
  per the [official format doc](https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md)).
  Labels used **only** for attribute filtering in Week 4 — no BDD boxes are evaluated.
- **Builder:** `python -m src.dataset.bdd100k_slices --root data/raw/bdd100k --seed 42`
  → committed slice lists `configs/splits/bdd-*.txt` + `bdd_manifest.json` (counts, hashes).
- **Rules:** `bdd-clear-day` = daytime+clear (transfer control); `bdd-night` = night;
  `bdd-rain` = rainy; `bdd-fog` = foggy. Target 500 images/slice, deterministic
  sample seed 42; if fewer available, all are used and the count recorded as a limitation.
- **Status:** BDD100K not yet downloaded locally — exact counts, label-source file,
  and slice hashes recorded here once `--bdd-slices` runs.

## KITTI acquisition + split record (Week 2)

- **Source:** KITTI object detection benchmark, official AWS mirror
  (`s3.eu-central-1.amazonaws.com/avg-kitti/`), downloaded 2026-07-13.
- **Files:** `data_object_label_2.zip` (5,601,213 bytes), `data_object_image_2.zip` (~12 GB).
- **Location:** `data/raw/kitti/` (untracked).
- **Validation:** `python -m src.dataset.validate_kitti --root data/raw/kitti --labels-only`
  → 7481/7481 labels, zero unmapped types, report at `results/kitti_validation.json`.
- **Label type counts:** Car 28742, Van 2914, Truck 1094, Pedestrian 4487,
  Person_sitting 222, Cyclist 1627; excluded by policy: DontCare 11295, Misc 973, Tram 511.
- **Split command:** `python -m src.dataset.make_splits --root data/raw/kitti --seed 42`
- **Counts:** train 5237 / val 1122 / test 1122 (70/15/15 of 7481).
- **Split file SHA-256** (committed under `configs/splits/`):
  - train: `660d423f58e5ea66d0675f4034e035a59375940d47030b7ec7945997ddebca6c`
  - val: `81c46ef86467fabccb36d45c3228aca9b913475a6d9f1238f49c08113716bd8f`
  - test: `0496ab48e627053e5ff4ba059c23352e7bd40eaf8438f29f648053ba1b24cbba`
- **Preprocessing:** YOLO conversion via `python -m src.dataset.kitti_to_yolo` — hardlinked
  images, per-image-dimension normalized boxes, degenerate boxes (<1 px after clamping) dropped.
- **Excluded frames:** none (all 7481 valid).
- **INT8 calibration rule:** calibration subset drawn from `kitti-train` only, never val/test.

## Roles

| Slice | Source | Role | Used for thresholds? |
|---|---|---|---|
| `kitti-train` | KITTI object detection, deterministic split, seed 42 | Detector training | No |
| `kitti-val` | KITTI, held out from train | ID validation: mAP, calibration fit, monitor thresholds (Q95/Q99) | **Yes — only this** |
| `kitti-test` | KITTI, held out from train+val | Final ID reporting | Never |
| `bdd-clear-day` | BDD100K subset: clear, daytime | Transfer control (domain shift without ODD exit) | No — report only |
| `bdd-night` | BDD100K subset: night | OOD slice (illumination) | No — report only |
| `bdd-rain` | BDD100K subset: rain | OOD slice (weather) | No — report only |
| `bdd-fog` | BDD100K subset: fog | OOD slice (weather) | No — report only |
| `kitti-corrupt-*` | Synthetic corruptions of kitti-test (fog, blur, noise, low light, exposure drift, dead pixels) | Week 7 fault injection | No — report only |
| demo video | Held-out driving clip | Week 6 demo only | No — never in metrics |

## Rules

1. **Split determinism:** split generated once by script with seed 42; file lists committed to repo (`configs/splits/`). Never regenerate silently.
2. **Threshold hygiene:** temperature scaling and Q95/Q99 thresholds fit on `kitti-val` only. Everything else is report-only.
3. **Class mapping:** KITTI → three classes: `Pedestrian`+`Person_sitting` → pedestrian; `Car`+`Van`+`Truck` → vehicle; `Cyclist` → cyclist. `Misc`/`Tram`/`DontCare` excluded. BDD100K mapped analogously (`person`→pedestrian, `car`/`truck`/`bus`→vehicle, `rider`+`bike` context → cyclist; mapping finalized Week 4 and recorded here).
4. **Separation:** ID / shifted / corrupted / demo evaluations never mixed in one table without slice labels.
5. **To record in Week 2:** KITTI version + download date, split file hashes, image counts per slice, preprocessing (resize, letterbox), any excluded frames + reason.

## Open items

- [x] KITTI train/val/test ratios — 70/15/15 confirmed (5237/1122/1122), seed 42
- [ ] BDD100K subset sizes (proposal: ~500 images/slice, balanced by attribute labels) — Week 4
- [ ] INT8 calibration set: subset of `kitti-train` (never val/test) — size set when TRT INT8 attempted
