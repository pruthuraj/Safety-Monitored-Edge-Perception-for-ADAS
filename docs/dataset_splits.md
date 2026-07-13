# Dataset Split Notes

Status: Week 1 plan. Exact file lists + checksums recorded in Week 2 when data is downloaded.

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

- [ ] KITTI train/val/test ratios (proposal: 70/15/15 of the 7,481 labeled images)
- [ ] BDD100K subset sizes (proposal: ~500 images/slice, balanced by attribute labels)
- [ ] INT8 calibration set: subset of `kitti-train` (never val/test) — size TBD Week 2
