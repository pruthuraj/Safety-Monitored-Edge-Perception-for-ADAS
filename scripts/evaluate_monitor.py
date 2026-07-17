"""evaluate_monitor: calibration + OOD monitor evidence (Weeks 4-5, EXP-006/007/008).

Modes:
  --calibrate       fit temperature on kitti-val calibration-fit subset, report
                    ECE before/after on the calibration-report subset (SR-01)
  --bdd-slices      generate deterministic BDD100K slice files (configs/splits/)
  --ood             score KITTI-val report subset vs BDD slices, export
                    AUROC / FPR@95 per slice and method (SR-02)
  --thresholds      freeze Q95/Q99 thresholds from kitti-val report scores only
                    and select the primary monitor score (Week 5, EXP-008)
  --risk-coverage   sweep thresholds on kitti-val report subset: coverage vs
                    accepted-frame detection quality; BDD coverage at frozen
                    thresholds (Week 5, EXP-008)
  (default)         all of the above, in order

Design notes:
- kitti-val is split 50/50 into calibration-fit / calibration-report with
  seed 42; kitti-test is never touched.
- Detections collected at conf >= 0.05 (below the 0.25 predict default, so
  calibration sees low-confidence detections; above mAP-style 0.001 noise).
- OOD scores are computed from raw post-NMS confidences. Temperature
  scaling is a monotonic per-detection transform, so max-confidence AUROC /
  FPR@95 are identical either way; energy-style top-k aggregation is
  reported on raw confidences and documented as such.

Usage:
    python scripts/evaluate_monitor.py --calibrate
    python scripts/evaluate_monitor.py --bdd-slices
    python scripts/evaluate_monitor.py --ood
    python scripts/evaluate_monitor.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.monitor.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    match_detections,
    reliability_bins,
)
from src.monitor.scoring import auroc, energy_score, fpr_at_tpr, max_conf_score
from src.monitor.thresholds import coverage_at, quantile_thresholds
from src.monitor.detection_metrics import mean_ap, mean_ap_50_95, precision_recall
from src.dataset.bdd100k_slices import SLICE_RULES, write_slices

SEED = 42
IMGSZ = 640
CONF_MIN = 0.05
IOU_THR = 0.50
N_BINS = 15

VAL_SPLIT = REPO / "configs" / "splits" / "val.txt"
KITTI_IMAGES = REPO / "data" / "processed" / "kitti_yolo" / "images" / "val"
KITTI_LABELS = REPO / "data" / "processed" / "kitti_yolo" / "labels" / "val"
BDD_ROOT = REPO / "data" / "raw" / "bdd100k"
BDD_IMAGES = BDD_ROOT / "images" / "100k" / "val"
SPLITS_DIR = REPO / "configs" / "splits"
RESULTS = REPO / "results"

DEFAULT_WEIGHTS = REPO / "runs" / "detect" / "baseline" / "weights" / "best.pt"


def calib_subsets(seed: int = SEED) -> tuple[list[str], list[str]]:
    """Deterministic 50/50 split of kitti-val into (fit, report) id lists."""
    ids = sorted(VAL_SPLIT.read_text().split())
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    return sorted(shuffled[:half]), sorted(shuffled[half:])


def load_gt(image_id: str, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO-format GT label -> (boxes xyxy pixels, class indices)."""
    label = KITTI_LABELS / f"{image_id}.txt"
    boxes, classes = [], []
    if label.exists():
        for line in label.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            boxes.append(
                [
                    (cx - w / 2) * img_w,
                    (cy - h / 2) * img_h,
                    (cx + w / 2) * img_w,
                    (cy + h / 2) * img_h,
                ]
            )
            classes.append(c)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(classes, dtype=int)


def predict_frames(model, image_paths: list[Path]) -> list[dict]:
    """Run detection per image; returns per-frame boxes/classes/confs + shape."""
    frames = []
    for p in image_paths:
        r = model.predict(p, imgsz=IMGSZ, conf=CONF_MIN, verbose=False)[0]
        frames.append(
            {
                "path": p,
                "boxes": r.boxes.xyxy.cpu().numpy(),
                "classes": r.boxes.cls.cpu().numpy().astype(int),
                "confs": r.boxes.conf.cpu().numpy(),
                "orig_h": r.orig_shape[0],
                "orig_w": r.orig_shape[1],
            }
        )
    return frames


def collect_matched(model, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Predict + GT-match a list of kitti-val ids -> (confs, correct)."""
    frames = predict_frames(model, [KITTI_IMAGES / f"{i}.png" for i in ids])
    confs, correct = [], []
    for i, fr in zip(ids, frames):
        gt_boxes, gt_classes = load_gt(i, fr["orig_w"], fr["orig_h"])
        ok = match_detections(fr["boxes"], fr["classes"], fr["confs"], gt_boxes, gt_classes, IOU_THR)
        confs.append(fr["confs"])
        correct.append(ok)
    return np.concatenate(confs), np.concatenate(correct)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_reliability(bins_before: list[dict], bins_after: list[dict], t: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, bins, title in (
        (axes[0], bins_before, "Before temperature scaling"),
        (axes[1], bins_after, f"After temperature scaling (T={t:.3f})"),
    ):
        centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
        accs = [b["accuracy"] for b in bins]
        width = bins[0]["bin_hi"] - bins[0]["bin_lo"]
        ax.bar(centers, accs, width=width * 0.92, color="#4878b0", label="accuracy")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ax.set_xlabel("confidence")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left")
    fig.suptitle("Reliability diagram — kitti-val calibration-report subset")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_calibrate(model, weights: Path) -> dict:
    fit_ids, report_ids = calib_subsets()
    print(f"calibration: fit={len(fit_ids)} report={len(report_ids)} images")

    t0 = time.perf_counter()
    fit_confs, fit_correct = collect_matched(model, fit_ids)
    rep_confs, rep_correct = collect_matched(model, report_ids)
    elapsed = time.perf_counter() - t0

    t = fit_temperature(fit_confs, fit_correct)
    rep_confs_cal = apply_temperature(rep_confs, t)

    ece_before = expected_calibration_error(rep_confs, rep_correct, N_BINS)
    ece_after = expected_calibration_error(rep_confs_cal, rep_correct, N_BINS)
    bins_before = reliability_bins(rep_confs, rep_correct, N_BINS)
    bins_after = reliability_bins(rep_confs_cal, rep_correct, N_BINS)

    RESULTS.mkdir(exist_ok=True)
    params = {
        "experiment": "EXP-006",
        "date": date.today().isoformat(),
        "temperature": round(t, 4),
        "seed": SEED,
        "n_bins": N_BINS,
        "iou_thr": IOU_THR,
        "conf_min": CONF_MIN,
        "imgsz": IMGSZ,
        "weights": str(weights.relative_to(REPO)),
        "fit_images": len(fit_ids),
        "report_images": len(report_ids),
        "fit_detections": int(len(fit_confs)),
        "report_detections": int(len(rep_confs)),
        "fit_ids_range": [fit_ids[0], fit_ids[-1]],
        "report_ids_range": [report_ids[0], report_ids[-1]],
    }
    (RESULTS / "calibration_params.json").write_text(json.dumps(params, indent=2))

    row = {
        "date": params["date"],
        "experiment": "EXP-006",
        "weights": params["weights"],
        "imgsz": IMGSZ,
        "seed": SEED,
        "conf_min": CONF_MIN,
        "iou_thr": IOU_THR,
        "n_bins": N_BINS,
        "fit_images": len(fit_ids),
        "report_images": len(report_ids),
        "fit_detections": int(len(fit_confs)),
        "report_detections": int(len(rep_confs)),
        "temperature": round(t, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "predict_seconds": round(elapsed, 1),
        "command": " ".join(sys.argv),
    }
    append_csv(RESULTS / "calibration_metrics.csv", row)
    plot_reliability(bins_before, bins_after, t, RESULTS / "reliability_diagram.png")

    print(json.dumps(row, indent=2))
    print(f"wrote {RESULTS / 'calibration_metrics.csv'}")
    return row


def run_bdd_slices() -> dict:
    manifest = write_slices(BDD_ROOT, SPLITS_DIR, SEED)
    print(json.dumps(manifest, indent=2))
    return manifest


def frame_scores(frames: list[dict]) -> list[dict]:
    return [
        {
            "frame": fr["path"].name,
            "n_detections": int(len(fr["confs"])),
            "max_conf_score": round(max_conf_score(fr["confs"]), 6),
            "energy_score": round(energy_score(fr["confs"]), 6),
        }
        for fr in frames
    ]


def roc_curve(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(np.concatenate([id_scores, ood_scores]))[::-1]
    tpr = [(ood_scores >= t).mean() for t in thresholds]
    fpr = [(id_scores >= t).mean() for t in thresholds]
    return np.array([0.0, *fpr, 1.0]), np.array([0.0, *tpr, 1.0])


def plot_ood(id_rows: list[dict], slice_rows: dict[str, list[dict]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["max_conf_score", "energy_score"]
    fig, axes = plt.subplots(2, len(methods), figsize=(12, 8))
    for col, method in enumerate(methods):
        id_s = np.array([r[method] for r in id_rows])
        ax = axes[0][col]
        ax.hist(id_s, bins=40, density=True, alpha=0.55, label="kitti-val (ID)")
        for name, rows in slice_rows.items():
            ax.hist(
                np.array([r[method] for r in rows]),
                bins=40,
                density=True,
                alpha=0.45,
                label=name,
            )
        ax.set_title(f"score distribution — {method}")
        ax.legend(fontsize=7)

        ax = axes[1][col]
        for name, rows in slice_rows.items():
            s = np.array([r[method] for r in rows])
            f, t = roc_curve(id_s, s)
            ax.plot(f, t, label=f"{name} (AUROC {auroc(id_s, s):.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("FPR (ID)")
        ax.set_ylabel("TPR (OOD)")
        ax.set_title(f"ROC — {method}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "ood_score_distributions.png", dpi=150)
    # same figure carries both panels; also save ROC-only view for the report
    fig2, axes2 = plt.subplots(1, len(methods), figsize=(12, 4.5))
    for col, method in enumerate(methods):
        id_s = np.array([r[method] for r in id_rows])
        for name, rows in slice_rows.items():
            s = np.array([r[method] for r in rows])
            f, t = roc_curve(id_s, s)
            axes2[col].plot(f, t, label=f"{name} (AUROC {auroc(id_s, s):.3f})")
        axes2[col].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes2[col].set_xlabel("FPR (ID)")
        axes2[col].set_ylabel("TPR (OOD)")
        axes2[col].set_title(f"ROC — {method}")
        axes2[col].legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(RESULTS / "ood_roc_curves.png", dpi=150)
    plt.close("all")


def run_ood(model, weights: Path) -> list[dict]:
    _, report_ids = calib_subsets()
    print(f"OOD: ID = kitti-val report subset ({len(report_ids)} images)")
    id_frames = predict_frames(model, [KITTI_IMAGES / f"{i}.png" for i in report_ids])
    id_rows = frame_scores(id_frames)
    write_rows_csv(RESULTS / "monitor_scores_kitti_val.csv", id_rows)

    slice_rows: dict[str, list[dict]] = {}
    bdd_all_rows: list[dict] = []
    for name in SLICE_RULES:
        slice_file = SPLITS_DIR / f"{name}.txt"
        if not slice_file.exists():
            raise SystemExit(f"missing slice file {slice_file}; run --bdd-slices first")
        images = [BDD_IMAGES / n for n in slice_file.read_text().split()]
        missing = [p for p in images if not p.exists()]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} slice images missing under {BDD_IMAGES}")
        print(f"OOD: scoring {name} ({len(images)} images)")
        frames = predict_frames(model, images)
        rows = frame_scores(frames)
        slice_rows[name] = rows
        bdd_all_rows.extend({"slice": name, **r} for r in rows)
    write_rows_csv(RESULTS / "monitor_scores_bdd100k.csv", bdd_all_rows)

    metrics_rows = []
    for name, rows in slice_rows.items():
        for method in ("max_conf_score", "energy_score"):
            id_s = np.array([r[method] for r in id_rows])
            ood_s = np.array([r[method] for r in rows])
            metrics_rows.append(
                {
                    "date": date.today().isoformat(),
                    "experiment": "EXP-007",
                    "slice": name,
                    "transfer_control": name == "bdd-clear-day",
                    "method": method,
                    "n_id": len(id_s),
                    "n_ood": len(ood_s),
                    "auroc": round(auroc(id_s, ood_s), 4),
                    "fpr_at_95tpr": round(fpr_at_tpr(id_s, ood_s), 4),
                    "weights": str(weights.relative_to(REPO)),
                    "conf_min": CONF_MIN,
                    "seed": SEED,
                    "command": " ".join(sys.argv),
                }
            )
    write_rows_csv(RESULTS / "ood_metrics.csv", metrics_rows)
    plot_ood(id_rows, slice_rows)

    print(json.dumps(metrics_rows, indent=2))
    print(f"wrote {RESULTS / 'ood_metrics.csv'}")
    return metrics_rows


METHODS = ("max_conf_score", "energy_score")
MATERIALITY_MARGIN = 0.01  # energy must beat max-conf mean AUROC by this to become primary


def read_scores_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run --ood first")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def run_thresholds() -> dict:
    """Freeze Q95/Q99 from kitti-val report scores; select primary method."""
    id_rows = read_scores_csv(RESULTS / "monitor_scores_kitti_val.csv")
    ood_rows = read_scores_csv(RESULTS / "ood_metrics.csv")

    thresholds = {
        m: quantile_thresholds(np.array([float(r[m]) for r in id_rows])) for m in METHODS
    }

    # primary selection: mean AUROC over true OOD slices (transfer control excluded);
    # energy wins only if materially better, else simpler max-conf (plan policy)
    mean_auroc = {}
    for m in METHODS:
        vals = [
            float(r["auroc"])
            for r in ood_rows
            if r["method"] == m and r["transfer_control"] == "False"
        ]
        mean_auroc[m] = round(float(np.mean(vals)), 4)
    energy_gain = mean_auroc["energy_score"] - mean_auroc["max_conf_score"]
    primary = "energy_score" if energy_gain >= MATERIALITY_MARGIN else "max_conf_score"

    out = {
        "experiment": "EXP-008",
        "date": date.today().isoformat(),
        "seed": SEED,
        "source_scores": "results/monitor_scores_kitti_val.csv (kitti-val calibration-report subset only)",
        "policy": "Q95=DEGRADED candidate, Q99=FAIL_SAFE_REQUEST candidate; score>threshold = suspect; thresholds from KITTI validation only, never BDD/kitti-test",
        "n_id_frames": len(id_rows),
        "thresholds": thresholds,
        "mean_ood_auroc_excl_transfer_control": mean_auroc,
        "primary_method": primary,
        "primary_justification": (
            f"energy mean AUROC gain {energy_gain:+.4f} vs max-conf is below the "
            f"{MATERIALITY_MARGIN} materiality margin; max_conf_score chosen as simpler "
            "and easier to argue in the safety case"
            if primary == "max_conf_score"
            else f"energy mean AUROC gain {energy_gain:+.4f} exceeds the {MATERIALITY_MARGIN} materiality margin"
        ),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "monitor_thresholds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {RESULTS / 'monitor_thresholds.json'}")
    return out


def collect_frames_with_gt(model, ids: list[str]) -> list[dict]:
    frames = predict_frames(model, [KITTI_IMAGES / f"{i}.png" for i in ids])
    for i, fr in zip(ids, frames):
        fr["gt_boxes"], fr["gt_classes"] = load_gt(i, fr["orig_w"], fr["orig_h"])
    return frames


def run_risk_coverage(model, weights: Path) -> list[dict]:
    thr_path = RESULTS / "monitor_thresholds.json"
    if not thr_path.exists():
        raise SystemExit(f"missing {thr_path}; run --thresholds first")
    frozen = json.loads(thr_path.read_text())

    _, report_ids = calib_subsets()
    print(f"risk-coverage: kitti-val report subset ({len(report_ids)} images)")
    frames = collect_frames_with_gt(model, report_ids)
    scores = {
        "max_conf_score": np.array([max_conf_score(fr["confs"]) for fr in frames]),
        "energy_score": np.array([energy_score(fr["confs"]) for fr in frames]),
    }

    rows = []
    sweep_quantiles = [round(0.05 * i, 2) for i in range(1, 21)]  # 0.05 .. 1.00
    for m in METHODS:
        named = {f"sweep-q{int(q * 100):03d}": float(np.quantile(scores[m], q)) for q in sweep_quantiles}
        named["frozen-q95"] = frozen["thresholds"][m]["q95"]
        named["frozen-q99"] = frozen["thresholds"][m]["q99"]
        for label, thr in sorted(named.items(), key=lambda kv: kv[1]):
            accepted = [fr for fr, s in zip(frames, scores[m]) if s <= thr]
            row = {
                "date": date.today().isoformat(),
                "experiment": "EXP-008",
                "slice": "kitti-val-report",
                "method": m,
                "threshold_label": label,
                "threshold": round(float(thr), 6),
                "n_frames": len(frames),
                "n_accepted": len(accepted),
                "accepted_fraction": round(len(accepted) / len(frames), 4),
                "rejected_fraction": round(1.0 - len(accepted) / len(frames), 4),
                "accepted_mAP50": "",
                "accepted_mAP50_95": "",
                "accepted_precision": "",
                "accepted_recall": "",
            }
            if accepted:
                p, r = precision_recall(accepted)
                row.update(
                    accepted_mAP50=round(mean_ap(accepted), 4),
                    accepted_mAP50_95=round(mean_ap_50_95(accepted), 4),
                    accepted_precision=round(p, 4),
                    accepted_recall=round(r, 4),
                )
            rows.append(row)

    # BDD slices: attribute labels only, no detection GT -> coverage/rejection only
    bdd_rows = read_scores_csv(RESULTS / "monitor_scores_bdd100k.csv")
    for name in SLICE_RULES:
        for m in METHODS:
            s = np.array([float(r[m]) for r in bdd_rows if r["slice"] == name])
            if s.size == 0:
                continue
            for label in ("frozen-q95", "frozen-q99"):
                thr = frozen["thresholds"][m][label.split("-")[1]]
                cov = coverage_at(s, thr)
                rows.append(
                    {
                        "date": date.today().isoformat(),
                        "experiment": "EXP-008",
                        "slice": name,
                        "method": m,
                        "threshold_label": label,
                        "threshold": round(float(thr), 6),
                        "n_frames": int(s.size),
                        "n_accepted": int(round(cov * s.size)),
                        "accepted_fraction": round(cov, 4),
                        "rejected_fraction": round(1.0 - cov, 4),
                        "accepted_mAP50": "",
                        "accepted_mAP50_95": "",
                        "accepted_precision": "",
                        "accepted_recall": "",
                    }
                )

    write_rows_csv(RESULTS / "risk_coverage.csv", rows)
    plot_risk_coverage(rows, frozen)
    print(f"wrote {RESULTS / 'risk_coverage.csv'}")
    return rows


def plot_risk_coverage(rows: list[dict], frozen: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, ylabel in (
        (axes[0], "accepted_mAP50", "accepted-frame mAP50"),
        (axes[1], "accepted_recall", "accepted-frame recall (conf>=0.25)"),
    ):
        for m in METHODS:
            pts = sorted(
                (
                    (float(r["accepted_fraction"]), float(r[metric]))
                    for r in rows
                    if r["slice"] == "kitti-val-report" and r["method"] == m and r[metric] != ""
                ),
            )
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label=m)
            for label, style in (("frozen-q95", ":"), ("frozen-q99", "--")):
                fr_row = next(
                    r
                    for r in rows
                    if r["slice"] == "kitti-val-report"
                    and r["method"] == m
                    and r["threshold_label"] == label
                )
                ax.axvline(float(fr_row["accepted_fraction"]), linestyle=style, alpha=0.4)
        ax.set_xlabel("coverage (accepted fraction)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.set_title(f"{ylabel} vs coverage — kitti-val report subset")
    fig.suptitle(
        f"Risk-coverage (primary: {frozen['primary_method']}; dotted=Q95, dashed=Q99)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "risk_coverage.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--bdd-slices", action="store_true")
    ap.add_argument("--ood", action="store_true")
    ap.add_argument("--thresholds", action="store_true")
    ap.add_argument("--risk-coverage", action="store_true")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = ap.parse_args()

    run_all = not (
        args.calibrate or args.bdd_slices or args.ood or args.thresholds or args.risk_coverage
    )

    model = None
    if run_all or args.calibrate or args.ood or args.risk_coverage:
        from ultralytics import YOLO

        model = YOLO(str(args.weights))

    if run_all or args.calibrate:
        run_calibrate(model, args.weights)
    if run_all or args.bdd_slices:
        run_bdd_slices()
    if run_all or args.ood:
        run_ood(model, args.weights)
    if run_all or args.thresholds:
        run_thresholds()
    if run_all or args.risk_coverage:
        run_risk_coverage(model, args.weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
