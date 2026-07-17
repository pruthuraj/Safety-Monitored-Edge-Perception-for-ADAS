"""audit_final_package: final release gate (Week 10, EXP-013).

Verifies the shipped package is internally consistent:
  1. all required deliverable files exist;
  2. markdown links in README point to existing local files;
  3. no unqualified forbidden safety claims in public docs;
  4. headline metrics in README/paper/CV match results/report_summary.json;
  5. demo/final_demo.mp4 opens and is 60-90 s.

Exit code 0 = ship-ready. Non-zero prints every failure.

Usage:
    python scripts/audit_final_package.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "PLAN.md",
    "paper/main.md",
    "paper/main.pdf",
    "paper/tables.md",
    "demo/final_demo.mp4",
    "demo/monitor_overlay.mp4",
    "demo/monitor_overlay.gif",
    "demo/demo_script.md",
    "docs/final_reproduction.md",
    "docs/cv_material.md",
    "docs/experiment_log.md",
    "results/report_summary.json",
    "safety/safety_case.md",
    "safety/gsn.svg",
    "safety/sotif_argument.md",
    "safety/iso_pas_8800_mapping.md",
    "safety/verification_report.md",
    "safety/evidence_index.csv",
    "safety/traceability_matrix.csv",
    "safety/requirements.csv",
]

FORBIDDEN = re.compile(r"(certified|iso-compliant|proven safe|production-safe)", re.IGNORECASE)
# lines that legitimately discuss the forbidden terms while disclaiming them
ALLOW = re.compile(
    r"(not\s+\"?certif|no\s+claim|not\s+claim|does not|not\s+ISO|not\s+proven|"
    r"\*\*not\*\*|least certifiable|not compliance|do-not-say|not \"production|forbidden)",
    re.IGNORECASE,
)


def check_files() -> list[str]:
    return [f"missing required file: {p}" for p in REQUIRED if not (REPO / p).exists()]


def check_readme_links() -> list[str]:
    errs = []
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for m in re.finditer(r"\]\(([^)]+)\)", readme):
        target = m.group(1)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#")[0]
        if path and not (REPO / path).exists():
            errs.append(f"README link target missing: {target}")
    return errs


def check_forbidden() -> list[str]:
    errs = []
    for rel in ("README.md", "paper/main.md", "docs/cv_material.md", "demo/demo_script.md",
                "safety/safety_case.md", "safety/sotif_argument.md", "safety/iso_pas_8800_mapping.md"):
        for n, line in enumerate((REPO / rel).read_text(encoding="utf-8").split("\n"), 1):
            if FORBIDDEN.search(line) and not ALLOW.search(line):
                errs.append(f"{rel}:{n} possible unqualified forbidden claim: {line.strip()[:80]}")
    return errs


def check_metrics() -> list[str]:
    errs = []
    s = json.loads((REPO / "results" / "report_summary.json").read_text())
    text = (
        (REPO / "README.md").read_text(encoding="utf-8")
        + (REPO / "paper" / "main.md").read_text(encoding="utf-8")
        + (REPO / "docs" / "cv_material.md").read_text(encoding="utf-8")
    )
    expect = {
        "trt mAP50 0.856": (s["baseline"]["trt_fp16_mAP50"] == 0.8564, "0.856"),
        "pytorch mAP50 0.8588": (s["baseline"]["pytorch_mAP50"] == 0.8588, "0.8588"),
        "night AUROC 0.982": (round(s["ood_max_conf"]["bdd-night"]["auroc"], 3) == 0.982, "0.982"),
        "latency 17.2": (s["latency_full_loop"]["tensorrt_fp16"]["p95"] == 17.2, "17.2"),
        "fog mAP 0.071": (s["fault_injection"]["fog_high"]["mAP50"] == 0.0711, "0.071"),
        "low-light mAP 0.689": (s["fault_injection"]["low_light_high"]["mAP50"] == 0.6887, "0.689"),
    }
    for name, (matches_summary, token) in expect.items():
        if not matches_summary:
            errs.append(f"summary value drifted: {name}")
        if token not in text:
            errs.append(f"metric absent from public docs: {name} ({token})")
    return errs


def check_demo() -> list[str]:
    import cv2

    cap = cv2.VideoCapture(str(REPO / "demo" / "final_demo.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if n == 0 or fps == 0:
        return ["final_demo.mp4 unreadable"]
    dur = n / fps
    if not 60 <= dur <= 90:
        return [f"final_demo.mp4 duration {dur:.1f}s outside 60-90s"]
    return []


def main() -> int:
    errors = []
    errors += check_files()
    errors += check_readme_links()
    errors += check_forbidden()
    errors += check_metrics()
    errors += check_demo()

    if errors:
        print(f"AUDIT FAILED ({len(errors)} issue(s)):")
        for e in errors:
            print("  -", e)
        return 1
    print("AUDIT PASSED: files present, links resolve, no forbidden claims, "
          "metrics match report_summary.json, demo 60-90s. Ship-ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
