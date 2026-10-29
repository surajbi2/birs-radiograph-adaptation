#!/usr/bin/env python3
"""Count source-training patients that appear in each MURA target split.

Patients shared with the source training set are removed from every target
training pool; this script also counts any that appear in a target evaluation
split, so the zero-shot scores can be checked for leakage.

Usage:
    python check_patient_overlap.py --mura-root <MURA-v1.1> --out <csv>

CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REGIONS = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND",
           "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]
SOURCES = ["XR_WRIST", "XR_HAND", "XR_HUMERUS"]
PATIENT_RE = re.compile(r"patient(\d+)")


def patient_of(path: str) -> str | None:
    """MURA paths look like .../XR_WRIST/patient11186/study1_positive/image1.png"""
    m = PATIENT_RE.search(str(path))
    return m.group(1) if m else None


def load_split(root: Path, split: str) -> pd.DataFrame:
    """Read the official MURA image list for one split."""
    for name in (f"{split}_image_paths.csv", f"MURA-v1.1/{split}_image_paths.csv"):
        p = root / name
        if p.exists():
            break
    else:
        hits = sorted(root.rglob(f"{split}_image_paths.csv"))
        if len(hits) != 1:
            raise SystemExit(
                f"!! could not locate a unique {split}_image_paths.csv under {root}\n"
                f"   found: {[str(h) for h in hits]}")
        p = hits[0]
    df = pd.read_csv(p, header=None, names=["path"])
    df["region"] = df.path.str.extract(r"(XR_[A-Z]+)", expand=False)
    df["patient"] = df.path.map(patient_of)
    missing = df.patient.isna().sum() + df.region.isna().sum()
    if missing:
        raise SystemExit(f"!! {missing} rows in {p} had no parseable region or "
                         f"patient id; refusing to report a partial count")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mura-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    cfg = ap.parse_args()

    train = load_split(cfg.mura_root, "train")
    valid = load_split(cfg.mura_root, "valid")
    print(f"train {len(train)} images, {train.patient.nunique()} patients")
    print(f"valid {len(valid)} images, {valid.patient.nunique()} patients")

    # Global check first. If the official split is patient disjoint overall,
    # every per-region number below is zero by construction and one sentence
    # closes the issue.
    g_overlap = set(train.patient) & set(valid.patient)
    print(f"\nGLOBAL train/valid patient overlap: {len(g_overlap)}")

    rows = []
    for src in SOURCES:
        src_pat = set(train.loc[train.region == src, "patient"])
        for tgt in REGIONS:
            if tgt == src:
                continue
            tv = valid[valid.region == tgt]
            tv_pat = set(tv.patient)
            shared = src_pat & tv_pat
            n_img = int(tv.patient.isin(shared).sum())
            rows.append({
                "source": src, "target": tgt,
                "source_train_patients": len(src_pat),
                "target_valid_patients": len(tv_pat),
                "target_valid_images": int(len(tv)),
                "shared_patients": len(shared),
                "affected_valid_images": n_img,
                "affected_valid_image_frac": (n_img / len(tv)) if len(tv) else float("nan"),
            })

    out = pd.DataFrame(rows)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    prov = {"provenance": {
        "script": "check_patient_overlap.py v1",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(), "pandas": pd.__version__,
        "mura_root": str(cfg.mura_root),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "source-TRAIN vs target-VALIDATION patient overlap; the paper "
                "already removes source patients from target TRAINING pools"}}
    with cfg.out.open("w") as fh:
        fh.write("# " + json.dumps(prov) + "\n")
        out.to_csv(fh, index=False)

    print("\n" + out.to_string(index=False))
    total = int(out.shared_patients.sum())
    print(f"\nTOTAL shared patients across all source-target pairs: {total}")
    if total == 0:
        print("\nSuggested sentence for the paper:")
        print("  'No patient in any source training set appears in any target")
        print("   validation set, so the zero shot results carry no validation")
        print("   leakage (0 shared patients across all source and target pairs).'")
    else:
        print("\nOverlap found. Report the counts per target and state the")
        print("affected image fraction. Do NOT describe the zero shot numbers")
        print("as leakage free.")
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    sys.exit(main())
