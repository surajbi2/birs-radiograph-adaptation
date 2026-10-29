#!/usr/bin/env python3
"""Write every cohort count used in the paper to one CSV.

Counts images, studies and prevalence for the source and target pools, the
FracAtlas exclusions, and the per-target pools before and after removing
patients shared with the source training set. Calls mskdata, mskmodel and
fracatlas_transfer without modifying them.

Output:
    cohort_counts.csv (path set by --out)

Usage:
    python cohort_counts.py --source XR_WRIST

CPU only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskdata
import mskmodel as M
from train_source_models import CsvAppender


def summarise(frame: pd.DataFrame, cohort_id: str, dataset: str, anatomy: str,
              split: str, stage: str, note: str = "",
              n_excluded: Optional[int] = None,
              excluded: Optional[pd.DataFrame] = None) -> Dict:
    """One cohort row. Study-level prevalence uses max-label aggregation."""
    row = {
        "cohort_id": cohort_id, "dataset": dataset, "anatomy": anatomy,
        "split": split, "stage": stage,
        "n_images": int(len(frame)),
        "n_studies": int(frame["study"].nunique()) if "study" in frame else -1,
        "n_patients": (int(frame["patient_group"].nunique())
                       if "patient_group" in frame else -1),
        "prevalence_image": (float(pd.to_numeric(frame["label"],
                                                 errors="coerce").mean())
                             if "label" in frame and len(frame) else np.nan),
        "prevalence_study": np.nan,
        "n_excluded": (int(n_excluded) if n_excluded is not None else 0),
        "excluded_label_0": 0, "excluded_label_1": 0,
        "note": note,
    }
    if "label" in frame and "study" in frame and len(frame):
        s = frame.groupby("study")["label"].max()
        row["prevalence_study"] = float(pd.to_numeric(s, errors="coerce").mean())
    if excluded is not None and len(excluded) and "label" in excluded:
        v = pd.to_numeric(excluded["label"], errors="coerce")
        row["excluded_label_0"] = int((v == 0).sum())
        row["excluded_label_1"] = int((v == 1).sum())
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--out", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "cohort_counts.csv"))
    ap.add_argument("--source", default="XR_WRIST",
                    help="source anatomy whose patients are removed from targets")
    ap.add_argument("--extra-sources", nargs="*", default=["XR_HAND", "XR_HUMERUS"],
                    help="additional sources used in the budget analysis")
    ap.add_argument("--skip-fracatlas", action="store_true",
                    help="FracAtlas decode verification opens every file; slow")
    cfg = ap.parse_args()

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    prov = M.provenance({
        "script": "cohort_counts.py v1",
        "purpose": "cohort counts for the Methods section; no file is modified",
        "source": cfg.source, "extra_sources": list(cfg.extra_sources),
        "index_rows": int(len(index)),
    })
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    res = CsvAppender(Path(cfg.out), prov)
    rows: List[Dict] = []

    # ---------------------------------------------------- MURA base pools --
    print("MURA, per anatomy, as indexed")
    for a in mskdata.MURA_ANATOMIES:
        for split in ("train", "valid"):
            f = M.frame_for(index, "mura", a, split).reset_index(drop=True)
            r = summarise(f, f"mura_{a}_{split}", "mura", a, split, "raw")
            rows.append(r)
            print(f"  {a:<12s} {split:<6s} {r['n_images']:>6d} img / "
                  f"{r['n_studies']:>5d} std / {r['n_patients']:>5d} pt   "
                  f"prev_img {r['prevalence_image']:.4f}  "
                  f"prev_std {r['prevalence_study']:.4f}")

    # --------------------------------------------- source training pools ---
    print("\nSource training pools (Methods 2.5)")
    for s in [cfg.source] + [x for x in cfg.extra_sources if x != cfg.source]:
        f = M.frame_for(index, "mura", s, "train").reset_index(drop=True)
        r = summarise(f, f"source_{s}", "mura", s, "train", "raw",
                      note="pool a source model was trained on")
        rows.append(r)
        print(f"  {s:<12s} {r['n_images']:>6d} img / {r['n_studies']:>5d} std / "
              f"{r['n_patients']:>5d} pt   prev_img {r['prevalence_image']:.4f}  "
              f"prev_std {r['prevalence_study']:.4f}")

    # --------------------------------- leak control, before and after ------
    print(f"\nSource-patient removal, source = {cfg.source} (Methods 2.2)")
    print(f"  {'target':<12}{'before':>8}{'after':>8}{'dropped':>9}{'pct':>8}")
    for t in mskdata.MURA_ANATOMIES:
        if t == cfg.source:
            continue
        before = M.frame_for(index, "mura", t, "train").reset_index(drop=True)
        after = mskdata.drop_source_patients(index, before, cfg.source, "train")
        after = after.reset_index(drop=True)
        dropped = len(before) - len(after)
        pct = 100.0 * dropped / max(1, len(before))
        excl = before[~before["uid"].isin(set(after["uid"]))]
        rows.append(summarise(before, f"target_{t}_train_preleak", "mura", t,
                              "train", "raw",
                              note="before source-patient removal"))
        rows.append(summarise(after, f"target_{t}_train_postleak", "mura", t,
                              "train", "leak_removed",
                              note=f"after removing patients shared with "
                                   f"{cfg.source}/train; {pct:.2f}% of images",
                              n_excluded=dropped, excluded=excl))
        print(f"  {t:<12}{len(before):>8d}{len(after):>8d}{dropped:>9d}{pct:>7.2f}%")

    # ------------------------------------------------------------ GRAZ ----
    print("\nGRAZPEDWRI-DX (Methods 2.2)")
    ie_col = "meta_initial_exam"
    for split in ("train", "valid", "test"):
        f = M.frame_for(index, "graz", "XR_WRIST", split,
                        "label_abnormal").reset_index(drop=True)
        if f.empty:
            continue
        rows.append(summarise(f, f"graz_{split}_full", "graz", "XR_WRIST", split,
                              "raw", note="full pool, initial + follow-up"))
        if ie_col in f.columns:
            v = pd.to_numeric(f[ie_col], errors="coerce")
            init = f[(v == 1).to_numpy()].reset_index(drop=True)
            foll = f[(v != 1).to_numpy()].reset_index(drop=True)
            rows.append(summarise(init, f"graz_{split}_initial", "graz",
                                  "XR_WRIST", split, "initial_only",
                                  note="initial_exam == 1; the analysed cohort",
                                  n_excluded=len(foll), excluded=foll))
            rows.append(summarise(foll, f"graz_{split}_followup", "graz",
                                  "XR_WRIST", split, "followup_only",
                                  note="initial_exam missing; excluded from analysis"))
            print(f"  {split:<6s} full {len(f):>6d}  initial {len(init):>6d} "
                  f"(prev {init.label.mean():.4f})  follow-up {len(foll):>6d} "
                  f"(prev {foll.label.mean():.4f})  "
                  f"follow-up share {len(foll)/max(1,len(f)):.4f}")

    # ------------------------------------------------------- FracAtlas ----
    if not cfg.skip_fracatlas:
        print("\nFracAtlas (Methods 2.2) — decode verification opens every file")
        from fracatlas_transfer import verify_readable, FRAC_TARGETS
        for t in FRAC_TARGETS:
            raw = M.frame_for(index, "fracatlas", t, "test",
                              "label_fracture").reset_index(drop=True)
            if raw.empty:
                print(f"  {t:<12s} absent from the index")
                continue
            ok = verify_readable(raw).reset_index(drop=True)
            bad = raw[~raw["uid"].isin(set(ok["uid"]))]
            rows.append(summarise(raw, f"frac_{t}_raw", "fracatlas", t, "test",
                                  "raw", note="as indexed"))
            rows.append(summarise(
                ok, f"frac_{t}_readable", "fracatlas", t, "test", "readable",
                note="after excluding images that do not decode; "
                     "LOAD_TRUNCATED_IMAGES is never set",
                n_excluded=len(bad), excluded=bad))
            n0 = int((pd.to_numeric(bad["label"], errors="coerce") == 0).sum())
            print(f"  {t:<12s} raw {len(raw):>5d} -> readable {len(ok):>5d}  "
                  f"excluded {len(bad):>3d} (label 0: {n0}, "
                  f"label 1: {len(bad)-n0})  prev {ok.label.mean():.4f}")
    else:
        print("\n[skip] FracAtlas — --skip-fracatlas set")

    res.write(rows)
    print(f"\nwrote {len(rows)} cohort rows to {cfg.out}")
    print("\nCross-check these against the n_images / n_train_images columns "
          "already present\nin the result CSVs. They must agree exactly; if they "
          "do not, the experiments\nand this script are not reading the same "
          "index and nothing downstream is safe.")


if __name__ == "__main__":
    main()
