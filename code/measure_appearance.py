#!/usr/bin/env python3
"""Measure low-level image appearance and test whether it predicts anatomy.

Computes five descriptors per image (padding fraction, content fraction,
aspect ratio and grey-level statistics after the standard preprocessing) and
checks how well they separate anatomies. MURA images carry variable black borders, which change the percentile
window and therefore the contrast applied to the anatomy, and resizing to a
square distorts elongated anatomies; both are cues a model could use instead of
pathology.

Outputs:
    appearance_audit.csv (path set by --out)

Usage:
    python measure_appearance.py --n-per-group 300

CPU only. Also imported by other stages for ``measure_one``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mskdata
import mskmodel as M


def measure_one(path: str) -> dict:
    with Image.open(path) as im:
        raw = np.asarray(im)

    # cross-channel check: mskdata line ~220 ASSUMES channels are duplicates
    chan_delta = 0
    if raw.ndim == 3 and raw.shape[2] >= 3:
        a = raw[..., :3].astype(np.int32)
        chan_delta = int(max(np.abs(a[..., 0] - a[..., 1]).max(),
                             np.abs(a[..., 1] - a[..., 2]).max()))
        gray_raw = raw[..., :3].mean(axis=2)
    else:
        gray_raw = raw.astype(np.float32)

    h, w = gray_raw.shape
    floor = gray_raw.min()
    processed = mskdata.load_gray_uint8(path)

    return {
        "h": h, "w": w,
        "aspect": max(h, w) / max(1, min(h, w)),
        "is_portrait": int(h > w),
        # fraction of pixels sitting at the darkest value = letterbox padding
        "pad_frac": float((gray_raw <= floor + 1e-6).mean()),
        # fraction of the processed image carrying any signal
        "content_frac": float((processed > 10).mean()),
        "proc_mean": float(processed.mean()),
        "proc_std": float(processed.std()),
        "chan_delta": chan_delta,
    }


FEATURES = ["aspect", "pad_frac", "content_frac", "proc_mean", "proc_std"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--n-per-group", type=int, default=300)
    ap.add_argument("--out", default=str(Path(M.DEFAULT_RESULTS) / "appearance_audit.csv"))
    a = ap.parse_args()

    idx = pd.read_csv(a.index, comment="#")
    rows = []
    for (ds, ana), g in idx.groupby(["dataset", "anatomy"]):
        sel = g.sample(min(a.n_per_group, len(g)), random_state=0)
        for p in sel.path:
            try:
                r = measure_one(p)
            except Exception as e:
                print(f"  !! {p}: {e}")
                continue
            r.update({"dataset": ds, "anatomy": ana, "path": p})
            rows.append(r)
    d = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)

    # ---- 1. the channel-duplicate assumption in mskdata line ~220 ---------
    print("\n" + "=" * 78)
    print("1. CHANNEL-DUPLICATE ASSUMPTION  (mskdata: 'channels are duplicates')")
    print("=" * 78)
    worst = int(d.chan_delta.max())
    n_bad = int((d.chan_delta > 0).sum())
    print(f"  max cross-channel delta : {worst}")
    print(f"  images with any colour  : {n_bad} / {len(d)}")
    if worst > 0:
        print("  >>> The comment is WRONG. Some images carry colour content "
              "(overlays, burned-in markers).")
        print("      .mean(axis=2) is silently blending it. Find out what it is "
              "before it becomes a shortcut.")
        print(d[d.chan_delta > 0].groupby(["dataset", "anatomy"])
                .size().to_string())
    else:
        print("  Verified: channels are genuinely duplicates. Comment is safe.")

    # ---- 2. is geometry anatomy-correlated? ------------------------------
    print("\n" + "=" * 78)
    print("2. GEOMETRY BY ANATOMY")
    print("=" * 78)
    tab = d.groupby(["dataset", "anatomy"])[FEATURES].agg(["mean", "std"]).round(3)
    print(tab.to_string())

    mura = d[d.dataset == "mura"]
    if len(mura) and mura.anatomy.nunique() > 1:
        print("\n  between- vs within-anatomy spread (MURA only)")
        print("  ratio > 1 means the feature separates anatomies more than it "
              "varies inside one")
        for f in FEATURES:
            between = mura.groupby("anatomy")[f].mean().std()
            within = mura.groupby("anatomy")[f].std().mean()
            ratio = between / within if within > 0 else float("inf")
            flag = "  <-- ANATOMY-PREDICTIVE" if ratio > 1.0 else ""
            print(f"    {f:<14s} between={between:.4f}  within={within:.4f}  "
                  f"ratio={ratio:.2f}{flag}")

    # ---- 3. can geometry ALONE predict anatomy? --------------------------
    print("\n" + "=" * 78)
    print("3. CAN GEOMETRY ALONE PREDICT ANATOMY?  (no pixels of pathology used)")
    print("=" * 78)
    if len(mura) and mura.anatomy.nunique() > 1:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = mura[FEATURES].to_numpy()
        y = mura.anatomy.to_numpy()
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000))
        acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
        chance = mura.anatomy.value_counts(normalize=True).max()
        print(f"  5-fold accuracy from geometry only : {acc:.3f}")
        print(f"  majority-class chance              : {chance:.3f}")
        if acc > chance + 0.15:
            print("  >>> Geometry alone identifies the anatomy. This is a live "
                  "H2 shortcut candidate.")
            print("      Add a content-cropped + aspect-preserving arm to the "
                  "layerwise experiment,")
            print("      or the bottom-up result will be uninterpretable.")
        else:
            print("  Geometry is weakly anatomy-predictive. Note in limitations "
                  "and proceed unchanged.")

    # ---- 4. MURA vs GRAZ: how much of 'site shift' is just framing? -------
    print("\n" + "=" * 78)
    print("4. MURA-WRIST vs GRAZ  (preview of the Stage B site control)")
    print("=" * 78)
    mw = d[(d.dataset == "mura") & (d.anatomy == "XR_WRIST")]
    gz = d[d.dataset == "graz"]
    if len(mw) and len(gz):
        cmp = pd.DataFrame({"MURA_XR_WRIST": mw[FEATURES].mean(),
                            "GRAZ": gz[FEATURES].mean()}).round(3)
        cmp["abs_diff"] = (cmp.MURA_XR_WRIST - cmp.GRAZ).abs().round(3)
        print(cmp.to_string())
        print("\n  Any large abs_diff here is appearance shift you will measure "
              "as 'site shift'\n  in Stage B unless you control for it.")

    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
