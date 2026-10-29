#!/usr/bin/env python3
"""Explain the high in-domain AUC on GRAZPEDWRI-DX.

Checks five explanations: which annotated classes carry the abnormal label, a
metadata-only baseline (gradient-boosted trees on tabular metadata), burned-in
text geometry from the annotation boxes, the share of follow-up examinations, and
exact duplicates by content hash (--hash-check).

Outputs:
    graz_indomain_audit.csv (path set by --out)

Usage:
    python graz_indomain_audit.py --graz-root ~/datasets/GRAZPEDWRI-DX

CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mskdata
import mskmodel as M

TEXT_CLASS = 8
PREREG_TEXT_AUC = 0.60
PREREG_META_AUC_HI = 0.80
PREREG_META_AUC_LO = 0.70


def cv_auc(X, y, groups=None, n_splits=5, seed=0, model="logreg") -> float:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.asarray(X, float)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    clf = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
           if model == "logreg"
           else HistGradientBoostingClassifier(max_iter=300, random_state=seed))
    cv = (StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
          if groups is not None
          else StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    m = ~np.isnan(oof)
    return float(roc_auc_score(y[m], oof[m])) if m.sum() > 10 else float("nan")


def text_geometry(root: Path, stems: List[str]) -> pd.DataFrame:
    """
    Class-8 (text) box geometry per image, from the YOLO label files.

    YOLO format: `cls xc yc w h`, all normalised to [0, 1]. We keep only class 8
    and summarise count, total area, and where the markers sit. Position matters
    as much as presence: a side marker placed consistently differently on
    splinted limbs is a label-correlated cue even when every image has text.
    """
    lab_dirs = [p for p in (root / "yolov5", root / "labels", root)
                if p.exists()]
    files: Dict[str, Path] = {}
    for d in lab_dirs:
        for p in d.rglob("*.txt"):
            files.setdefault(p.stem, p)
    print(f"  found {len(files)} YOLO label files under {root}")
    rows = []
    for s in stems:
        p = files.get(s)
        r = {"filestem": s, "n_text": 0, "text_area": 0.0, "text_xc": np.nan,
             "text_yc": np.nan, "text_wmax": 0.0, "text_hmax": 0.0,
             "n_boxes_total": 0}
        if p is not None:
            try:
                xs, ys, areas = [], [], []
                for line in p.read_text().split("\n"):
                    f = line.split()
                    if len(f) < 5:
                        continue
                    r["n_boxes_total"] += 1
                    if int(float(f[0])) != TEXT_CLASS:
                        continue
                    xc, yc, w, h = (float(f[1]), float(f[2]),
                                    float(f[3]), float(f[4]))
                    r["n_text"] += 1
                    areas.append(w * h)
                    xs.append(xc)
                    ys.append(yc)
                    r["text_wmax"] = max(r["text_wmax"], w)
                    r["text_hmax"] = max(r["text_hmax"], h)
                if areas:
                    r["text_area"] = float(np.sum(areas))
                    r["text_xc"] = float(np.mean(xs))
                    r["text_yc"] = float(np.mean(ys))
            except Exception as e:
                print(f"  !! {p}: {e}")
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--graz-root", default=str(Path.home() / "datasets" /
                                               "GRAZPEDWRI-DX"))
    ap.add_argument("--out", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "graz_indomain_audit.csv"))
    ap.add_argument("--hash-check", action="store_true",
                    help="content-hash every GRAZ image (slow, ~2 min)")
    cfg = ap.parse_args()

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    g = index[index.dataset == "graz"].reset_index(drop=True)
    y_ab = pd.to_numeric(g.label_abnormal, errors="coerce").fillna(0).astype(int).values
    y_fr = pd.to_numeric(g.label_fracture, errors="coerce").fillna(0).astype(int).values
    grp = g.patient_group.values
    rows = []

    print("=" * 78)
    print("1. WHAT ACTUALLY CARRIES label_abnormal?")
    print("=" * 78)
    yc = [c for c in g.columns if c.startswith("yolo_")]
    if yc:
        t = pd.DataFrame({
            "prevalence": [pd.to_numeric(g[c], errors="coerce").fillna(0).mean()
                           for c in yc],
            "share_of_positives": [
                float(((pd.to_numeric(g[c], errors="coerce").fillna(0) > 0)
                       & (y_ab == 1)).sum()) / max(1, y_ab.sum()) for c in yc],
        }, index=[c.replace("yolo_", "") for c in yc])
        print(t.round(4).sort_values("prevalence", ascending=False).to_string())
        print("\n  If one class dominates, label_abnormal IS that class and the")
        print("  'any finding' framing is cosmetic.")

    print("\n" + "=" * 78)
    print("2. METADATA-ONLY BASELINE  (no pixels at all)")
    print("=" * 78)
    cand = ["meta_age", "meta_gender", "meta_laterality", "meta_projection",
            "meta_initial_exam", "has_cast", "meta_pixel_spacing"]
    use = [c for c in cand if c in g.columns]
    print(f"  features: {use}")
    Xm = np.column_stack([pd.to_numeric(g[c], errors="coerce")
                          .fillna(-1).values for c in use])
    for nm, yv in (("label_abnormal", y_ab), ("label_fracture", y_fr)):
        a_g = cv_auc(Xm, yv, grp, model="hgb")
        a_u = cv_auc(Xm, yv, None, model="hgb")
        print(f"  {nm:<16s} grouped={a_g:.4f}  ungrouped={a_u:.4f}")
        rows.append({"analysis": f"metadata_only::{nm}", "auc_grouped": a_g,
                     "auc_ungrouped": a_u, "n": len(yv)})
    meta_auc = rows[-2]["auc_grouped"]
    print(f"\n  image model reaches 0.978. metadata alone reaches "
          f"{meta_auc:.4f}.")
    if meta_auc >= PREREG_META_AUC_HI:
        print("  >>> M1: the cohort is separable WITHOUT pixels. Report this")
        print("      baseline beside every GRAZ number so a reader can see what")
        print("      the image actually adds.")
    elif meta_auc < PREREG_META_AUC_LO:
        print("  >>> M2: the image is doing the work. A pixel-level shortcut is")
        print("      now the leading explanation -- see section 3.")

    print("\n" + "=" * 78)
    print("3. BURNED-IN TEXT GEOMETRY  (class 8 is the MOST common annotation")
    print("   in this dataset: 23,722 text labels vs 18,090 fracture labels)")
    print("=" * 78)
    # Derive the stem from the file path, not from `uid`: uid does not match the
    # label file names, and a failed join silently yields constant features
    # (an AUC of exactly 0.5 on every feature means the join failed).
    stems = [Path(str(x)).stem for x in g.path]
    tg = text_geometry(Path(cfg.graz_root).expanduser(), stems)
    feats = ["n_text", "text_area", "text_xc", "text_yc", "text_wmax", "text_hmax"]
    Xt = tg[feats].fillna(-1).values
    matched = int((tg.n_boxes_total > 0).sum())
    print(f"  label files joined: {matched} / {len(tg)}")
    if matched < 0.5 * len(tg):
        print("\n  !! JOIN FAILED -- fewer than half the images matched a label")
        print("     file. The text test below is MEANINGLESS. Compare a few:")
        print(f"     index stems : {stems[:3]}")
        raise SystemExit("aborting rather than printing a fake null")
    if "yolo_text" in g.columns:
        idx_rate = pd.to_numeric(g.yolo_text, errors="coerce").fillna(0).mean()
        obs_rate = float((tg.n_text > 0).mean())
        print(f"  text present: index says {idx_rate:.4f}, this parse says "
              f"{obs_rate:.4f}")
        if abs(idx_rate - obs_rate) > 0.05:
            raise SystemExit(
                "aborting: this parse disagrees with mskdata's yolo_text flag, "
                "so one of the two is reading the wrong files")
    print(f"  images with >=1 text box: {int((tg.n_text > 0).sum())} / {len(tg)}"
          f"  ({(tg.n_text > 0).mean():.1%})")
    print(f"  mean text boxes per image: {tg.n_text.mean():.2f}")
    for nm, yv in (("label_abnormal", y_ab), ("label_fracture", y_fr)):
        a_g = cv_auc(Xt, yv, grp, model="hgb")
        print(f"  text geometry -> {nm:<16s} grouped AUC = {a_g:.4f}")
        rows.append({"analysis": f"text_geometry::{nm}", "auc_grouped": a_g,
                     "n": len(yv)})
    ta = rows[-2]["auc_grouped"]
    # which single feature does the damage
    for f in feats:
        a = cv_auc(tg[[f]].fillna(-1).values, y_ab, grp, model="hgb")
        print(f"    {f:<12s} alone: {a:.4f}")
    if ta >= PREREG_TEXT_AUC:
        print(f"\n  >>> T1: BURNED-IN TEXT PREDICTS THE LABEL AT {ta:.3f}.")
        print("      Every GRAZ result in this thesis must be re-run with the")
        print("      class-8 boxes masked to background. This is a finding about")
        print("      GRAZPEDWRI-DX itself and belongs in the paper.")
    else:
        print(f"\n  >>> T2: text geometry gives {ta:.3f}, below {PREREG_TEXT_AUC}."
              " Do not mask.")

    print("\n" + "=" * 78)
    print("4. INITIAL EXAM vs FOLLOW-UP")
    print("=" * 78)
    if "meta_initial_exam" in g.columns:
        ie = pd.to_numeric(g.meta_initial_exam, errors="coerce")
        print(pd.crosstab(ie.fillna(-1), y_ab, normalize="index").round(4).to_string())
        print(f"\n  counts: {ie.value_counts(dropna=False).to_dict()}")
        for v in sorted(ie.dropna().unique()):
            m = (ie == v).values
            print(f"  initial_exam={v}: n={int(m.sum())}, "
                  f"prevalence={y_ab[m].mean():.4f}")
        print("\n  Follow-ups carry healing callus, casts and hardware. They")
        print("  are easy positives, in metadata AND in pixels.")
        miss = ie.isna()
        if miss.any() and miss.mean() > 0.05:
            print(f"\n  *** {int(miss.sum())} of {len(ie)} rows ({miss.mean():.1%}) "
                  "have initial_exam MISSING,")
            print(f"      and that group is {y_ab[miss.values].mean():.4f} "
                  "positive against "
                  f"{y_ab[(~miss).values].mean():.4f} for initial_exam=1.")
            # AUC of a single binary predictor = 0.5*(1 + TPR - FPR).
            # `0.5 + 0.5*|lift|` is only correct when the two groups are the
            # same size, which they are not; it printed 0.7609 against a true
            # 0.8014.
            mv = miss.values
            tp = float((mv & (y_ab == 1)).sum())
            fp = float((mv & (y_ab == 0)).sum())
            tpr = tp / max(1.0, float((y_ab == 1).sum()))
            fpr = fp / max(1.0, float((y_ab == 0).sum()))
            print(f"      A SINGLE is-missing indicator scores AUC "
                  f"{0.5 * (1 + tpr - fpr):.4f} on its own "
                  f"(TPR {tpr:.4f}, FPR {fpr:.4f}).")
            print("      Check the raw dataset.csv: if the column is 1 for")
            print("      initial and BLANK for follow-up, then missingness IS")
            print("      follow-up status and roughly half this dataset is")
            print("      repeat imaging of an already-diagnosed fracture.")
        init = (ie == 1).values
        if init.sum() > 1000:
            print(f"\n  INITIAL-EXAM-ONLY SUBSET: n={int(init.sum())}, "
                  f"prevalence={y_ab[init].mean():.4f}")
            print("  MURA XR_WRIST train prevalence is 0.409. This subset is the")
            print("  prevalence-matched, diagnostic-imaging-matched comparison")
            print("  set; the full GRAZ pool at 0.686 is not.")
            am = cv_auc(Xm[init], y_ab[init], grp[init], model="hgb")
            print(f"  metadata-only AUC on this subset: {am:.4f} "
                  f"(full pool {meta_auc:.4f})")
            rows.append({"analysis": "metadata_only::initial_exam_only",
                         "auc_grouped": am, "n": int(init.sum())})
    else:
        print("  meta_initial_exam not in the index")

    if cfg.hash_check:
        print("\n" + "=" * 78)
        print("5. EXACT DUPLICATE SCAN")
        print("=" * 78)
        h = {}
        dup = 0
        for p in g.path.astype(str):
            try:
                d = hashlib.md5(Path(p).read_bytes()).hexdigest()
            except Exception:
                continue
            if d in h:
                dup += 1
            else:
                h[d] = p
        print(f"  {dup} exact duplicate files among {len(g)} images")
        print("  (patient grouping blocks the usual route; this catches "
              "identical files under different stems)")

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
