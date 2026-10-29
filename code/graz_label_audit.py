#!/usr/bin/env python3
"""Do appearance descriptors predict the GRAZPEDWRI-DX label, and is plaster
cast the mechanism?

Compares the two GRAZPEDWRI-DX label definitions (any pathological finding
versus visible fracture), measures how much of each the five appearance
descriptors explain, and checks the role of casts and metal.

Outputs (under --out):
    graz_label_audit.csv, graz_mura_wrist_descriptors.csv (feature cache)

Usage:
    python graz_label_audit.py --workers 16

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

import mskmodel as M
from measure_appearance import measure_one, FEATURES

PREREG_LABEL_AUC = 0.60
PREREG_CAST_AUC = 0.75
PREREG_STRATA_DROP = 0.03


def _one(path: str) -> Dict:
    try:
        r = measure_one(path)
    except Exception as e:
        r = {k: np.nan for k in FEATURES}
        r["error"] = str(e)
    r["path"] = path
    return r


def descriptors(paths: List[str], workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = [_one(p) for p in paths]
    else:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            rows = list(pool.imap(_one, paths, chunksize=64))
    d = pd.DataFrame(rows)
    bad = int(d[FEATURES].isna().any(axis=1).sum())
    if bad:
        print(f"  !! {bad} images produced unusable descriptors; dropped")
    return d


def cv_auc(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
           n_splits: int = 5, seed: int = 0) -> float:
    """Cross-validated AUC. Patient-grouped when `groups` is supplied."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2 or len(y) < 5 * n_splits:
        return float("nan")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000))
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
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], oof[m]))


def line(name: str, X, y, g, n_extra: str = "") -> Dict:
    a_g = cv_auc(X, y, g)
    a_u = cv_auc(X, y, None)
    print(f"  {name:<34s} n={len(y):>6d}  prev={y.mean():.3f}  "
          f"grouped={a_g:.3f}  ungrouped={a_u:.3f}  diff={a_g-a_u:+.3f} {n_extra}")
    return {"analysis": name, "n": len(y), "prevalence": float(y.mean()),
            "auc_grouped": a_g, "auc_ungrouped": a_u, "diff": a_g - a_u}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--out", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "graz_label_audit.csv"))
    ap.add_argument("--feature-cache",
                    default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                "graz_mura_wrist_descriptors.csv"))
    ap.add_argument("--workers", type=int, default=8)
    cfg = ap.parse_args()
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    g = index[index.dataset == "graz"].reset_index(drop=True)
    mw = index[(index.dataset == "mura")
               & (index.anatomy == "XR_WRIST")].reset_index(drop=True)
    if g.empty:
        raise SystemExit("no GRAZ rows in the index")

    cache = Path(cfg.feature_cache)
    if cache.exists():
        print(f"[features] cache hit -> {cache}")
        feats = pd.read_csv(cache)
    else:
        print(f"[features] FULL coverage: {len(g)} GRAZ + {len(mw)} MURA-wrist "
              f"images, {cfg.workers} workers", flush=True)
        allp = g.path.astype(str).tolist() + mw.path.astype(str).tolist()
        feats = descriptors(allp, cfg.workers)
        feats.to_csv(cache, index=False)
        print(f"[features] cached -> {cache}")

    fmap = feats.set_index("path")[FEATURES]
    gg = g.join(fmap, on="path").dropna(subset=FEATURES).reset_index(drop=True)
    mm = mw.join(fmap, on="path").dropna(subset=FEATURES).reset_index(drop=True)
    X = gg[FEATURES].to_numpy(float)
    grp = gg.patient_group.to_numpy()
    rows: List[Dict] = []

    print("\n" + "=" * 78)
    print("1. DESCRIPTORS -> GRAZ LABEL   (patient-grouped is PRIMARY)")
    print("=" * 78)
    for lc in ("label_abnormal", "label_fracture"):
        y = pd.to_numeric(gg[lc], errors="coerce").fillna(0).astype(int).to_numpy()
        rows.append(line(lc, X, y, grp))

    print("\n" + "=" * 78)
    print("2. IS CAST APPEARANCE-DETECTABLE?   (clause b of P1')")
    print("=" * 78)
    cast = pd.to_numeric(gg.get("has_cast", 0), errors="coerce").fillna(0)
    cast = (cast > 0).astype(int).to_numpy()
    rows.append(line("has_cast", X, cast, grp))
    metal = pd.to_numeric(gg.get("has_metal", 0), errors="coerce").fillna(0)
    metal = (metal > 0).astype(int).to_numpy()
    rows.append(line("has_metal", X, metal, grp))

    print("\n" + "=" * 78)
    print("3. IS CAST LABEL-CORRELATED?   (clause c of P1')")
    print("=" * 78)
    lab = pd.to_numeric(gg.label_abnormal, errors="coerce").fillna(0).astype(int)
    lab = lab.to_numpy()
    ct = pd.crosstab(pd.Series(cast, name="has_cast"),
                     pd.Series(lab, name="label_abnormal"), normalize="index")
    print(ct.round(4).to_string())
    print(f"\n  P(label | cast=1) = {lab[cast == 1].mean():.4f}   "
          f"P(label | cast=0) = {lab[cast == 0].mean():.4f}   "
          f"lift = {lab[cast == 1].mean() - lab[cast == 0].mean():+.4f}")
    print(f"  cast ALONE as a classifier: AUC = "
          f"{0.5 + 0.5*(lab[cast==1].mean() - lab[cast==0].mean()):.4f} "
          f"(a single binary feature)")

    print("\n" + "=" * 78)
    print("4. DOES THE SIGNAL SURVIVE WITHIN CAST STRATA?   (clause d -- "
          "THE LOAD-BEARING ONE)")
    print("=" * 78)
    pooled = [r for r in rows if r["analysis"] == "label_abnormal"][0]["auc_grouped"]
    strata, weights = [], []
    for v in (0, 1):
        m = cast == v
        r = line(f"label_abnormal | has_cast={v}", X[m], lab[m], grp[m])
        r["stratum"] = v
        rows.append(r)
        if np.isfinite(r["auc_grouped"]):
            strata.append(r["auc_grouped"])
            weights.append(int(m.sum()))
    within = (float(np.average(strata, weights=weights)) if strata
              else float("nan"))
    print(f"\n  pooled                 = {pooled:.4f}")
    print(f"  prevalence-weighted within-cast = {within:.4f}")
    print(f"  drop                   = {pooled - within:+.4f}"
          f"   (pre-registered threshold {PREREG_STRATA_DROP})")

    cast_auc = [r for r in rows if r["analysis"] == "has_cast"][0]["auc_grouped"]
    a = pooled >= PREREG_LABEL_AUC
    b = cast_auc >= PREREG_CAST_AUC
    d = (pooled - within) >= PREREG_STRATA_DROP
    print("\n  P1' clauses:")
    print(f"    (a) label AUC >= {PREREG_LABEL_AUC}      {pooled:.3f}   "
          f"{'YES' if a else 'NO'}")
    print(f"    (b) cast AUC  >= {PREREG_CAST_AUC}      {cast_auc:.3f}   "
          f"{'YES' if b else 'NO'}")
    print(f"    (d) strata drop >= {PREREG_STRATA_DROP}  "
          f"{pooled-within:+.3f}   {'YES' if d else 'NO'}")
    print(f"\n  >>> P1' {'HOLDS' if (a and b and d) else 'FAILS'}. "
          "Report either way; a failure means the appearance shortcut is\n"
          "      dataset-specific and does not transfer to GRAZ as claimed.")

    print("\n" + "=" * 78)
    print("4b. AGE x CAST JOINT STRATA")
    print("=" * 78)
    print("  Added after the first run. The descriptors predict")
    print("  `is_below_median_age` at 0.681 -- HIGHER than they predict the")
    print("  label (0.614) and higher than cast contributes. Age is strongly")
    print("  appearance-encoded (a 3-year-old wrist is not a 17-year-old wrist)")
    print("  and age is label-correlated. Stratifying on cast alone therefore")
    print("  leaves a second appearance-mediated path open, and clause (d) of")
    print("  P1' would fire on an age effect just as readily as a cast effect.\n")
    if "meta_age" in gg.columns:
        agev = pd.to_numeric(gg.meta_age, errors="coerce")
        okv = agev.notna().to_numpy()
        band = pd.cut(agev, [-1, 8, 12, 15, 100],
                      labels=["<=8", "8-12", "12-15", ">15"])
        jstr, jw = [], []
        for b in band.cat.categories:
            for v in (0, 1):
                m = okv & (band == b).to_numpy() & (cast == v)
                if m.sum() < 200 or len(np.unique(lab[m])) < 2:
                    print(f"  age {b:<6s} cast={v}  n={int(m.sum()):>6d}  "
                          f"skipped (too few or single-class)")
                    continue
                r = line(f"label | age {b}, cast={v}", X[m], lab[m], grp[m])
                rows.append(r)
                if np.isfinite(r["auc_grouped"]):
                    jstr.append(r["auc_grouped"])
                    jw.append(int(m.sum()))
        if jstr:
            jw_auc = float(np.average(jstr, weights=jw))
            print(f"\n  pooled                          = {pooled:.4f}")
            print(f"  within cast only                = {within:.4f}"
                  f"   (drop {pooled-within:+.4f})")
            print(f"  within age x cast               = {jw_auc:.4f}"
                  f"   (drop {pooled-jw_auc:+.4f})")
            extra = within - jw_auc
            print(f"\n  additional drop from adding AGE = {extra:+.4f}")
            if extra >= 0.02:
                print("  >>> A SECOND appearance-mediated path runs through AGE.")
                print("      Clause (d) of P1' is then NOT specific to cast, and")
                print("      the write-up must say 'cast AND age', not 'cast'.")
            else:
                print("  >>> Age adds little beyond cast. Clause (d) is")
                print("      cast-specific and P1' can be stated as written.")

    print("\n" + "=" * 78)
    print("5. AGE -- the confound that is NOT a nuisance variable")
    print("=" * 78)
    if "meta_age" in gg.columns:
        age = pd.to_numeric(gg.meta_age, errors="coerce")
        ok = age.notna().to_numpy()
        print(f"  age range {age.min():.1f}-{age.max():.1f}, median "
              f"{age.median():.1f}. EVERY patient is pediatric; the 'oldest "
              f"band'\n  subgroup reduces n and is still not adult. This is a "
              "limitation, not a control.")
        q = age.quantile(0.75)
        m = ok & (age >= q).to_numpy()
        rows.append(line(f"label_abnormal | age>={q:.1f}", X[m], lab[m], grp[m]))
        # does appearance encode age? if so, "appearance shift" partly means
        # "the patients are younger", which is not a shortcut at all
        young = (age < age.median()).to_numpy().astype(int)
        rows.append(line("is_below_median_age", X[ok], young[ok], grp[ok]))

    print("\n" + "=" * 78)
    print("6. MURA-WRIST vs GRAZ-WRIST SEPARABILITY, FULL COVERAGE")
    print("=" * 78)
    print("  An earlier n=1500 estimate put this at AUC 0.879 on descriptors "
          "alone --\n  more separable than most cross-anatomy pairs. Recomputed "
          "at full coverage:")
    Xs = np.vstack([gg[FEATURES].to_numpy(float), mm[FEATURES].to_numpy(float)])
    ys = np.r_[np.ones(len(gg), int), np.zeros(len(mm), int)]
    gs = np.r_[gg.patient_group.to_numpy(), mm.patient_group.to_numpy()]
    rows.append(line("dataset (GRAZ vs MURA-wrist)", Xs, ys, gs))
    cmp = pd.DataFrame({"GRAZ": gg[FEATURES].mean(),
                        "MURA_XR_WRIST": mm[FEATURES].mean()})
    cmp["abs_diff"] = (cmp.GRAZ - cmp.MURA_XR_WRIST).abs()
    print("\n" + cmp.round(3).to_string())
    print("\n  These are the descriptors the projection is fitted on. A map "
          "fitted on GRAZ\n  is a DIFFERENT linear map from one fitted on MURA "
          "even with identical code.")

    out = pd.DataFrame(rows)
    out.to_csv(cfg.out, index=False)
    print(f"\nwritten to {cfg.out}")


if __name__ == "__main__":
    main()
