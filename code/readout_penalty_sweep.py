#!/usr/bin/env python3
"""Readout refit across a grid of l2 penalties.

Repeats the budget refit of readout_probe.py with the inverse penalty C swept
over {1e-4, ..., 10} and with C chosen by patient-grouped cross-validation
inside the budget sample (the own-penalty refit), to check whether refit harm
at small budgets is an artefact of the fixed penalty.

Output (under --out-dir):
    readout_penalty_sweep_<source>.csv

Usage:
    python readout_penalty_sweep.py --source XR_WRIST --gpu 0

Requires a GPU for feature extraction.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskdata
import mskmodel as M
import preprocessing_variants as P
from train_source_models import ckpt_path, CsvAppender
from dataset_shift import graz_frame, load_index, load_source, cell_rng

# Import the probe's own helpers. Do NOT re-implement them: if the sampling
# differs from the published run, any change in the harm rate is
# uninterpretable and the whole exercise is wasted.
from readout_probe import (extract, sample_studies, BUDGETS, N_RESAMPLE,
                           ALL_ANATOMIES, DEFAULT_SOURCE)

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
PUBLISHED_C = 1.0          # what readout_probe.fit_probe used


def fit_scores(ftr, ytr, groups, fva, c_grid):
    """Fit one head per C, plus a nested-CV C. Returns {label: scores}.

    The nested C is selected by patient-grouped CV inside the budget sample.
    It never sees the validation set, so it is a legitimate deployment
    procedure rather than an oracle.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    if len(np.unique(ytr)) < 2:
        return {}

    out = {}
    for c in c_grid:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=c))
        clf.fit(ftr, ytr)
        out[f"C={c:g}"] = clf.decision_function(fva)

    # ---- nested, patient-grouped CV over the same grid --------------------
    n_groups = len(np.unique(groups))
    k = int(min(5, n_groups))
    chosen = PUBLISHED_C
    if k >= 2:
        best_auc = -np.inf
        for c in c_grid:
            aucs = []
            for tr_i, va_i in GroupKFold(n_splits=k).split(ftr, ytr, groups):
                if len(np.unique(ytr[tr_i])) < 2 or len(np.unique(ytr[va_i])) < 2:
                    continue
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=3000, C=c))
                m.fit(ftr[tr_i], ytr[tr_i])
                aucs.append(roc_auc_score(ytr[va_i], m.decision_function(ftr[va_i])))
            if aucs and np.mean(aucs) > best_auc:
                best_auc, chosen = float(np.mean(aucs)), c
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=chosen))
    clf.fit(ftr, ytr)
    out["nestedCV"] = clf.decision_function(fva)
    out["_chosen_C"] = chosen
    return out


def summarise(path: Path) -> None:
    """Harm rate by budget and C. This is the table that decides the claim."""
    d = pd.read_csv(path, comment="#")
    zs = (d[d.budget == 0][["backbone", "seed", "target", "source", "auc_study"]]
          .rename(columns={"auc_study": "zs"}).drop_duplicates())
    m = d[d.budget > 0].merge(zs, on=["backbone", "seed", "target", "source"])
    m["harm"] = m.auc_study < m.zs
    piv = m.pivot_table(index="budget", columns="setting", values="harm",
                        aggfunc="mean")
    cols = [c for c in piv.columns if c.startswith("C=")] + \
           [c for c in piv.columns if c == "nestedCV"]
    print("\nFRACTION OF CELLS WHERE RE-FITTING IS WORSE THAN THE SOURCE HEAD")
    print("(the published run corresponds to the C=1 column)\n")
    print(piv[cols].round(3).to_string())
    print("\nMEAN AUC GAIN OVER ZERO SHOT\n")
    m["gain"] = m.auc_study - m.zs
    print(m.pivot_table(index="budget", columns="setting", values="gain",
                        aggfunc="mean")[cols].round(4).to_string())
    if "chosen_C" in m.columns:
        print("\nC SELECTED BY NESTED CV, BY BUDGET\n")
        sel = m[m.setting == "nestedCV"].groupby("budget").chosen_C
        print(sel.agg(["median", "min", "max"]).to_string())
    print("\nRead this against the paper: if the C=1 column is an outlier and "
          "\nnestedCV shows a much lower harm rate, the published claim is a "
          "\nregularisation artifact and must be withdrawn.")


def main() -> None:
    import torch
    from shortcut_reliance import score

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-root", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--backbones", nargs="+", default=list(M.BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(M.SEEDS))
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLE)
    ap.add_argument("--c-grid", nargs="+", type=float, default=C_GRID)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--source", default=DEFAULT_SOURCE, choices=ALL_ANATOMIES)
    ap.add_argument("--skip-graz", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--summarise", type=str, default=None,
                    help="path to an existing sweep CSV; prints the harm table "
                         "and exits without running anything")
    cfg = ap.parse_args()

    if cfg.summarise:
        summarise(Path(cfg.summarise))
        return

    SOURCE = cfg.source
    MURA_TARGETS = [a for a in ALL_ANATOMIES if a != SOURCE]

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({
        "script": "readout_penalty_sweep.py v1", "source": SOURCE,
        "budgets": list(cfg.budgets), "resamples": cfg.resamples,
        "c_grid": list(cfg.c_grid), "published_C": PUBLISHED_C,
        "note": "regularisation sensitivity of the budget probe; extraction, "
                "pools and sampling are imported from readout_probe.py so the "
                "ONLY difference from the published run is C"})
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(cfg.out_dir) / f"readout_penalty_sweep_{SOURCE}.csv"
    res = CsvAppender(out, prov)
    index = load_index(cfg)
    ck_dir = Path(cfg.ckpt_root) / "source_models"

    # ---- target pools: identical construction to readout_probe.main ------
    pools = {}
    for t in MURA_TARGETS:
        tr = M.frame_for(index, "mura", t, "train").reset_index(drop=True)
        tr = mskdata.drop_source_patients(index, tr, SOURCE, "train")
        va = M.frame_for(index, "mura", t, "valid").reset_index(drop=True)
        pools[t] = (tr.reset_index(drop=True), va)
    if not cfg.skip_graz:
        gtr = graz_frame(index, "train", "label_abnormal")
        gva = graz_frame(index, "valid", "label_abnormal")
        ie = lambda f: pd.to_numeric(f.meta_initial_exam, errors="coerce") == 1
        pools["GRAZ_initial"] = (gtr[ie(gtr).to_numpy()].reset_index(drop=True),
                                 gva[ie(gva).to_numpy()].reset_index(drop=True))

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            cp = ckpt_path(ck_dir, SOURCE, bb, seed)
            model, _ = load_source(cp, bb, device)
            for t, (tr, va) in pools.items():
                t0 = time.time()
                ftr, _, ytr_all = extract(model, tr, device, cfg, P.load_standard)
                fva, zva, yva = extract(model, va, device, cfg, P.load_standard)
                st = va.study.to_numpy()
                zs = score(zva, yva, st)["auc_study"]

                rows = [{"backbone": bb, "seed": seed, "target": t,
                         "source": SOURCE, "budget": 0, "resample": 0,
                         "setting": "zeroshot", "chosen_C": np.nan,
                         "n_train_studies": 0, "auc_study": zs,
                         "delta_vs_zeroshot": 0.0, "zeroshot_auc": zs,
                         "minutes": (time.time() - t0) / 60.0}]

                for b in cfg.budgets:
                    if tr.study.nunique() < b:
                        continue
                    for r in range(cfg.resamples):
                        # SAME rng stream as the published run, so the samples
                        # are identical and C is the only thing that varies.
                        rng = cell_rng("probe", bb, seed, t, b, r)
                        sub = sample_studies(tr, b, rng)
                        idx = sub.index.to_numpy()
                        y = tr.loc[idx, "label"].to_numpy()
                        grp = tr.loc[idx, "patient_group"].to_numpy()
                        fits = fit_scores(ftr[idx], y, grp, fva, cfg.c_grid)
                        if not fits:
                            continue
                        chosen = fits.pop("_chosen_C", np.nan)
                        for setting, s in fits.items():
                            a = score(s, yva, st)["auc_study"]
                            rows.append({
                                "backbone": bb, "seed": seed, "target": t,
                                "source": SOURCE, "budget": b, "resample": r,
                                "setting": setting,
                                "chosen_C": chosen if setting == "nestedCV" else np.nan,
                                "n_train_studies": int(sub.study.nunique()),
                                "auc_study": a, "delta_vs_zeroshot": a - zs,
                                "zeroshot_auc": zs, "minutes": np.nan})
                res.write(rows)
                print(f"  [creg] {M.BACKBONE_SHORT[bb]:<12s} s{seed} {t:<14s} "
                      f"zs={zs:.4f}  {len(rows)-1} fits  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"\nwrote {out}")
    print("Now run:  python readout_penalty_sweep.py --summarise", out)


if __name__ == "__main__":
    sys.exit(main())
