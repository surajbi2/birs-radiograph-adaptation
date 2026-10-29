#!/usr/bin/env python3
"""Shrink the refitted readout toward the source readout.

Refits the readout on the budget sample in raw feature space and blends it with
the source readout:

    w(lam) = (1 - lam) * w_src + lam * w_tgt
    b(lam) = (1 - lam) * b_src + lam * b_tgt

lam = 0 is the transferred model and lam = 1 the unrestricted refit. For every
budget cell the script records the result at each lam on the grid, at a lam and
C chosen jointly by patient-grouped cross-validation inside the sample (the
joint-penalty refit), and at a fixed C. Pools, budgets, resamples and the random
stream are those of readout_probe.py, so only the readout differs.

Guards: the reconstructed source readout must reproduce the transferred model's
study AUC before any budget is fitted, and a budget sample holding a single
class is skipped.

Output (under --out-dir):
    readout_shrinkage_<source>.csv (six MURA targets, GRAZPEDWRI-DX and MFXR)

Usage:
    python readout_shrinkage.py --source XR_WRIST --gpu 0
    python readout_shrinkage.py --summarise <csv>

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
from readout_probe import (extract, sample_studies, BUDGETS, N_RESAMPLE,
                           ALL_ANATOMIES, DEFAULT_SOURCE)

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
PUBLISHED_C = 1.0
PAPER_BACKBONES = ["densenet121", "convnext_tiny", "swin_tiny_patch4_window7_224"]

# Tolerances for the two runtime identity checks.
#
# The logit check is loose because extraction runs in float32 on GPU while the
# reconstruction is float64 on CPU.
#
# The AUC check CANNOT be a fixed constant. AUC is a rank statistic whose
# resolution is one rank unit = 1/(n_pos*n_neg) at study level. A float
# difference of 1e-16 between two tied study scores flips that pair, moving
# AUC by half a rank unit -- observed as 1.173e-05 on the FracAtlas XR_HAND
# eval half (42,626 pairs). A flat constant is the wrong SHAPE: 1e-4 is ~4
# rank units there but only 0.06 of a rank unit on a 25-vs-25 subgroup, i.e.
# tighter than a single tie, so it would fail exactly where ties are most
# likely. The tolerance is expressed in rank units and scales with the
# evaluation set.
LOGIT_ATOL = 2e-3
AUC_RANK_UNITS = 4.0      # a broken blend moves AUC by ~1e-2, which is >100x
AUC_ATOL_FLOOR = 1e-12    # this even on the largest cohort in the project

ROW_COLUMNS = ["source", "target", "backbone", "seed", "budget", "resample",
               "setting", "lam", "C", "chosen_C", "chosen_lam",
               "n_train_studies", "auc_study", "auc_image", "zeroshot_auc",
               "delta_vs_zeroshot", "harm", "minutes"]


# ------------------------------------------------------------------ readout --

def source_head(model):
    """(w, b) of the source model's linear head, float64, raw feature space.

    mskmodel.BinaryNet is `self.head = nn.Linear(feat_dim, 1)` applied to the
    trunk output, so the extracted logit is exactly `f @ w + b`. That identity
    is asserted per cell in main(); if BinaryNet ever gains a non-linear head
    this function is the thing that must change.
    """
    w = model.head.weight.detach().cpu().numpy().astype(np.float64).ravel()
    b = float(model.head.bias.detach().cpu().numpy().astype(np.float64).ravel()[0])
    return w, b


def auc_rank_tol(study_label):
    """Numerical tolerance for a study-level AUC identity, in rank units.

    One rank unit is 1/(n_pos*n_neg): the smallest amount a single pair can
    move the statistic. A tie contributes half a unit. Anything below a few
    rank units is float noise in the ordering, not a real difference.
    """
    y = np.asarray(study_label)
    n_pos = int((y == 1).sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return AUC_ATOL_FLOOR
    return max(AUC_RANK_UNITS / float(n_pos * n_neg), AUC_ATOL_FLOOR)

def raw_space_readout(feats, y, C):
    """Logistic readout returned in RAW feature space as (w, b).

    Fitting is done on standardised features, as everywhere else in this
    project, then converted back exactly:

        logit = coef @ ((f - mean) / scale) + intercept
              = (coef / scale) @ f + (intercept - (coef / scale) @ mean)

    The second term is NOT optional here: the blend feeds a sigmoid before
    study aggregation, so an offset changes the study-level score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2:
        return None, None
    sc = StandardScaler().fit(feats)
    lr = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(feats), y)
    scale = np.maximum(sc.scale_.astype(np.float64), 1e-12)
    w = lr.coef_.ravel().astype(np.float64) / scale
    b = float(lr.intercept_[0]) - float(w @ sc.mean_.astype(np.float64))
    return w, b


def blend(w_src, b_src, w_t, b_t, lam):
    """Convex combination in raw feature space. lam = 0 is the source head."""
    lam = float(lam)
    return ((1.0 - lam) * w_src + lam * w_t,
            (1.0 - lam) * b_src + lam * b_t)


def choose_c_lam_nested(ftr, ytr, groups, w_src, b_src, c_grid, lam_grid):
    """Pick (C, lam) by patient-grouped CV inside the budget sample.

    Returns (C, lam, mean_cv_auc). The validation set is never seen. Selection
    uses image-level AUC, matching `readout_penalty_sweep.fit_scores` and
    `oracle_readout_cv.choose_c_nested`; at 10 studies a study-level fold AUC would be
    computed on ~2 studies and would be noise.

    The target readout is fitted once per (C, fold) and reused across lam,
    which is what keeps this cheap.
    """
    from sklearn.model_selection import GroupKFold

    k = int(min(5, len(np.unique(groups))))
    if k < 2 or len(np.unique(ytr)) < 2:
        return PUBLISHED_C, 0.0, np.nan

    folds = list(GroupKFold(n_splits=k).split(ftr, ytr, groups))
    best_c, best_lam, best_auc = PUBLISHED_C, 0.0, -np.inf

    for c in c_grid:
        fitted = []
        for tr_i, va_i in folds:
            if len(np.unique(ytr[tr_i])) < 2 or len(np.unique(ytr[va_i])) < 2:
                continue
            w_t, b_t = raw_space_readout(ftr[tr_i], ytr[tr_i], c)
            if w_t is None:
                continue
            fitted.append((va_i, w_t, b_t))
        if not fitted:
            continue
        for lam in lam_grid:
            aucs = []
            for va_i, w_t, b_t in fitted:
                w, b = blend(w_src, b_src, w_t, b_t, lam)
                a = M.safe_auc(ytr[va_i], ftr[va_i] @ w + b)
                if np.isfinite(a):
                    aucs.append(a)
            if aucs:
                m = float(np.mean(aucs))
                if m > best_auc:
                    best_auc, best_c, best_lam = m, c, float(lam)
    return best_c, best_lam, best_auc


# ----------------------------------------------------------------- plumbing --

def preflight(out: Path) -> None:
    """Refuse a schema change or a second append into a populated file.

    Appending into a populated file leaves every mean untouched while doubling
    every count and shrinking every SD, and nothing downstream looks wrong.
    Same guard as oracle_readout_cv, same reason.
    """
    if not out.exists():
        return
    header = None
    with out.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                header = [c.strip() for c in line.rstrip("\n").split(",")]
                break
    if header is None:
        return
    if set(header) != set(ROW_COLUMNS):
        missing = sorted(set(ROW_COLUMNS) - set(header))
        extra = sorted(set(header) - set(ROW_COLUMNS))
        raise SystemExit(
            f"!! {out.name} exists with a different schema.\n"
            f"   this run would add: {missing}\n"
            f"   in the file but not this run: {extra}\n"
            f"   mv {out} {out.with_name(out.stem + '_prev.csv')}  (or --out-name)")
    with out.open() as fh:
        n = sum(1 for line in fh if not line.startswith("#")) - 1
    if n > 0:
        raise SystemExit(
            f"!! {out.name} already holds {n} data row(s); appending would "
            f"duplicate cells.\n   mv {out} "
            f"{out.with_name(out.stem + '_prev.csv')}  (or --out-name)")


def verify_source_head(fva, zva, yva, st, w_src, b_src, zs, where):
    """The two identities this script's correctness rests on.

    Checked per cell, before any budget is fitted, so a mismatch costs one
    extraction rather than a whole run.
    """
    recon = fva @ w_src + b_src
    d = float(np.nanmax(np.abs(recon - zva)))
    if not np.isfinite(d) or d > LOGIT_ATOL:
        raise SystemExit(
            f"!! [{where}] reconstructed source logit does not match the "
            f"extracted logit (max |diff| = {d:.3e} > {LOGIT_ATOL:.0e}).\n"
            "   `f @ head.weight + head.bias` is no longer the model's logit, "
            "so every blend below would be meaningless. Check BinaryNet.forward.")
    from shortcut_reliance import score
    a0 = score(recon, yva, st)["auc_study"]
    _, _, sy = M.aggregate_studies(M.sigmoid(recon), yva, st)
    tol = auc_rank_tol(sy)
    if not np.isfinite(a0) or abs(a0 - zs) > tol:
        n_pos = int((np.asarray(sy) == 1).sum())
        n_neg = int(len(sy) - n_pos)
        unit = 1.0 / max(1, n_pos * n_neg)
        msg = [
            f"!! [{where}] study AUC at lam=0 is {a0!r}, zero-shot is {zs!r}.",
            f"   difference {abs(a0 - zs):.3e} = "
            f"{abs(a0 - zs) / unit:.2f} rank units",
            f"   tolerance {tol:.3e} = {AUC_RANK_UNITS:g} rank units on "
            f"{n_pos} x {n_neg} pairs",
            "   A few rank units is float noise flipping tied study pairs.",
            "   This is far more, so the lam=0 arm is NOT reproducing the",
            "   transferred model and the do-no-harm property is not being",
            "   measured.",
        ]
        raise SystemExit(chr(10).join(msg))
    return d


# ---------------------------------------------------------------- summarise --

def summarise(path: Path, backbones=None) -> None:
    d = pd.read_csv(path, comment="#")
    bb = backbones or PAPER_BACKBONES
    d = d[d.backbone.isin(bb)]
    if not len(d):
        raise SystemExit("!! no rows for the paper backbones in this file")
    m = d[d.budget > 0].copy()

    print("\nHARM RATE (fraction of cells below the transferred model)")
    print("rows = budget, cols = setting\n")
    print(m.pivot_table(index="budget", columns="setting", values="harm",
                        aggfunc="mean").round(4).to_string())

    print("\nMEAN AUC CHANGE vs the transferred model\n")
    print(m.pivot_table(index="budget", columns="setting",
                        values="delta_vs_zeroshot", aggfunc="mean")
          .round(4).to_string())

    for s in ("shrink_nested", "refit_nestedC", "refit_C1"):
        if s not in set(m.setting):
            continue
        g = m[m.setting == s]
        print(f"\n[{s}] harm rate overall {g.harm.mean():.4f} "
              f"({int(g.harm.sum())}/{len(g)} cells)")

    if {"shrink_nested", "refit_nestedC"} <= set(m.setting):
        big = m[m.budget == m.budget.max()]
        a = big[big.setting == "shrink_nested"].groupby(
            ["target", "backbone", "seed", "resample"]).auc_study.mean()
        b = big[big.setting == "refit_nestedC"].groupby(
            ["target", "backbone", "seed", "resample"]).auc_study.mean()
        j = pd.concat([a.rename("shrink"), b.rename("refit")], axis=1).dropna()
        if len(j):
            cost = float((j.refit - j.shrink).mean())
            print(f"\nPRICE OF THE GUARANTEE at budget {int(big.budget.max())}: "
                  f"{cost:+.4f} AUC (refit minus shrink, paired, n={len(j)})")
            print("  <= 0.01 supports 'the guarantee is nearly free'.")

    if "chosen_lam" in m.columns:
        sel = m[m.setting == "shrink_nested"]
        if len(sel):
            print("\nLAMBDA SELECTED BY NESTED CV, median by budget\n")
            print(sel.groupby("budget").chosen_lam.median().round(3).to_string())

    print("\nHARM RATE BY FIXED LAMBDA (sensitivity)\n")
    fx = m[m.setting.str.startswith("shrink_lam")]
    if len(fx):
        print(fx.pivot_table(index="budget", columns="lam", values="harm",
                             aggfunc="mean").round(4).to_string())


# --------------------------------------------------------------------- main --

def main() -> None:
    import torch
    from shortcut_reliance import score

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-root", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--backbones", nargs="+", default=PAPER_BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(M.SEEDS))
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLE)
    ap.add_argument("--c-grid", nargs="+", type=float, default=C_GRID)
    ap.add_argument("--lam-grid", nargs="+", type=float, default=LAMBDA_GRID)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--source", default=DEFAULT_SOURCE, choices=ALL_ANATOMIES)
    ap.add_argument("--skip-graz", action="store_true")
    ap.add_argument("--with-mfxr", action="store_true",
                    help="add MFXR as a cross-dataset target pool. MFXR has "
                         "patient-disjoint splits, so unlike FracAtlas its "
                         "budget curve needs no leakage caveat.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--summarise", type=str, default=None)
    cfg = ap.parse_args()

    if cfg.summarise:
        summarise(Path(cfg.summarise))
        return

    SRC = cfg.source
    out = Path(cfg.out_dir) / (cfg.out_name or f"readout_shrinkage_{SRC}.csv")
    preflight(out)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()

    index = load_index(cfg)
    ck_dir = Path(cfg.ckpt_root) / "source_models"

    # ---- pools: identical construction to readout_probe / readout_penalty_sweep --------
    pools = {}
    for t in [a for a in ALL_ANATOMIES if a != SRC]:
        tr = M.frame_for(index, "mura", t, "train").reset_index(drop=True)
        tr = mskdata.drop_source_patients(index, tr, SRC, "train")
        va = M.frame_for(index, "mura", t, "valid").reset_index(drop=True)
        pools[t] = (tr.reset_index(drop=True), va)
    if SRC == "XR_WRIST" and not cfg.skip_graz:
        gtr = graz_frame(index, "train", "label_abnormal")
        gva = graz_frame(index, "valid", "label_abnormal")
        ie = lambda f: pd.to_numeric(f.meta_initial_exam, errors="coerce") == 1
        pools["GRAZ_initial"] = (gtr[ie(gtr).to_numpy()].reset_index(drop=True),
                                 gva[ie(gva).to_numpy()].reset_index(drop=True))

    if cfg.with_mfxr:
        # Imported here, not at module scope: mfxr_external imports from this
        # module, so a top-level import would be circular.
        from mfxr_external import mfxr_frame
        pools["MFXR"] = (mfxr_frame(index, "train").reset_index(drop=True),
                         mfxr_frame(index, "valid").reset_index(drop=True))
        print("[mfxr] added as a cross-dataset pool; prevalence ~0.61 "
              "(case-control cohort), so absolute AUCs are not "
              "screening-comparable. The harm rate is a within-target "
              "comparison and is unaffected.")

    prov = M.provenance({
        "script": "readout_shrinkage.py v1", "source": SRC,
        "c_grid": list(cfg.c_grid), "lam_grid": list(cfg.lam_grid),
        "budgets": list(cfg.budgets), "resamples": cfg.resamples,
        "note": "readout shrinkage toward the source head. lam=0 is the "
                "transferred model exactly, lam=1 is the readout_penalty_sweep refit. "
                "Pools, extraction and the RNG stream are taken from "
                "readout_probe so the budget samples are identical to the "
                "published run and the readout is the only thing that varies."})
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    res = CsvAppender(out, prov)

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            cp = ckpt_path(ck_dir, SRC, bb, seed)
            model, _ = load_source(cp, bb, device)
            w_src, b_src = source_head(model)

            for t, (tr, va) in pools.items():
                t0 = time.time()
                ftr, _, ytr_all = extract(model, tr, device, cfg, P.load_standard)
                fva, zva, yva = extract(model, va, device, cfg, P.load_standard)
                ftr = ftr.astype(np.float64)
                fva = fva.astype(np.float64)
                st = va.study.to_numpy()
                zs = score(zva, yva, st)["auc_study"]

                dmax = verify_source_head(fva, zva, yva, st, w_src, b_src, zs,
                                          f"{bb}/s{seed}/{t}")

                rows = [{"source": SRC, "target": t, "backbone": bb,
                         "seed": seed, "budget": 0, "resample": 0,
                         "setting": "zeroshot", "lam": np.nan, "C": np.nan,
                         "chosen_C": np.nan, "chosen_lam": np.nan,
                         "n_train_studies": 0, "auc_study": zs,
                         "auc_image": score(zva, yva, st)["auc_image"],
                         "zeroshot_auc": zs, "delta_vs_zeroshot": 0.0,
                         "harm": 0, "minutes": (time.time() - t0) / 60.0}]

                for b in cfg.budgets:
                    if tr.study.nunique() < b:
                        continue
                    for r in range(cfg.resamples):
                        # SAME rng stream as readout_probe / readout_penalty_sweep, so the
                        # budget samples are bit-identical to the published run.
                        rng = cell_rng("probe", bb, seed, t, b, r)
                        sub = sample_studies(tr, b, rng)
                        idx = sub.index.to_numpy()
                        y = tr.loc[idx, "label"].to_numpy()
                        grp = tr.loc[idx, "patient_group"].to_numpy()
                        if len(np.unique(y)) < 2:
                            continue
                        fsub = ftr[idx]

                        c_n, lam_n, _ = choose_c_lam_nested(
                            fsub, y, grp, w_src, b_src,
                            list(cfg.c_grid), list(cfg.lam_grid))

                        fits = []   # (setting, lam, C, w, b)

                        w1, b1 = raw_space_readout(fsub, y, PUBLISHED_C)
                        if w1 is not None:
                            fits.append(("refit_C1", 1.0, PUBLISHED_C, w1, b1))

                        w_nc, b_nc = raw_space_readout(fsub, y, c_n)
                        if w_nc is not None:
                            fits.append(("refit_nestedC", 1.0, c_n, w_nc, b_nc))
                            wS, bS = blend(w_src, b_src, w_nc, b_nc, lam_n)
                            fits.append(("shrink_nested", lam_n, c_n, wS, bS))
                            for lam in cfg.lam_grid:
                                wL, bL = blend(w_src, b_src, w_nc, b_nc, lam)
                                fits.append((f"shrink_lam{lam:g}", float(lam),
                                             c_n, wL, bL))

                        for setting, lam, C, w, bb_ in fits:
                            # At lam == 0 the deployed model IS the transferred
                            # model, and zva is what it actually produces. The
                            # float64 reconstruction approximates zva, not the
                            # reverse, and can sit half a rank unit below it --
                            # which would record a SPURIOUS harm at lam=0 in the
                            # very grid used to choose the shrinkage rule.
                            s = (zva if float(lam) == 0.0
                                 else fva @ w + bb_)
                            sc_ = score(s, yva, st)
                            a = sc_["auc_study"]
                            rows.append({
                                "source": SRC, "target": t, "backbone": bb,
                                "seed": seed, "budget": b, "resample": r,
                                "setting": setting, "lam": lam, "C": C,
                                "chosen_C": c_n, "chosen_lam": lam_n,
                                "n_train_studies": int(sub.study.nunique()),
                                "auc_study": a, "auc_image": sc_["auc_image"],
                                "zeroshot_auc": zs,
                                "delta_vs_zeroshot": a - zs,
                                "harm": int(a < zs) if np.isfinite(a) else 0,
                                "minutes": np.nan})

                rows = [{c: r_.get(c, np.nan) for c in ROW_COLUMNS}
                        for r_ in rows]
                res.write(rows)

                sh = [r_ for r_ in rows if r_["setting"] == "shrink_nested"]
                rf = [r_ for r_ in rows if r_["setting"] == "refit_nestedC"]
                hs = np.mean([r_["harm"] for r_ in sh]) if sh else np.nan
                hr = np.mean([r_["harm"] for r_ in rf]) if rf else np.nan
                print(f"  [shrink] {M.BACKBONE_SHORT[bb]:<12s} s{seed} "
                      f"{t:<14s} zs={zs:.4f} | harm refit={hr:.3f} "
                      f"shrink={hs:.3f} | logit-recon dmax={dmax:.1e} | "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"\nwrote {out}")
    print(f"Now run:  python readout_shrinkage.py --summarise {out}")


if __name__ == "__main__":
    sys.exit(main())
