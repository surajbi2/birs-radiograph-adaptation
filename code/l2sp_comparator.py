#!/usr/bin/env python3
"""L2-SP comparator: refit the readout with an L2 penalty toward the source readout.

Instead of penalising the readout toward zero, the refit is penalised toward the
transferred (source) readout. Features, budget samples, random streams and
evaluation splits are those of

    readout_shrinkage.py     six MURA targets, GRAZPEDWRI-DX and MFXR
    fracatlas_budget_cells.py   FracAtlas hand

so every row pairs with an existing b13/b14 cell. Pools, extraction, sampling,
source readout and scoring are imported from those stages, not copied.

Estimator. In the standardised space of the refits, z = (f - mu) / sigma fitted
on the budget sample, the source readout is w_s' = w_s * sigma and
b_s' = b_s + w_s . mu. The objective (scikit-learn convention) is

    C * sum_i logloss(w, b; z_i, y_i) + 1/2 ||w - w_s'||^2 + 1/2 (b - b_s')^2

minimised by L-BFGS-B started at (w_s', b_s'), so C -> 0 recovers the
transferred model. The result is mapped back to raw feature space as in
readout_shrinkage.raw_space_readout.

Settings written per cell:
    l2sp_cv   C chosen by the patient-grouped nested CV of readout_shrinkage (GroupKFold,
              k = min(5, patient groups), image-level fold AUC, first maximum
              wins, C = 1 if no fold is usable) over {1e-8, ..., 1e-5} plus the
              readout_shrinkage grid 1e-4..10 (see L2SP_C_GRID)
    l2sp_C1   C = 1, mirroring refit_C1
The abstain-then-L2-SP policy is derived from these rows in make_paper_outputs.py.

The four values below 1e-4 were added before the full run, when the gate showed
that at 1e-4 L2-SP could not yet reproduce the transferred model.

Modes:
    extract   one feature cache per (backbone, seed, target) under --cache-dir (GPU)
    gate      re-scores a sample of cells: the cached features must reproduce the
              stored b13/b14 cells within 1e-6, and L2-SP at the smallest C must
              reproduce the transferred model within 1e-3; exits 1 on failure
    fit       all cells, in parallel over units (CPU)

Output (under --cache-dir):
    l2sp_comparator_XR_WRIST.csv

Usage:
    python l2sp_comparator.py extract --gpu 0
    python l2sp_comparator.py gate
    python l2sp_comparator.py fit --jobs 48
"""

from __future__ import annotations

import argparse
import os
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
from readout_shrinkage import (C_GRID, PUBLISHED_C, PAPER_BACKBONES,
                                 source_head, raw_space_readout, blend,
                                 choose_c_lam_nested, LAMBDA_GRID,
                                 verify_source_head)
from fracatlas_budget_cells import frac_pools, EVAL_FRACTION

FRAC_ANATOMY = "XR_HAND"
FRAC_TARGET = f"FRAC_{FRAC_ANATOMY}"
DEFAULT_CACHE = Path("/path/to/imagenet_feature_cache")
ROW_COLUMNS = ["source", "target", "backbone", "seed", "budget", "resample",
               "setting", "lam", "C", "chosen_C", "chosen_lam",
               "n_train_studies", "auc_study", "auc_image", "zeroshot_auc",
               "delta_vs_zeroshot", "harm", "minutes",
               "cv_auc", "converged", "n_iter"]
L2SP_OPTS = {"maxiter": 15000, "ftol": 1e-12, "gtol": 1e-8, "maxcor": 20}
# Grid extension, fixed before the full run: the refit grid 1e-4..10 cannot reach
# the transferred model for L2-SP. In standardised space the source head is
# small (|w_s'| ~ 0.1-0.5) while the data gradient at the source is ~1e2-1e3 x C,
# so at C = 1e-4 the fit is already 8-31% of |w_s'| away and study AUC moves by
# up to 0.11 (gate criterion 2 failed on 20 of 36 cells). Four smaller values are
# added; the gate's zero-shot check runs at the smallest.
L2SP_C_GRID = [1e-8, 1e-7, 1e-6, 1e-5] + list(C_GRID)


# ------------------------------------------------------------------- pools --

def pools(index, source=DEFAULT_SOURCE):
    """{target: (pool frame, eval frame, rng tag parts)} exactly as b13/b14."""
    from mfxr_external import mfxr_frame
    out = {}
    for t in [a for a in ALL_ANATOMIES if a != source]:
        tr = M.frame_for(index, "mura", t, "train").reset_index(drop=True)
        tr = mskdata.drop_source_patients(index, tr, source, "train")
        va = M.frame_for(index, "mura", t, "valid").reset_index(drop=True)
        out[t] = (tr.reset_index(drop=True), va, ("probe", t))
    gtr = graz_frame(index, "train", "label_abnormal")
    gva = graz_frame(index, "valid", "label_abnormal")
    ie = lambda f: pd.to_numeric(f.meta_initial_exam, errors="coerce") == 1
    out["GRAZ_initial"] = (gtr[ie(gtr).to_numpy()].reset_index(drop=True),
                           gva[ie(gva).to_numpy()].reset_index(drop=True),
                           ("probe", "GRAZ_initial"))
    out["MFXR"] = (mfxr_frame(index, "train").reset_index(drop=True),
                   mfxr_frame(index, "valid").reset_index(drop=True),
                   ("probe", "MFXR"))
    tr, va, _, _ = frac_pools(index, FRAC_ANATOMY, False, EVAL_FRACTION)
    out[FRAC_TARGET] = (tr, va, ("fracbudget", FRAC_ANATOMY))
    return out


def rng_for(tag, bb, seed, b, r):
    kind, name = tag
    return cell_rng(kind, bb, seed, name, b, r)


def cache_path(cache_dir: Path, bb, seed, t) -> Path:
    return cache_dir / "features" / f"{bb}__s{seed}__{t}.npz"


# ------------------------------------------------------------------- L2-SP --

def l2sp_fit(F, y, w_src, b_src, C):
    """(w, b, converged, n_iter) in RAW feature space; see module docstring."""
    from scipy.optimize import minimize
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2:
        return None, None, False, 0
    sc = StandardScaler().fit(F)
    mu = sc.mean_.astype(np.float64)
    sig = np.maximum(sc.scale_.astype(np.float64), 1e-12)
    Z = (F - mu) / sig
    w0 = w_src * sig
    b0 = float(b_src + w_src @ mu)
    yf = y.astype(np.float64)
    d = Z.shape[1]

    def f(theta):
        w, b = theta[:d], theta[d]
        z = Z @ w + b
        loss = np.logaddexp(0.0, z) - yf * z
        p = 0.5 * (1.0 + np.tanh(0.5 * z))           # stable sigmoid
        r = p - yf
        dw, db = w - w0, b - b0
        val = C * loss.sum() + 0.5 * (dw @ dw) + 0.5 * db * db
        g = np.empty(d + 1)
        g[:d] = C * (Z.T @ r) + dw
        g[d] = C * r.sum() + db
        return val, g

    res = minimize(f, np.concatenate([w0, [b0]]), jac=True, method="L-BFGS-B",
                   options=L2SP_OPTS)
    w = res.x[:d] / sig
    b = float(res.x[d] - w @ mu)
    return w, b, bool(res.success), int(res.nit)


def choose_c_l2sp(F, y, groups, w_src, b_src, c_grid):
    """Same folds, criterion and tie rule as readout_shrinkage.choose_c_lam_nested."""
    from sklearn.model_selection import GroupKFold
    k = int(min(5, len(np.unique(groups))))
    if k < 2 or len(np.unique(y)) < 2:
        return PUBLISHED_C, np.nan
    folds = list(GroupKFold(n_splits=k).split(F, y, groups))
    best_c, best_auc = PUBLISHED_C, -np.inf
    for c in c_grid:
        aucs = []
        for tr_i, va_i in folds:
            if len(np.unique(y[tr_i])) < 2 or len(np.unique(y[va_i])) < 2:
                continue
            w, b, _, _ = l2sp_fit(F[tr_i], y[tr_i], w_src, b_src, c)
            if w is None:
                continue
            a = M.safe_auc(y[va_i], F[va_i] @ w + b)
            if np.isfinite(a):
                aucs.append(a)
        if aucs:
            m = float(np.mean(aucs))
            if m > best_auc:
                best_auc, best_c = m, c
    return best_c, (best_auc if np.isfinite(best_auc) else np.nan)


# ----------------------------------------------------------------- extract --

def do_extract(cfg, index, units):
    import torch
    from shortcut_reliance import score
    device = torch.device(f"cuda:{cfg.gpu}")
    M.raise_fd_limit()
    M.configure_sharing()
    P_ = pools(index)
    ck_dir = Path(cfg.ckpt_root) / "source_models"
    for bb in cfg.backbones:
        for seed in cfg.seeds:
            todo = [t for t in cfg.targets if (bb, seed, t) in units
                    and not cache_path(cfg.cache_dir, bb, seed, t).exists()]
            if not todo:
                continue
            model, _ = load_source(ckpt_path(ck_dir, DEFAULT_SOURCE, bb, seed),
                                   bb, device)
            w_src, b_src = source_head(model)
            for t in todo:
                t0 = time.time()
                tr, va, _ = P_[t]
                ftr, _, _ = extract(model, tr, device, cfg, P.load_standard)
                fva, zva, yva = extract(model, va, device, cfg, P.load_standard)
                st = va.study.to_numpy()
                zs = score(zva, yva, st)["auc_study"]
                verify_source_head(fva.astype(np.float64), zva, yva, st,
                                   w_src, b_src, zs, f"{bb}/s{seed}/{t}")
                p = cache_path(cfg.cache_dir, bb, seed, t)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp.npz")
                np.savez(tmp, ftr=ftr, fva=fva, zva=zva, yva=yva,
                         tr_uid=tr.uid.to_numpy().astype(str),
                         va_uid=va.uid.to_numpy().astype(str),
                         w_src=w_src, b_src=np.array(b_src), zs=np.array(zs))
                os.replace(tmp, p)
                print(f"  [extract] {M.BACKBONE_SHORT[bb]:<12s} s{seed} {t:<14s} "
                      f"pool {len(tr)} eval {len(va)} zs={zs:.4f} "
                      f"{(time.time() - t0) / 60:.2f} min", flush=True)
            del model
            torch.cuda.empty_cache()


# --------------------------------------------------------------------- fit --

def load_unit(cache_dir, P_, bb, seed, t):
    z = np.load(cache_path(cache_dir, bb, seed, t), allow_pickle=False)
    tr, va, tag = P_[t]
    if not (np.array_equal(z["tr_uid"], tr.uid.to_numpy().astype(str))
            and np.array_equal(z["va_uid"], va.uid.to_numpy().astype(str))):
        raise SystemExit(f"!! cache rows for {bb}/s{seed}/{t} do not match "
                         "the rebuilt pools; delete the cache and re-extract")
    return z, tr, va, tag


def cells(tr, budgets, resamples, tag, bb, seed):
    """Yield (b, r, idx, y, groups, n_studies), skipping exactly as b13/b14."""
    n_pool = tr.study.nunique()
    for b in budgets:
        if n_pool < b:
            continue
        for r in range(resamples):
            sub = sample_studies(tr, b, rng_for(tag, bb, seed, b, r))
            idx = sub.index.to_numpy()
            y = tr.loc[idx, "label"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            yield (b, r, idx, y, tr.loc[idx, "patient_group"].to_numpy(),
                   int(sub.study.nunique()))


def fit_unit(args):
    """All cells of one (backbone, seed, target): l2sp_cv and l2sp_C1 rows."""
    cache_dir, index_path, bb, seed, t, budgets, resamples, c_grid = args
    from shortcut_reliance import score
    index = _INDEX if _INDEX is not None else pd.read_csv(index_path, comment="#",
                                                          low_memory=False)
    P_ = _POOLS if _POOLS is not None else pools(index)
    z, tr, va, tag = load_unit(cache_dir, P_, bb, seed, t)
    ftr = z["ftr"].astype(np.float64)
    fva = z["fva"].astype(np.float64)
    yva, st = z["yva"], va.study.to_numpy()
    w_src, b_src, zs = z["w_src"], float(z["b_src"]), float(z["zs"])
    base = {"source": DEFAULT_SOURCE, "target": t, "backbone": bb, "seed": seed,
            "lam": np.nan, "chosen_lam": np.nan, "zeroshot_auc": zs}
    rows = []
    for b, r, idx, y, grp, n_st in cells(tr, budgets, resamples, tag, bb, seed):
        t0 = time.time()
        F = ftr[idx]
        c_cv, cv_auc = choose_c_l2sp(F, y, grp, w_src, b_src, c_grid)
        for setting, C in (("l2sp_cv", c_cv), ("l2sp_C1", PUBLISHED_C)):
            w, bb_, ok, nit = l2sp_fit(F, y, w_src, b_src, C)
            s_ = score(fva @ w + bb_, yva, st)
            a = s_["auc_study"]
            rows.append({**base, "budget": b, "resample": r, "setting": setting,
                         "C": C, "chosen_C": c_cv, "n_train_studies": n_st,
                         "auc_study": a, "auc_image": s_["auc_image"],
                         "delta_vs_zeroshot": a - zs,
                         "harm": int(a < zs) if np.isfinite(a) else 0,
                         "minutes": (time.time() - t0) / 60.0,
                         "cv_auc": cv_auc if setting == "l2sp_cv" else np.nan,
                         "converged": int(ok), "n_iter": nit})
    return (bb, seed, t), rows


_INDEX = None
_POOLS = None


def _init_worker():
    """Pools are built once in the parent and inherited by fork; each worker
    is pinned to one BLAS thread so 48 workers do not oversubscribe."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


# -------------------------------------------------------------------- gate --

def do_gate(cfg, index):
    """Re-extracted features must reproduce stored b13/b14 cells (<= 1e-6), and
    L2-SP at the smallest C must reproduce zero-shot (<= 1e-3)."""
    from shortcut_reliance import score
    P_ = pools(index)
    stored = pd.concat([pd.read_csv(p, comment="#") for p in
                        (cfg.b13, cfg.b14)], ignore_index=True)
    K = ["target", "backbone", "seed", "budget", "resample", "setting"]
    stored = stored.set_index(K)
    worst = {"refit_nestedC": 0.0, "shrink_lam0.5": 0.0, "l2sp_Cmin_vs_zs": 0.0}
    n = 0
    fail = []
    for (bb, seed, t) in cfg.gate_units:
        z, tr, va, tag = load_unit(cfg.cache_dir, P_, bb, seed, t)
        ftr, fva = z["ftr"].astype(np.float64), z["fva"].astype(np.float64)
        yva, st = z["yva"], va.study.to_numpy()
        w_src, b_src, zs = z["w_src"], float(z["b_src"]), float(z["zs"])
        for b, r, idx, y, grp, _ in cells(tr, cfg.gate_budgets, 1, tag, bb, seed):
            F = ftr[idx]
            c_n, lam_n, _ = choose_c_lam_nested(F, y, grp, w_src, b_src,
                                                list(C_GRID), list(LAMBDA_GRID))
            w_nc, b_nc = raw_space_readout(F, y, c_n)
            got = {"refit_nestedC": score(fva @ w_nc + b_nc, yva, st)["auc_study"]}
            wl, bl = blend(w_src, b_src, w_nc, b_nc, 0.5)
            got["shrink_lam0.5"] = score(fva @ wl + bl, yva, st)["auc_study"]
            for s, a in got.items():
                ref = float(stored.loc[(t, bb, seed, b, r, s), "auc_study"])
                d = abs(a - ref)
                worst[s] = max(worst[s], d)
                if d > 1e-6:
                    fail.append(f"{t}/{bb}/s{seed}/n{b}/r{r} {s}: {a!r} vs stored {ref!r}")
            w, b2, ok, _ = l2sp_fit(F, y, w_src, b_src, min(L2SP_C_GRID))
            a0 = score(fva @ w + b2, yva, st)["auc_study"]
            worst["l2sp_Cmin_vs_zs"] = max(worst["l2sp_Cmin_vs_zs"], abs(a0 - zs))
            if abs(a0 - zs) > 1e-3:
                fail.append(f"{t}/{bb}/s{seed}/n{b}/r{r} l2sp C={min(L2SP_C_GRID)}: "
                            f"{a0!r} vs zero-shot {zs!r}")
            n += 1
    print(f"[gate] {n} cells over {len(cfg.gate_units)} units")
    for k, v in worst.items():
        print(f"[gate] max |diff| {k:<18s} {v:.3e}")
    if fail:
        print("[gate] FAIL\n  " + "\n  ".join(fail[:20]))
        return 1
    print("[gate] PASS")
    return 0


# -------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extract", "gate", "fit"])
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-root", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--backbones", nargs="+", default=PAPER_BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(M.SEEDS))
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLE)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--gate-only-units", action="store_true",
                    help="extract: only the gate units")
    ap.add_argument("--b13", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "readout_shrinkage_XR_WRIST.csv"))
    ap.add_argument("--b14", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "fracatlas_budget_cells_XR_WRIST_XR_HAND.csv"))
    ap.add_argument("--out-name", default="l2sp_comparator_XR_WRIST.csv")
    cfg = ap.parse_args()
    cfg.targets = cfg.targets or ([a for a in ALL_ANATOMIES if a != DEFAULT_SOURCE]
                                  + ["GRAZ_initial", "MFXR", FRAC_TARGET])
    # Gate sample: every backbone, seed 42, one anatomy and all three held-out
    # targets, three budgets, resample 0.
    cfg.gate_units = [(bb, 42, t) for bb in cfg.backbones
                      for t in ("XR_HAND", "GRAZ_initial", "MFXR", FRAC_TARGET)]
    cfg.gate_budgets = [10, 100, 500]

    index = load_index(cfg)
    if cfg.mode == "extract":
        units = (set(cfg.gate_units) if cfg.gate_only_units else
                 {(bb, s, t) for bb in cfg.backbones for s in cfg.seeds
                  for t in cfg.targets})
        do_extract(cfg, index, units)
        return 0
    if cfg.mode == "gate":
        return do_gate(cfg, index)

    out = cfg.cache_dir / cfg.out_name
    if out.exists():
        raise SystemExit(f"!! {out} exists; move it aside first (no appends)")
    units = [(bb, s, t) for bb in cfg.backbones for s in cfg.seeds
             for t in cfg.targets]
    missing = [u for u in units if not cache_path(cfg.cache_dir, *u).exists()]
    if missing:
        raise SystemExit(f"!! {len(missing)} unit(s) not extracted, e.g. {missing[:3]}")
    prov = M.provenance({
        "script": "l2sp_comparator.py v1", "source": DEFAULT_SOURCE,
        "c_grid_l2sp_cv": list(L2SP_C_GRID), "C_l2sp_C1": PUBLISHED_C,
        "amendment_1": "C grid extended to 1e-8 (see L2SP_C_GRID comment)",
        "budgets": list(cfg.budgets),
        "resamples": cfg.resamples, "l2sp_opts": L2SP_OPTS,
        "cache_dir": str(cfg.cache_dir),
        "note": "L2-SP readout toward the source head; cells pair with b13/b14"})
    from multiprocessing import get_context
    global _INDEX, _POOLS
    _INDEX, _POOLS = index, pools(index)
    t0 = time.time()
    args = [(cfg.cache_dir, cfg.index, bb, s, t, list(cfg.budgets),
             cfg.resamples, list(L2SP_C_GRID)) for bb, s, t in units]
    with get_context("fork").Pool(cfg.jobs, initializer=_init_worker) as pool:
        done = {}
        for key, rows in pool.imap_unordered(fit_unit, args):
            done[key] = rows
            print(f"  [fit] {len(done)}/{len(units)} {key} {len(rows)} rows "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    res = CsvAppender(out, prov)
    for u in units:
        res.write([{c: r_.get(c, np.nan) for c in ROW_COLUMNS} for r_ in done[u]])
    print(f"\nwrote {out}  ({sum(len(v) for v in done.values())} rows, "
          f"{(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
