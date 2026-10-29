#!/usr/bin/env python3
"""Budget cells for the FracAtlas hand target.

Runs the readout_shrinkage budget design (refit, shrinkage grid and cross-validated
selections) on FracAtlas hand. FracAtlas has no patient identifier, so budgets
are drawn by image and no in-domain reference is fitted.

Output (under --out-dir):
    fracatlas_budget_cells_<source>_<frac anatomy>.csv

Usage:
    python fracatlas_budget_cells.py --source XR_WRIST --frac-anatomy XR_HAND --gpu 0

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

import mskmodel as M
import preprocessing_variants as P
from train_source_models import ckpt_path, CsvAppender
from dataset_shift import load_index, load_source, cell_rng
from readout_probe import (extract, sample_studies, BUDGETS, N_RESAMPLE,
                           ALL_ANATOMIES, DEFAULT_SOURCE)
from fracatlas_transfer import verify_readable
from readout_shrinkage import (C_GRID, LAMBDA_GRID, PUBLISHED_C,
                                 PAPER_BACKBONES, source_head,
                                 raw_space_readout, blend,
                                 choose_c_lam_nested, verify_source_head)

FRAC_ANATOMIES = ["XR_HAND", "XR_SHOULDER"]
EVAL_FRACTION = 0.40
SPLIT_SEED_TAG = "stage_b14_frac_split_v1"

ROW_COLUMNS = ["source", "target", "frac_anatomy", "backbone", "seed",
               "budget", "resample", "setting", "lam", "C", "chosen_C",
               "chosen_lam", "n_train_studies", "n_train_images",
               "n_eval_studies", "n_eval_images", "eval_prevalence",
               "n_unreadable_dropped",
               "auc_study", "auc_image", "zeroshot_auc", "delta_vs_zeroshot",
               "harm", "n_multiscan_dropped", "minutes"]


def frac_pools(index, frac_anatomy, keep_multiscan, eval_fraction):
    """Split FracAtlas into a budget pool and a held-out eval half.

    Stratified by label so both halves carry both classes, grouped by image id
    (which is all FracAtlas offers), deterministic from a fixed tag so the
    split does not move between backbones, seeds or re-runs.
    """
    f = M.frame_for(index, "fracatlas", frac_anatomy, "test", "label_fracture")
    f = f.reset_index(drop=True)

    # MANDATORY, and it must happen BEFORE the split. FracAtlas ships truncated
    # JPEGs that sit almost entirely in Non_fractured/, so they are damaged
    # NEGATIVES. The split below is label-stratified, which would spread them
    # across both the budget pool and the evaluation half -- straight into the
    # harm rate, which is the number this script exists to produce.
    # fracatlas_transfer.py documents this guard as non-optional; b14 omitted it
    # and that was a bug.
    n_before = len(f)
    f = verify_readable(f)
    n_unreadable = n_before - len(f)

    n_drop = 0
    if not keep_multiscan and "meta_multiscan" in f.columns:
        ms = pd.to_numeric(f["meta_multiscan"], errors="coerce").fillna(0) == 1
        n_drop = int(ms.sum())
        f = f[~ms.to_numpy()].reset_index(drop=True)
    elif "meta_multiscan" not in f.columns:
        print("[frac] NOTE: no meta_multiscan column in the index; the "
              "repeat-imaging channel cannot be removed. Report this.")

    # MURA/GRAZ/FracAtlas patient_groups are namespaced, so overlap is
    # impossible by construction. Assert anyway; it is free.
    mura_pg = set(index[index.dataset == "mura"]["patient_group"])
    bad = set(f["patient_group"]) & mura_pg
    if bad:
        raise AssertionError(f"FracAtlas/MURA patient_group collision: "
                             f"{sorted(bad)[:3]}")

    rng = cell_rng(SPLIT_SEED_TAG, frac_anatomy, int(keep_multiscan))
    eval_mask = np.zeros(len(f), dtype=bool)
    for lab in sorted(f["label"].unique()):
        idx = np.flatnonzero(f["label"].to_numpy() == lab)
        rng.shuffle(idx)
        k = int(round(eval_fraction * len(idx)))
        eval_mask[idx[:k]] = True

    va = f[eval_mask].reset_index(drop=True)
    tr = f[~eval_mask].reset_index(drop=True)
    for name, fr in (("pool", tr), ("eval", va)):
        if len(np.unique(fr["label"].to_numpy())) < 2:
            raise SystemExit(f"!! FracAtlas {frac_anatomy} {name} half has one "
                             f"class only; lower --eval-fraction or pick "
                             f"another anatomy.")
    return tr, va, n_drop, n_unreadable


def summarise(path: Path, backbones=None) -> None:
    d = pd.read_csv(path, comment="#")
    bb = backbones or PAPER_BACKBONES
    d = d[d.backbone.isin(bb)]
    if not len(d):
        raise SystemExit("!! no rows for the paper backbones in this file")
    m = d[d.budget > 0].copy()

    z = d[d.budget == 0]
    print(f"\nTARGET: FracAtlas {d.frac_anatomy.iloc[0]}  "
          f"source {d.source.iloc[0]}")
    print(f"eval half: {int(z.n_eval_studies.iloc[0])} studies / "
          f"{int(z.n_eval_images.iloc[0])} images, "
          f"prevalence {z.eval_prevalence.iloc[0]:.3f}")
    print(f"multiscan images dropped: {int(z.n_multiscan_dropped.iloc[0])}")
    if "n_unreadable_dropped" in z.columns:
        print(f"unreadable (truncated) images dropped: "
              f"{int(z.n_unreadable_dropped.iloc[0])}  "
              f"(class-correlated: expected to be negatives)")
    print(f"zero-shot study AUC: {z.auc_study.mean():.4f} "
          f"(mean over {len(z)} backbone-seed cells)")

    print("\nHARM RATE (fraction of cells below the transferred model)\n")
    print(m.pivot_table(index="budget", columns="setting", values="harm",
                        aggfunc="mean").round(4).to_string())

    print("\nMEAN AUC CHANGE vs the transferred model\n")
    print(m.pivot_table(index="budget", columns="setting",
                        values="delta_vs_zeroshot", aggfunc="mean")
          .round(4).to_string())

    for s in ("refit_C1", "refit_nestedC", "shrink_nested"):
        g = m[m.setting == s]
        if not len(g):
            continue
        print(f"\n[{s}] overall harm {g.harm.mean():.4f} "
              f"({int(g.harm.sum())}/{len(g)} cells)")
        bb_ = g.groupby("budget").delta_vs_zeroshot.mean()
        pos = bb_[bb_ > 0]
        print(f"  break-even budget: "
              f"{int(pos.index.min()) if len(pos) else 'never'}"
              "   (smallest budget with positive mean gain; check it stays "
              "positive at every larger budget before quoting it)")


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
    ap.add_argument("--frac-anatomy", default="XR_HAND", choices=FRAC_ANATOMIES)
    ap.add_argument("--eval-fraction", type=float, default=EVAL_FRACTION)
    ap.add_argument("--keep-multiscan", action="store_true",
                    help="retain images flagged multiscan; they are dropped by "
                         "default because they are the one repeat-imaging "
                         "channel the metadata exposes")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--summarise", type=str, default=None)
    cfg = ap.parse_args()

    if cfg.summarise:
        summarise(Path(cfg.summarise))
        return

    SRC, FA = cfg.source, cfg.frac_anatomy
    out = Path(cfg.out_dir) / (cfg.out_name or
                               f"fracatlas_budget_cells_{SRC}_{FA}.csv")
    # same two guards as oracle_readout_cv/b13: no schema drift, no double append
    from readout_shrinkage import preflight as _pf
    if out.exists():
        hdr = None
        with out.open() as fh:
            for line in fh:
                if not line.startswith("#"):
                    hdr = [c.strip() for c in line.rstrip("\n").split(",")]
                    break
        if hdr is not None:
            if set(hdr) != set(ROW_COLUMNS):
                raise SystemExit(f"!! {out.name} exists with a different "
                                 f"schema; mv it aside or pass --out-name")
            with out.open() as fh:
                n = sum(1 for line in fh if not line.startswith("#")) - 1
            if n > 0:
                raise SystemExit(f"!! {out.name} already holds {n} data row(s); "
                                 f"appending would duplicate cells. mv it aside "
                                 f"or pass --out-name")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()
    index = load_index(cfg)
    ck_dir = Path(cfg.ckpt_root) / "source_models"

    tr, va, n_drop, n_unread = frac_pools(index, FA, cfg.keep_multiscan,
                                          cfg.eval_fraction)
    n_pool_studies = int(tr.study.nunique())
    print(f"[frac] {FA}: pool {n_pool_studies} studies / {len(tr)} images | "
          f"eval {va.study.nunique()} studies / {len(va)} images | "
          f"multiscan dropped {n_drop} | unreadable dropped {n_unread} | "
          f"eval prevalence {va['label'].mean():.3f}")
    budgets = [b for b in cfg.budgets if b <= n_pool_studies]
    if not budgets:
        raise SystemExit(f"!! pool has {n_pool_studies} studies, smaller than "
                         f"the smallest budget {min(cfg.budgets)}")
    if len(budgets) < len(cfg.budgets):
        print(f"[frac] budgets {sorted(set(cfg.budgets) - set(budgets))} exceed "
              f"the pool and are skipped")

    prov = M.provenance({
        "script": "fracatlas_budget_cells.py v1", "source": SRC,
        "frac_anatomy": FA, "budgets": budgets, "resamples": cfg.resamples,
        "c_grid": list(cfg.c_grid), "lam_grid": list(cfg.lam_grid),
        "eval_fraction": cfg.eval_fraction, "multiscan_dropped": n_drop,
        "unreadable_dropped": n_unread,
        "split_seed_tag": SPLIT_SEED_TAG,
        "note": "budget curve and harm rate on FracAtlas. NO reference model, "
                "NO encoder residual: FracAtlas has no patient column so any "
                "in-domain ceiling leaks, and the leak would inflate the "
                "residual toward the paper's own conclusion. Residual leakage "
                "inflates the ADAPTED arm only, which biases AGAINST the harm "
                "finding reported here."})
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    res = CsvAppender(out, prov)

    ev = {"n_eval_studies": int(va.study.nunique()), "n_eval_images": len(va),
          "eval_prevalence": float(va["label"].mean()),
          "n_multiscan_dropped": n_drop,
          "n_unreadable_dropped": n_unread}

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            model, _ = load_source(ckpt_path(ck_dir, SRC, bb, seed), bb, device)
            w_src, b_src = source_head(model)
            t0 = time.time()

            ftr, _, _ = extract(model, tr, device, cfg, P.load_standard)
            fva, zva, yva = extract(model, va, device, cfg, P.load_standard)
            ftr = ftr.astype(np.float64)
            fva = fva.astype(np.float64)
            st = va.study.to_numpy()
            zs = score(zva, yva, st)["auc_study"]
            dmax = verify_source_head(fva, zva, yva, st, w_src, b_src, zs,
                                      f"{bb}/s{seed}/FRAC_{FA}")

            base = {"source": SRC, "target": f"FRAC_{FA}", "frac_anatomy": FA,
                    "backbone": bb, "seed": seed, **ev}
            rows = [{**base, "budget": 0, "resample": 0, "setting": "zeroshot",
                     "lam": np.nan, "C": np.nan, "chosen_C": np.nan,
                     "chosen_lam": np.nan, "n_train_studies": 0,
                     "n_train_images": 0, "auc_study": zs,
                     "auc_image": score(zva, yva, st)["auc_image"],
                     "zeroshot_auc": zs, "delta_vs_zeroshot": 0.0, "harm": 0,
                     "minutes": (time.time() - t0) / 60.0}]

            for b in budgets:
                for r in range(cfg.resamples):
                    rng = cell_rng("fracbudget", bb, seed, FA, b, r)
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

                    fits = []
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

                    for setting, lam, C, w, b_ in fits:
                        # At lam == 0 the deployed model IS the transferred
                        # model, and zva is what it actually produces. The
                        # float64 reconstruction approximates zva, not the
                        # reverse, and can sit half a rank unit below it --
                        # which would record a SPURIOUS harm at lam=0 in the
                        # very grid used to choose the shrinkage rule.
                        sc_ = score(zva if float(lam) == 0.0
                                    else fva @ w + b_, yva, st)
                        a = sc_["auc_study"]
                        rows.append({
                            **base, "budget": b, "resample": r,
                            "setting": setting, "lam": lam, "C": C,
                            "chosen_C": c_n, "chosen_lam": lam_n,
                            "n_train_studies": int(sub.study.nunique()),
                            "n_train_images": int(len(sub)),
                            "auc_study": a, "auc_image": sc_["auc_image"],
                            "zeroshot_auc": zs, "delta_vs_zeroshot": a - zs,
                            "harm": int(a < zs) if np.isfinite(a) else 0,
                            "minutes": np.nan})

            rows = [{c: r_.get(c, np.nan) for c in ROW_COLUMNS} for r_ in rows]
            res.write(rows)

            sh = [r_ for r_ in rows if r_["setting"] == "shrink_nested"]
            rf = [r_ for r_ in rows if r_["setting"] == "refit_nestedC"]
            print(f"  [frac-budget] {M.BACKBONE_SHORT[bb]:<12s} s{seed} "
                  f"{FA:<12s} zs={zs:.4f} | harm refit="
                  f"{np.mean([x['harm'] for x in rf]) if rf else float('nan'):.3f} "
                  f"shrink="
                  f"{np.mean([x['harm'] for x in sh]) if sh else float('nan'):.3f}"
                  f" | dmax={dmax:.1e} | {(time.time()-t0)/60:.1f} min",
                  flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"\nwrote {out}")
    print(f"Now run:  python fracatlas_budget_cells.py --summarise {out}")


if __name__ == "__main__":
    sys.exit(main())
