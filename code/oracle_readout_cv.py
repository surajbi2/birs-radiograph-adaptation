#!/usr/bin/env python3
"""Oracle readouts with a cross-validated penalty.

Refits the oracle readout (all target training labels) with C chosen by
patient-grouped cross-validation instead of a fixed C, so the part of the
transfer loss that a readout can recover is not understated by an
under-regularised fit.

Output (under --out-dir):
    oracle_readout_cv_<source>.csv

Usage:
    python oracle_readout_cv.py --source XR_WRIST --gpu 0
    python oracle_readout_cv.py --summarise <csv>

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
from dataset_shift import graz_frame, load_index, load_source
from readout_probe import extract, ALL_ANATOMIES

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
PUBLISHED_C = 1.0
GAP_GATE = 0.10
PAPER_BACKBONES = ["densenet121", "convnext_tiny", "swin_tiny_patch4_window7_224"]

# Declared once so the output file can be checked BEFORE any extraction runs.
# CsvAppender refuses a mid-run schema change, which is correct, but finding
# out after the first cell means the GPU work is wasted.
ROW_COLUMNS = ["source", "target", "backbone", "seed", "C", "setting",
               "is_published_C", "n_train_images", "n_train_studies",
               "zeroshot", "oracle_readout", "ceiling", "gap",
               "readout_recoverable", "encoder_residual", "readout_frac",
               "minutes"]


def dump_cell(dump_dir, cond, target, backbone, seed, frame, logit, label):
    """Per-image scores for one (condition, target, backbone, seed) cell.

    Written so the paired differences can be bootstrapped with patients as the
    resampling unit. The AUCs this script reports are aggregates and cannot
    carry an interval; these dumps are the only way to put one on them. Naming
    matches bootstrap_intervals.py: <cond>__<target>__<backbone>__seed<k>.npz
    """
    d = Path(dump_dir)
    d.mkdir(parents=True, exist_ok=True)
    pat = (frame["patient_group"].to_numpy() if "patient_group" in frame
           else frame["patient"].to_numpy() if "patient" in frame
           else np.array([str(s) for s in frame["study"].to_numpy()]))
    np.savez_compressed(
        d / f"{cond}__{target}__{backbone}__seed{seed}.npz",
        uid=frame["uid"].to_numpy().astype(str) if "uid" in frame
            else frame.index.to_numpy().astype(str),
        study=frame["study"].to_numpy().astype(str),
        patient=pat.astype(str),
        label=np.asarray(label).astype(np.int64),
        logit=np.asarray(logit).astype(np.float64))


def preflight(out: Path) -> None:
    """Stop now, not after an hour of extraction, if the file cannot be appended."""
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
            f"!! {out.name} already exists with a different schema.\n"
            f"   columns this run would add: {missing}\n"
            f"   columns in the file but not this run: {extra}\n"
            "   That file is from the grid-only run and is still valid; this run\n"
            "   adds the nested-CV setting. Either:\n"
            f"     mv {out} {out.with_name(out.stem + '_gridonly.csv')}\n"
            "   or pass --out-name to write somewhere else. Do NOT delete it\n"
            "   until the new run has finished.")

    # Matching schema is not enough. A re-run with the same schema appends a
    # second copy of every row, which leaves every mean untouched and silently
    # doubles every count and shrinks every standard deviation. That reached
    # Table 2 once. Refuse rather than append into a populated file.
    import csv as _csv
    with out.open() as fh:
        n = sum(1 for line in fh if not line.startswith("#")) - 1
    if n > 0:
        raise SystemExit(
            f"!! {out.name} already holds {n} data row(s).\n"
            "   Appending would duplicate cells: the AUC columns would not\n"
            "   move, but n_gated would double and every SD would shrink, and\n"
            "   nothing downstream would look wrong. If you are re-running,\n"
            f"     mv {out} {out.with_name(out.stem + '_prev.csv')}\n"
            "   and if you only want the per-image dumps, pass --out-name to\n"
            "   send the csv somewhere disposable.")


def fit_direction_c(feats: np.ndarray, y: np.ndarray, C: float):
    """`readout_angle.fit_direction` with C exposed. Nothing else changed.

    The scaler is undone (`w / scale_`) because the source head sees raw trunk
    output; comparing a standardised-space vector against a raw-space one
    would produce a cosine with no meaning.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2:
        return None
    sc = StandardScaler().fit(feats)
    lr = LogisticRegression(max_iter=5000, C=C).fit(sc.transform(feats), y)
    w_eff = lr.coef_.ravel() / np.maximum(sc.scale_, 1e-12)
    n = np.linalg.norm(w_eff)
    return (w_eff / n) if n > 0 else None


def choose_c_nested(feats, y, groups, c_grid):
    """Pick C by patient-grouped CV INSIDE the target training pool.

    The validation set is never touched. Selecting C by validation AUC would
    be the same class of error as tuning a threshold on test data, and the
    encoder residual computed that way would not be a quantity anyone could
    act on.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    k = int(min(5, len(np.unique(groups))))
    if k < 2 or len(np.unique(y)) < 2:
        return PUBLISHED_C, np.nan
    best_c, best_auc = PUBLISHED_C, -np.inf
    for c in c_grid:
        aucs = []
        for tr_i, va_i in GroupKFold(n_splits=k).split(feats, y, groups):
            if len(np.unique(y[tr_i])) < 2 or len(np.unique(y[va_i])) < 2:
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, C=c))
            m.fit(feats[tr_i], y[tr_i])
            aucs.append(roc_auc_score(y[va_i], m.decision_function(feats[va_i])))
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_c = float(np.mean(aucs)), c
    return best_c, best_auc


def summarise(path: Path, backbones=None) -> None:
    d = pd.read_csv(path, comment="#")
    bb = backbones or PAPER_BACKBONES
    d = d[d.backbone.isin(bb)]
    if "setting" not in d.columns:
        # v1 grid-only file: synthesise the label so the rest works unchanged
        d["setting"] = d.C.map(lambda c: f"C={c:g}")
        print("[note] this file has no nested-CV rows (grid-only run). "
              "The nested block below will be skipped.")
    d["arm"] = np.where(d.target.astype(str).str.startswith("GRAZ"),
                        "dataset", "anatomy")

    print("\nORACLE READOUT AUC by target and C")
    print(d.pivot_table(index="target", columns="C", values="oracle_readout")
          .round(4).to_string())

    print("\nENCODER RESIDUAL (reference - oracle) by target and C")
    piv = d.pivot_table(index="target", columns="C", values="encoder_residual")
    print(piv.round(4).to_string())

    print("\nTHE HEADLINE: residual by shift type, and the ratio")
    rows = []
    for c, g in d.groupby("C"):
        an = g[g.arm == "anatomy"].groupby("target").encoder_residual.mean()
        gz = g[g.arm == "dataset"].encoder_residual.mean()
        if not len(an) or not np.isfinite(gz):
            continue
        rows.append({"C": c, "anatomy_mean": an.mean(),
                     "anatomy_worst": an.max(), "dataset": gz,
                     "ratio_to_mean": gz / an.mean() if an.mean() else np.nan,
                     "ratio_to_worst": gz / an.max() if an.max() else np.nan})
    t = pd.DataFrame(rows).set_index("C")
    print(t.round(4).to_string())
    if "setting" in d.columns and (d.setting == "nested").any():
        nd = d[d.setting == "nested"]
        an = nd[nd.arm == "anatomy"].groupby("target").encoder_residual.mean()
        gz = nd[nd.arm == "dataset"].encoder_residual.mean()
        print("\nNESTED-CV C (selected inside the training pool, validation "
              "never touched)")
        print(f"  anatomy mean   {an.mean():.4f}")
        print(f"  anatomy worst  {an.max():.4f}  ({an.idxmax()})")
        print(f"  dataset        {gz:.4f}")
        print(f"  ratio to worst {gz / an.max():.2f}")
        print("  C chosen, median by target:")
        print(nd.groupby("target").C.median().to_string())
        print("\n  >>> THESE are the numbers to put in the paper. <<<")

    print("\nThe paper reports anatomy 0.046, dataset 0.111, ratio to worst 1.87,")
    print("all at C = 1. Compare the C = 1 row above against the best-penalty row.")
    print("If ratio_to_worst is not stable across the grid, the abstract is wrong.")

    print("\nREADOUT FRACTION (gap >= 0.10 cells only) by target and C")
    gd = d[d.gap >= GAP_GATE]
    if len(gd):
        print(gd.pivot_table(index="target", columns="C", values="readout_frac")
              .round(3).to_string())
        print("\nn gated cells by target and C")
        print(gd.pivot_table(index="target", columns="C", values="readout_frac",
                             aggfunc="size").to_string())


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
    ap.add_argument("--c-grid", nargs="+", type=float, default=C_GRID)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--source", default="XR_WRIST", choices=ALL_ANATOMIES)
    ap.add_argument("--skip-graz", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-nested", action="store_true",
                    help="skip the nested-CV C selection (it is the defensible "
                         "one; only skip for a quick grid-only pass)")
    ap.add_argument("--out-name", type=str, default=None,
                    help="output filename; defaults to "
                         "oracle_readout_cv_<SOURCE>.csv")
    ap.add_argument("--summarise", type=str, default=None)
    ap.add_argument("--dump-dir", type=str, default=None,
                    help="if set, write per-image scores for the three "
                         "decomposition conditions (zeroshot, oracle, "
                         "reference) at the NESTED penalty, so the paired "
                         "differences can be bootstrapped. Inference only; it "
                         "does not change any number this script reports.")
    cfg = ap.parse_args()

    if cfg.summarise:
        summarise(Path(cfg.summarise))
        return

    SRC = cfg.source
    preflight(Path(cfg.out_dir) / (cfg.out_name or
                                   f"oracle_readout_cv_{SRC}.csv"))
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()

    index = load_index(cfg)
    ck_dir = Path(cfg.ckpt_root) / "source_models"

    # ---- ceilings, read from the SAME files readout_angle uses --------------
    a = pd.read_csv(Path(M.DEFAULT_RESULTS) / "source_models" /
                    "anatomy_transfer_arms_merged.csv", comment="#", low_memory=False)
    ceil = (a[a.condition == "ceiling"]
            .groupby(["target", "backbone"]).auc_study.mean().to_dict())
    GRAZ_CEILING = {}
    gpath = Path(cfg.out_dir) / "graz_initial_only_ceiling.csv"
    if gpath.exists():
        gc = pd.read_csv(gpath, comment="#")
        gc = gc[gc.score_label == "label_abnormal"]
        GRAZ_CEILING = gc.groupby("backbone").zs_auc_study.mean().to_dict()
        print("[graz-ceiling] per backbone: "
              + str({k: round(v, 4) for k, v in GRAZ_CEILING.items()}))
    else:
        raise SystemExit(
            f"!! {gpath} not found. Without the per-backbone GRAZ ceiling the "
            "dataset-shift residual cannot be computed, and that residual is "
            "the whole point of this script.")

    # ---- pools: identical construction to readout_angle.main ---------------
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

    prov = M.provenance({
        "script": "oracle_readout_cv.py v1", "source": SRC,
        "c_grid": list(cfg.c_grid), "published_C": PUBLISHED_C,
        "note": "regularisation sensitivity of the ORACLE readout and therefore "
                "of the encoder residual; pools, extraction and ceilings are "
                "taken from readout_angle.py so the ONLY difference from the "
                "published run is C"})
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(cfg.out_dir) / (cfg.out_name or
                               f"oracle_readout_cv_{SRC}.csv")
    preflight(out)
    res = CsvAppender(out, prov)

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            model, _ = load_source(ckpt_path(ck_dir, SRC, bb, seed), bb, device)
            for t, (tr, va) in pools.items():
                t0 = time.time()
                ftr, _, ytr = extract(model, tr, device, cfg, P.load_standard)
                fva, zva, yva = extract(model, va, device, cfg, P.load_standard)
                st = va.study.to_numpy()
                zs = score(zva, yva, st)["auc_study"]
                c = (GRAZ_CEILING.get(bb, np.nan) if t.startswith("GRAZ")
                     else ceil.get((t, bb), np.nan))
                if not np.isfinite(c):
                    print(f"  !! no reference for {t}/{bb}; skipping")
                    continue
                gap = c - zs
                rows = []
                grid = list(cfg.c_grid)
                nested_c = np.nan
                if not cfg.no_nested:
                    grp = tr["patient_group"].to_numpy()
                    nested_c, _ = choose_c_nested(ftr, ytr, grp, cfg.c_grid)
                    grid = grid + [("nested", nested_c)]
                w_nested = None
                for C in grid:
                    tag = None
                    if isinstance(C, tuple):
                        tag, C = C
                    w = fit_direction_c(ftr, ytr, C)
                    if w is None:
                        continue
                    if tag == "nested":
                        w_nested = w
                    orc = score(fva @ w, yva, st)["auc_study"]
                    rows.append({
                        "source": SRC, "target": t, "backbone": bb, "seed": seed,
                        "C": C, "setting": tag or f"C={C:g}",
                        "is_published_C": bool(tag is None and C == PUBLISHED_C),
                        "n_train_images": len(tr),
                        "n_train_studies": int(tr.study.nunique()),
                        "zeroshot": zs, "oracle_readout": orc, "ceiling": c,
                        "gap": gap, "readout_recoverable": orc - zs,
                        "encoder_residual": c - orc,
                        "readout_frac": ((orc - zs) / gap
                                         if gap >= GAP_GATE else np.nan),
                        "minutes": (time.time() - t0) / 60.0})
                rows = [{c: r.get(c, np.nan) for c in ROW_COLUMNS} for r in rows]
                res.write(rows)

                if cfg.dump_dir and w_nested is not None:
                    # zero-shot and oracle share this cell's validation frame,
                    # so the pairing the bootstrap depends on is exact here.
                    dump_cell(cfg.dump_dir, "zeroshot", t, bb, seed,
                              va, zva, yva)
                    dump_cell(cfg.dump_dir, "oracle", t, bb, seed,
                              va, fva @ w_nested, yva)
                    # The reference is a separate model: the target's own
                    # Stage A checkpoint scored on the target's own validation
                    # split. GRAZ is skipped because its reference was trained
                    # by graz_initial_only_ceiling and is already dumped as initial_only__*.
                    if not t.startswith("GRAZ"):
                        rp = ckpt_path(ck_dir, t, bb, seed)
                        if Path(rp).exists():
                            rmodel, _ = load_source(rp, bb, device)
                            _, rz, ry = extract(rmodel, va, device, cfg,
                                                P.load_standard)
                            dump_cell(cfg.dump_dir, "reference", t, bb, seed,
                                      va, rz, ry)
                            del rmodel
                        else:
                            print(f"  !! no reference checkpoint {rp}; "
                                  f"reference not dumped for {t}/{bb}/s{seed}")
                best = max(rows, key=lambda r: r["oracle_readout"]) if rows else None
                pub = next((r for r in rows if r["is_published_C"]), None)
                if best and pub:
                    print(f"  [oracle] {M.BACKBONE_SHORT[bb]:<12s} s{seed} "
                          f"{t:<14s} zs={zs:.4f} ref={c:.4f} | "
                          f"C=1 orc={pub['oracle_readout']:.4f} "
                          f"resid={pub['encoder_residual']:.4f} | "
                          f"best C={best['C']:g} orc={best['oracle_readout']:.4f} "
                          f"resid={best['encoder_residual']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"\nwrote {out}")
    print(f"Now run:  python oracle_readout_cv.py --summarise {out}")


if __name__ == "__main__":
    sys.exit(main())
