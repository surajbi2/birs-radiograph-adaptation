#!/usr/bin/env python3
"""MFXR as a second external dataset: reference models and readout cells.

MFXR splits by patient (the case id in the file name), so an in-domain reference
model can be trained without leakage. Stage 1 trains a reference model per
(backbone, seed) with the preprocessing_variants training recipe; stage 2 decomposes the
transfer loss as in oracle_readout_cv:

    G = reference - zeroshot     total loss
    R = oracle    - zeroshot     recovered by a refitted readout
    E = reference - oracle       encoder residual

The oracle readout is fitted on frozen source features with C chosen by
patient-grouped cross-validation inside the MFXR training pool.

Output (under --out-dir):
    mfxr_external_<source>.csv

Usage:
    python mfxr_external.py --source XR_WRIST --gpu 0
    python mfxr_external.py --stage reference --gpu 0

Requires a GPU.
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
from dataset_shift import load_index, load_source
from readout_probe import extract, ALL_ANATOMIES, DEFAULT_SOURCE
from readout_shrinkage import (C_GRID, PUBLISHED_C, PAPER_BACKBONES,
                                 source_head, verify_source_head)

MFXR_ANATOMY = "XR_FOREARM"
GAP_GATE = 0.10

ROW_COLUMNS = ["source", "target", "backbone", "seed", "C", "setting",
               "n_train_images", "n_train_studies", "train_prev",
               "n_eval_images", "n_eval_studies", "eval_prev",
               "zeroshot", "oracle_readout", "reference", "gap",
               "readout_recoverable", "encoder_residual", "readout_frac",
               "final_loss", "minutes"]


def mfxr_frame(index: pd.DataFrame, split: str) -> pd.DataFrame:
    f = M.frame_for(index, "mfxr", MFXR_ANATOMY, split, "label_abnormal")
    f = f.reset_index(drop=True)
    # Namespaced ids make collision impossible by construction. Assert anyway.
    other = set(index[index.dataset != "mfxr"]["patient_group"])
    bad = set(f["patient_group"]) & other
    if bad:
        raise AssertionError(f"MFXR patient_group collision: {sorted(bad)[:3]}")
    return f


def mfxr_ckpt(ckpt_root: Path, backbone: str, seed: int) -> Path:
    return (Path(ckpt_root) / "mfxr_reference" /
            f"mfxr_label_abnormal__standard__{backbone}__seed{seed}.pt")


def fit_oracle(feats, y, groups, c_grid):
    """Oracle readout on frozen source features, C by patient-grouped CV.

    Returns (w_raw, b_raw, C). Selection never sees the evaluation half, so
    this is a procedure a deploying team could actually run, not an oracle in
    the cheating sense -- the only thing 'oracle' means here is that it uses
    ALL available target labels.
    """
    from readout_shrinkage import raw_space_readout
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    k = int(min(5, len(np.unique(groups))))
    best_c = PUBLISHED_C
    if k >= 2 and len(np.unique(y)) >= 2:
        folds = list(GroupKFold(n_splits=k).split(feats, y, groups))
        best_auc = -np.inf
        for c in c_grid:
            aucs = []
            for tr_i, va_i in folds:
                if len(np.unique(y[tr_i])) < 2 or len(np.unique(y[va_i])) < 2:
                    continue
                w, b = raw_space_readout(feats[tr_i], y[tr_i], c)
                if w is None:
                    continue
                aucs.append(roc_auc_score(y[va_i], feats[va_i] @ w + b))
            if aucs and np.mean(aucs) > best_auc:
                best_auc, best_c = float(np.mean(aucs)), c
    w, b = raw_space_readout(feats, y, best_c)
    return w, b, best_c


def summarise(path: Path, backbones=None) -> None:
    d = pd.read_csv(path, comment="#")
    d = d[d.backbone.isin(backbones or PAPER_BACKBONES)]
    if not len(d):
        raise SystemExit("!! no rows for the paper backbones")
    print(f"\nMFXR as a cross-dataset target, source {d.source.iloc[0]}")
    print(f"  train {int(d.n_train_studies.iloc[0])} studies / "
          f"{int(d.n_train_images.iloc[0])} images, prev "
          f"{d.train_prev.iloc[0]:.3f}")
    print(f"  eval  {int(d.n_eval_studies.iloc[0])} studies / "
          f"{int(d.n_eval_images.iloc[0])} images, prev "
          f"{d.eval_prev.iloc[0]:.3f}")
    print("  (case-control cohort: absolute AUCs are not screening-comparable)")
    print("\nDECOMPOSITION, mean over backbone x seed\n")
    print(d[["zeroshot", "oracle_readout", "reference", "gap",
             "readout_recoverable", "encoder_residual"]].mean()
          .round(4).to_string())
    print("\nby backbone\n")
    print(d.groupby("backbone")[["zeroshot", "oracle_readout", "reference",
                                 "encoder_residual"]].mean()
          .round(4).to_string())
    print(f"\nENCODER RESIDUAL  mean {d.encoder_residual.mean():.4f}  "
          f"sd {d.encoder_residual.std():.4f}  n={len(d)} cells")
    print("\n>>> Compare against the GRAZ residual and the cross-anatomy mean "
          "from\n    oracle_readout_cv. Two cross-dataset observations instead of one "
          "is the point.")


def main() -> int:
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
    ap.add_argument("--source", default=DEFAULT_SOURCE, choices=ALL_ANATOMIES)
    ap.add_argument("--stage", choices=["reference", "decompose", "all"],
                    default="all")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--summarise", type=str, default=None)
    cfg = ap.parse_args()

    if cfg.summarise:
        summarise(Path(cfg.summarise))
        return 0

    SRC = cfg.source
    out = Path(cfg.out_dir) / (cfg.out_name or f"mfxr_external_{SRC}.csv")
    if cfg.stage in ("decompose", "all") and out.exists():
        with out.open() as fh:
            n = sum(1 for line in fh if not line.startswith("#")) - 1
        if n > 0:
            raise SystemExit(
                f"!! {out.name} already holds {n} data row(s); appending would "
                f"duplicate cells and silently shrink every SD.\n"
                f"   mv {out} {out.with_name(out.stem + '_prev.csv')}  "
                f"(or --out-name)")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()
    index = load_index(cfg)

    tr = mfxr_frame(index, "train")
    va = mfxr_frame(index, "valid")
    print(f"[mfxr] train {len(tr)} images / {tr.study.nunique()} studies "
          f"prev={tr.label.mean():.3f} | valid {len(va)} images / "
          f"{va.study.nunique()} studies prev={va.label.mean():.3f}")
    print("[mfxr] case-control cohort; absolute AUCs are not comparable to a "
          "screening setting. Within-target differences are.")

    ck_root = Path(cfg.ckpt_root)
    ref_auc, ref_loss = {}, {}

    # ------------------------------------------------------- stage 1: train --
    if cfg.stage in ("reference", "all"):
        for bb in cfg.backbones:
            for seed in cfg.seeds:
                cp = mfxr_ckpt(ck_root, bb, seed)
                cp.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                if cp.exists():
                    print(f"  [load] {cp.name}", flush=True)
                    model, _ = load_source(cp, bb, device)
                    hist = [{"loss": float("nan")}]
                else:
                    print(f"  [train] MFXR {M.BACKBONE_SHORT[bb]} s{seed}",
                          flush=True)
                    model, hist = P.train(tr, bb, seed, device, P.load_standard,
                                          epochs=cfg.epochs,
                                          batch_size=cfg.batch_size, lr=cfg.lr,
                                          num_workers=cfg.num_workers,
                                          img_size=cfg.img_size)
                    torch.save({"state_dict": model.state_dict(),
                                "backbone": bb, "source": "MFXR", "seed": seed,
                                "prep": "standard",
                                "label_col": "label_abnormal",
                                "source_prev": float(tr.label.mean()),
                                "history": hist,
                                "provenance": M.provenance(
                                    {"script": "mfxr_external.py v1"})}, cp)
                pr = P.predict(model, va, device, cfg.eval_batch_size,
                               cfg.num_workers, cfg.img_size, P.load_standard)
                a = score(pr["logit"], va["label"].to_numpy(),
                          va.study.to_numpy())["auc_study"]
                ref_auc[(bb, seed)] = a
                ref_loss[(bb, seed)] = float(hist[-1]["loss"])
                print(f"    reference AUC={a:.4f}  "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                del model
                torch.cuda.empty_cache()
        if cfg.stage == "reference":
            print("\n[mfxr] reference models done. Now run --stage decompose.")
            return 0

    # --------------------------------------------------- stage 2: decompose --
    missing = [(bb, s) for bb in cfg.backbones for s in cfg.seeds
               if (bb, s) not in ref_auc and not mfxr_ckpt(ck_root, bb, s).exists()]
    if missing:
        raise SystemExit(
            f"!! no MFXR reference for {missing[:3]}{'...' if len(missing)>3 else ''}\n"
            "   The encoder residual is reference - oracle; without the "
            "reference there is no residual.\n"
            "   Run:  python mfxr_external.py --stage reference")

    prov = M.provenance({
        "script": "mfxr_external.py v1", "source": SRC, "target": "MFXR",
        "c_grid": list(cfg.c_grid), "epochs": cfg.epochs, "lr": cfg.lr,
        "mfxr_train_prev": float(tr.label.mean()),
        "mfxr_valid_prev": float(va.label.mean()),
        "note": "MFXR as a second cross-dataset target. Reference trained on "
                "MFXR train with preprocessing_variants.train hyperparameters unchanged "
                "(not tuned for MFXR, for comparability with the MURA and GRAZ "
                "references). Case-control cohort: absolute AUCs are not "
                "screening-comparable; G, R and E are within-target "
                "differences and are."})
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    res = CsvAppender(out, prov)
    st = va.study.to_numpy()
    yva_true = va["label"].to_numpy()

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            t0 = time.time()
            if (bb, seed) not in ref_auc:
                cp = mfxr_ckpt(ck_root, bb, seed)
                rmodel, _ = load_source(cp, bb, device)
                pr = P.predict(rmodel, va, device, cfg.eval_batch_size,
                               cfg.num_workers, cfg.img_size, P.load_standard)
                ref_auc[(bb, seed)] = score(pr["logit"], yva_true,
                                            st)["auc_study"]
                ref_loss[(bb, seed)] = float("nan")
                del rmodel
                torch.cuda.empty_cache()

            smodel, _ = load_source(
                ckpt_path(Path(cfg.ckpt_root) / "source_models", SRC, bb, seed),
                bb, device)
            w_src, b_src = source_head(smodel)
            ftr, _, ytr = extract(smodel, tr, device, cfg, P.load_standard)
            fva, zva, yva = extract(smodel, va, device, cfg, P.load_standard)
            ftr = ftr.astype(np.float64)
            fva = fva.astype(np.float64)
            zs = score(zva, yva, st)["auc_study"]
            verify_source_head(fva, zva, yva, st, w_src, b_src, zs,
                               f"{bb}/s{seed}/MFXR")

            w, b, c_sel = fit_oracle(ftr, ytr, tr["patient_group"].to_numpy(),
                                     list(cfg.c_grid))
            orc = score(fva @ w + b, yva, st)["auc_study"]
            ref = ref_auc[(bb, seed)]
            gap = ref - zs
            res.write([{
                "source": SRC, "target": "MFXR", "backbone": bb, "seed": seed,
                "C": c_sel, "setting": "nested",
                "n_train_images": len(tr),
                "n_train_studies": int(tr.study.nunique()),
                "train_prev": float(tr.label.mean()),
                "n_eval_images": len(va),
                "n_eval_studies": int(va.study.nunique()),
                "eval_prev": float(va.label.mean()),
                "zeroshot": zs, "oracle_readout": orc, "reference": ref,
                "gap": gap, "readout_recoverable": orc - zs,
                "encoder_residual": ref - orc,
                "readout_frac": ((orc - zs) / gap if gap >= GAP_GATE else np.nan),
                "final_loss": ref_loss.get((bb, seed), float("nan")),
                "minutes": (time.time() - t0) / 60.0}])
            print(f"  [mfxr] {M.BACKBONE_SHORT[bb]:<12s} s{seed} "
                  f"zs={zs:.4f} orc={orc:.4f} ref={ref:.4f} | "
                  f"gap={gap:.4f} resid={ref-orc:.4f} | C={c_sel:g} | "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            del smodel
            torch.cuda.empty_cache()

    print(f"\nwrote {out}")
    print(f"Now run:  python mfxr_external.py --summarise {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
