#!/usr/bin/env python3
"""Transfer into FracAtlas hand and shoulder from every MURA source.

With a FracAtlas target held fixed, all seven MURA source anatomies are applied
to it, so anatomy match varies while dataset, target images and label stay the
same.

Output (under --out-dir):
    fracatlas_transfer.csv and per-cell .npz dumps

Usage:
    python fracatlas_transfer.py --gpu 0

Requires a GPU. Also imported for the FracAtlas pools.
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
from dataset_shift import (with_descriptors, projection_cell, save_dump,
                         load_index, load_source)

FRAC_TARGETS = ["XR_HAND", "XR_SHOULDER"]
MURA_SOURCES = list(mskdata.MURA_ANATOMIES)


def verify_readable(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Drop images that will not decode.

    NEVER set ImageFile.LOAD_TRUNCATED_IMAGES. FracAtlas's truncated files sit
    almost entirely in Non_fractured/, so silently decoding half an image would
    feed the model damaged NEGATIVES only -- a class-correlated artifact
    manufactured by the fix.
    """
    ok, bad = [], []
    for i, p in enumerate(frame.path.astype(str)):
        try:
            mskdata.load_gray_uint8(p)
            ok.append(i)
        except Exception:
            bad.append(i)
    if bad:
        b = frame.iloc[bad]
        print(f"  [truncated] dropped {len(bad)} unreadable images; "
              f"label distribution among them: "
              f"{b.label.value_counts().to_dict()}")
        print("  (class-correlated corruption is expected here -- report it)")
    return frame.iloc[ok].reset_index(drop=True)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-root", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--backbones", nargs="+",
                    default=["densenet121", "swin_tiny_patch4_window7_224"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(M.SEEDS))
    ap.add_argument("--label", default="label_fracture")
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    cfg = ap.parse_args()

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({"script": "fracatlas_transfer.py v1",
                         "note": "EVALUATION ONLY -- no FracAtlas ceiling exists",
                         "targets": FRAC_TARGETS, "sources": MURA_SOURCES,
                         "label": cfg.label})
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(cfg.out_dir) / "fracatlas_transfer.csv"
    res = CsvAppender(out, prov)
    index = load_index(cfg)
    ck_dir = Path(cfg.ckpt_root) / "source_models"

    frames, Xs = {}, {}
    for t in FRAC_TARGETS:
        f = M.frame_for(index, "fracatlas", t, "test", cfg.label).reset_index(drop=True)
        if f.empty:
            print(f"  !! no FracAtlas {t} rows; skipping")
            continue
        f = verify_readable(f)
        f, X = with_descriptors(f)
        if len(f) < cfg.min_n:
            print(f"  !! FracAtlas {t}: only {len(f)} usable images "
                  f"(min {cfg.min_n}); NOT RUNNABLE, skipping")
            continue
        frames[t], Xs[t] = f, X
        print(f"[frac] {t}: {len(f)} images, prevalence {f.label.mean():.4f}")
    if not frames:
        raise SystemExit("no runnable FracAtlas targets")

    print("\nNOTE: no in-domain ceiling is computed. FracAtlas has no patient\n"
          "column and carries a `multiscan` flag, so any in-domain split leaks.\n"
          "Absolute AUCs below are NOT comparable to MURA or GRAZ numbers.\n"
          "Only the WITHIN-TARGET contrast across source anatomies is valid.\n")

    for t in frames:
        f, X = frames[t], Xs[t]
        for src in MURA_SOURCES:
            for bb in cfg.backbones:
                rows = []
                for seed in cfg.seeds:
                    cp = ckpt_path(ck_dir, src, bb, seed)
                    if not cp.exists():
                        print(f"  !! missing {cp.name}")
                        continue
                    model, ck = load_source(cp, bb, device)
                    pr = P.predict(model, f, device, cfg.eval_batch_size,
                                   cfg.num_workers, cfg.img_size, P.load_standard)
                    z, y, st = pr["logit"], pr["label"], pr["study"]
                    cell, resid = projection_cell(
                        z, y, st, X, ("frac", bb, seed, src, t))
                    row = {"backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                           "seed": seed, "source": src, "target": t,
                           "matched": int(src == t), "label_col": cfg.label,
                           "n_images": len(f),
                           "n_studies": int(f.study.nunique()),
                           "target_prev": float(y.mean()),
                           "source_prev": float(ck.get("source_prev", np.nan))}
                    row.update(cell)
                    rows.append(row)
                    save_dump(Path(cfg.out_dir) / "dumps" /
                              f"frac__{t}__from_{src}__{bb}__seed{seed}.npz",
                              f, z, resid, y, st, X, row)
                    del model
                    torch.cuda.empty_cache()
                if rows:
                    res.write(rows)
                    m = np.mean([r["zs_auc_study"] for r in rows])
                    print(f"  [frac] {t:<12s} <- {src:<12s} "
                          f"{M.BACKBONE_SHORT[bb]:<12s} "
                          f"auc={m:.4f}{'   <-- MATCHED' if src == t else ''}",
                          flush=True)

    r = pd.read_csv(out, comment="#")
    print("\n" + "=" * 78)
    print("WITHIN-TARGET CONTRAST: does the anatomy-matched source win?")
    print("=" * 78)
    verdicts = {}
    for t in sorted(r.target.unique()):
        s = r[r.target == t]
        tab = s.groupby(["source", "backbone"]).zs_auc_study.mean().unstack()
        tab["mean"] = tab.mean(axis=1)
        print(f"\nTARGET {t}   (n={int(s.n_images.iloc[0])}, "
              f"prevalence={s.target_prev.iloc[0]:.3f})")
        print(tab.round(4).sort_values("mean", ascending=False).to_string())
        if t in tab.index:
            mt = float(tab.loc[t, "mean"])
            mis = float(tab.drop(index=t)["mean"].mean())
            rank = int((tab["mean"] > mt).sum()) + 1
            print(f"  matched ({t}) {mt:.4f}   mismatched mean {mis:.4f}   "
                  f"advantage {mt - mis:+.4f}   rank {rank}/{len(tab)}")
            verdicts[t] = (mt > mis) and rank <= 2
        else:
            print(f"  no MURA source checkpoint for {t}")
    if len(verdicts) == 2:
        v = list(verdicts.values())
        print("\n  >>> " + ("F1: anatomy match survives the dataset change."
                            if all(v) else
                            "F2: no matched advantage; anatomy match does NOT "
                            "survive the dataset change."
                            if not any(v) else
                            "F3: SPLIT verdict. Report as split. Do not pick "
                            "the target that agrees with the preferred answer."))
    print("\n" + "=" * 78)
    print("S1: DOES XR_SHOULDER'S DIFFICULTY REPLICATE OFF MURA?")
    print("=" * 78)
    print(r.groupby("target").zs_auc_study.agg(["mean", "min", "max"])
           .round(4).to_string())
    print("  Stage A: XR_SHOULDER was the hardest MURA target from all six")
    print("  sources (gap 0.191-0.333). If FA-shoulder is far below FA-hand")
    print("  from EVERY source, that difficulty is a property of the anatomy.")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
