#!/usr/bin/env python3
"""How much of a frozen encoder's decision is explained by low-level appearance,
for every source -> target pair of MURA anatomies.

For each pair, regresses the encoder's target scores on the five appearance
descriptors of measure_appearance.py and measures whether removing that
component helps or hurts transfer, against a permutation null.

Outputs (under --out-dir):
    shortcut_reliance_matrix.csv, shortcut_reliance_report.csv

Usage:
    python shortcut_reliance.py --backbones densenet121 --seeds 42 123 456 789 2024 --gpu 0
    python shortcut_reliance.py --report-only

Requires a GPU. Also imported by other stages for ``score``.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskdata
import mskmodel as M
from train_source_models import ckpt_path, CsvAppender
from layerwise_unfreezing import predict
from measure_appearance import measure_one, FEATURES

ANATOMIES = list(mskdata.MURA_ANATOMIES)
N_PERM = 200
# Raised from 20. The permutation mean is what the `excess > 0.01` gate is
# applied against, so a thin null propagates straight into the "cells clearing
# the permutation null" and "20 of 84" counts. The shoulder conclusion is
# threshold-free (56/60 positive) and does not depend on this, but the cell
# counts do. Cost is negligible: each draw is a five-predictor OLS, not a
# forward pass.
CELL_KEYS = ["backbone", "seed", "source", "target"]

# v1 reference, DenseNet121 + Swin-T, 5 seeds, source = XR_WRIST.
V1_WRIST_DELTA = {
    ("densenet121", "XR_SHOULDER"): +0.0570, ("densenet121", "XR_HUMERUS"): -0.0254,
    ("densenet121", "XR_HAND"): -0.0447, ("densenet121", "XR_ELBOW"): -0.0703,
    ("densenet121", "XR_FINGER"): -0.1104, ("densenet121", "XR_FOREARM"): -0.0149,
}


def geometry_for(frame: pd.DataFrame) -> pd.DataFrame:
    """Appearance features on exactly these images. No sampling, no merge loss."""
    rows = []
    for p in frame.path:
        try:
            rows.append(measure_one(p))
        except Exception as e:
            rows.append({k: np.nan for k in FEATURES})
            print(f"  !! {p}: {e}")
    return pd.DataFrame(rows, index=frame.index)[FEATURES]


def project_out(z: np.ndarray, X: np.ndarray):
    """Residual of z after removing its least-squares projection onto X."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    fit = LinearRegression().fit(Xs, z)
    return z - fit.predict(Xs), float(fit.score(Xs, z))


def score(logit, label, study) -> Dict[str, float]:
    """
    Mirrors train_source_models.eval_block: sigmoid FIRST, then aggregate by study.
    aggregate_studies returns (study_ids, mean_prob, max_label); safe_auc takes
    (label, score).
    """
    p = M.sigmoid(logit)
    _, sp, sy = M.aggregate_studies(p, label, study)
    return {"auc_image": M.safe_auc(label, p), "auc_study": M.safe_auc(sy, sp)}


def run(cfg, prov) -> None:
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir) / "shortcut_reliance_matrix.csv"
    res = CsvAppender(out, prov)

    done = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, comment="#")
        done = set(map(tuple, prev[CELL_KEYS].values.tolist()))
        print(f"[resume] {len(done)} cells complete")

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)

    # features are a property of the images, not of any model -- compute once
    geo, frames = {}, {}
    for t in ANATOMIES:
        fr = M.frame_for(index, "mura", t, "valid", cfg.label_col
                         ).reset_index(drop=True)
        g = geometry_for(fr)
        keep = g.notna().all(axis=1)
        frames[t] = fr[keep].reset_index(drop=True)
        geo[t] = g[keep].reset_index(drop=True).to_numpy(dtype=float)
        print(f"[geometry] {t:<12s} {int(keep.sum())}/{len(fr)} images "
              f"({frames[t].study.nunique()} studies)", flush=True)

    n_total = len(cfg.backbones) * len(cfg.seeds) * len(cfg.sources) * 6
    n = 0
    for bb in cfg.backbones:
        for seed in cfg.seeds:
            for src in cfg.sources:
                targets = [t for t in ANATOMIES if t != src]
                if all((bb, seed, src, t) in done for t in targets):
                    continue
                cp = Path(ckpt_path(cfg.source_ckpt_dir, src, bb, seed))
                if not cp.exists():
                    raise FileNotFoundError(f"missing source checkpoint {cp}")
                ck = torch.load(cp, map_location="cpu", weights_only=False)
                model = M.build_model(bb)
                model.load_state_dict(ck["state_dict"])
                model = model.to(device)

                rows = []
                for t in targets:
                    n += 1
                    if (bb, seed, src, t) in done:
                        continue
                    fr, X = frames[t], geo[t]
                    pr = predict(model, fr, device, cfg.eval_batch_size,
                                 cfg.num_workers, cfg.img_size, "standard")
                    z, y, st = pr["logit"], pr["label"], pr["study"]

                    base = score(z, y, st)
                    resid, r2 = project_out(z, X)
                    red = score(resid, y, st)

                    # Per-cell RNG derived from the cell identity, so each
                    # cell's null is reproducible on its own rather than
                    # depending on the order cells happened to be visited.
                    #
                    # NOT hash(): Python salts string hashing per process
                    # unless PYTHONHASHSEED is set, so hash() on the same tuple
                    # returns a different value in every run. Using it here
                    # would make the null LESS reproducible than the single
                    # RandomState(0) it replaced. crc32 is a fixed function of
                    # the bytes and is stable across processes and machines.
                    cell_key = f"{bb}|{int(seed)}|{src}|{t}".encode()
                    cell_rng = np.random.RandomState(zlib.crc32(cell_key))
                    perm_std, perm_r2 = [], []
                    for _ in range(N_PERM):
                        rp, r2p = project_out(
                            z, X[cell_rng.permutation(len(X))])
                        perm_std.append(score(rp, y, st)["auc_study"])
                        perm_r2.append(r2p)

                    rows.append({
                        "backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                        "seed": seed, "source": src, "target": t,
                        "n_images": len(fr),
                        "n_studies": int(fr.study.nunique()),
                        "zs_auc_image": base["auc_image"],
                        "zs_auc_study": base["auc_study"], "appearance_r2": r2,
                        "resid_auc_image": red["auc_image"],
                        "resid_auc_study": red["auc_study"],
                        "delta_image": red["auc_image"] - base["auc_image"],
                        "delta_study": red["auc_study"] - base["auc_study"],
                        "perm_r2_mean": float(np.mean(perm_r2)),
                        "perm_delta_study_mean": float(np.mean(perm_std)
                                                       - base["auc_study"]),
                        "perm_delta_study_sd": float(np.std(perm_std, ddof=1)),
                    })
                    print(f"  [{n}/{n_total}] {M.BACKBONE_SHORT[bb]} s{seed} "
                          f"{src[3:]:>8s}->{t[3:]:<9s} zs={base['auc_study']:.4f} "
                          f"R2={r2:.3f} delta={rows[-1]['delta_study']:+.4f}",
                          flush=True)
                if rows:
                    res.write(rows)
                del model
                torch.cuda.empty_cache()
    print("\n[run] complete", flush=True)


def _matrix(d, value, agg="mean"):
    return d.pivot_table(index="source", columns="target", values=value,
                         aggfunc=agg).reindex(index=ANATOMIES, columns=ANATOMIES)


def report(cfg) -> None:
    from scipy import stats
    p = Path(cfg.out_dir) / "shortcut_reliance_matrix.csv"
    if not p.exists():
        print(f"no results at {p}")
        return
    d = pd.read_csv(p, comment="#")
    d["excess"] = d.delta_study - d.perm_delta_study_mean

    for bb, g in d.groupby("backbone"):
        short = M.BACKBONE_SHORT.get(bb, bb)
        print("\n" + "=" * 78)
        print(f"{short}  --  delta AUC from removing the geometry component")
        print("rows = source, cols = target.  POSITIVE = removing geometry HELPS")
        print("=" * 78)
        print(_matrix(g, "delta_study").round(3).to_string())

        print(f"\n{short}  --  R^2, share of the encoder's decision explained "
              f"by geometry")
        print(_matrix(g, "appearance_r2").round(3).to_string())

    print("\n" + "=" * 78)
    print("IS THE HARMFUL COMPONENT A TARGET PROPERTY OR A PAIR PROPERTY?")
    print("=" * 78)
    print("per TARGET, pooled over all sources:")
    t = (d.groupby("target")
           .agg(delta=("delta_study", "mean"), sd=("delta_study", "std"),
                excess=("excess", "mean"), r2=("appearance_r2", "mean"),
                zs=("zs_auc_study", "mean"), n=("delta_study", "size"),
                n_pos=("delta_study", lambda s: int((s > 0).sum())))
           .sort_values("delta", ascending=False).round(4))
    t["frac_pos"] = (t.n_pos / t.n).round(3)
    print(t.to_string())
    print("\nper SOURCE, pooled over all targets:")
    print(d.groupby("source")
           .agg(delta=("delta_study", "mean"), excess=("excess", "mean"),
                r2=("appearance_r2", "mean")).sort_values("delta", ascending=False)
           .round(4).to_string())

    print("\n" + "=" * 78)
    print("CELLS WITH A HARMFUL GEOMETRY COMPONENT")
    print("(delta > 0 AND excess > 0.01, i.e. clear of the permutation null)")
    print("=" * 78)
    cell = (d.groupby(["backbone", "source", "target"])
              .agg(delta=("delta_study", "mean"), sd=("delta_study", "std"),
                   excess=("excess", "mean"), r2=("appearance_r2", "mean"),
                   zs=("zs_auc_study", "mean"), n=("seed", "nunique")))
    harm = cell[(cell.delta > 0) & (cell.excess > 0.01)].copy()
    harm["t"] = (harm.delta / (harm.sd / np.sqrt(harm.n))).round(2)
    if len(harm):
        print(harm.sort_values("delta", ascending=False).round(4).to_string())
        print(f"\n  {len(harm)} of {len(cell)} (backbone, source, target) cells")
        by_t = harm.reset_index().target.value_counts()
        print("  harmful cells by TARGET:")
        for k, v in by_t.items():
            print(f"    {k:<12s} {v:>2d}")
    else:
        print("  none")

    print("\n" + "=" * 78)
    print("DOES ZERO-SHOT COLLAPSE PREDICT A HARMFUL COMPONENT?")
    print("=" * 78)
    c = cell.reset_index()
    r, pv = stats.spearmanr(c.zs, c.delta)
    print(f"  Spearman(zero-shot AUC, delta) = {r:+.3f}  p={pv:.4g}  n={len(c)}")
    print("  Strongly negative means the geometry component turns harmful exactly")
    print("  where transfer has collapsed -- which would make it diagnosable")
    print("  label-free, and therefore actionable at deployment.")

    if "XR_WRIST" in set(d.source):
        print("\n" + "=" * 78)
        print("CONSISTENCY WITH v1 (source = XR_WRIST, densenet121)")
        print("=" * 78)
        w = (d[(d.source == "XR_WRIST") & (d.backbone == "densenet121")]
             .groupby("target").delta_study.mean())
        for tgt, v1 in sorted(V1_WRIST_DELTA.items(), key=lambda kv: kv[0][1]):
            if tgt[1] in w.index:
                print(f"  {tgt[1]:<12s} v1 {v1:+.4f}   v2 {w[tgt[1]]:+.4f}   "
                      f"diff {w[tgt[1]] - v1:+.4f}")
        print("  These should agree to ~0.001. Larger drift means something changed.")

    d.to_csv(Path(cfg.out_dir) / "shortcut_reliance_report.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--source-ckpt-dir",
                    default=str(Path(M.DEFAULT_CKPT) / "source_models"))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "layerwise"))
    ap.add_argument("--sources", nargs="+", default=ANATOMIES, choices=ANATOMIES)
    ap.add_argument("--backbones", nargs="+", default=["densenet121"],
                    choices=list(M.BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    ap.add_argument("--label-col", default="label_abnormal")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    cfg = ap.parse_args()

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    if cfg.report_only:
        report(cfg)
        return

    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({"script": "shortcut_reliance.py v2",
                         "sources": list(cfg.sources),
                         "backbones": list(cfg.backbones),
                         "seeds": list(cfg.seeds), "features": FEATURES,
                         "n_perm": N_PERM})
    run(cfg, prov)
    report(cfg)


if __name__ == "__main__":
    main()
