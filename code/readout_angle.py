#!/usr/bin/env python3
"""Measure the angle between the source readout and the best frozen readout.

For each source -> target cell, fits the best linear readout on the target's
frozen features (the oracle readout) and measures its angle to the transferred
source readout, then relates the angle to how much AUC the refit recovers. Also
tests a label-free conditional readout.

Output (under --out-dir):
    readout_angle_<source>.csv

Usage:
    python readout_angle.py --source XR_WRIST --gpu 0

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

import mskdata
import mskmodel as M
import preprocessing_variants as P
from train_source_models import ckpt_path, CsvAppender
from dataset_shift import graz_frame, load_index, load_source
from readout_probe import extract, ALL_ANATOMIES

GAP_GATE = 0.10
PREREG_ANGLE_RHO = 0.5
PREREG_LOTO_FRAC = 0.50
PREREG_LOTO_CELLS = 8


def fit_direction(feats: np.ndarray, y: np.ndarray):
    """
    Oracle readout: logistic head on frozen features, ALL available target data.

    Returns the L2-normalised RAW-feature-space direction. The scaler is undone
    (`w / scale_`) because the source head sees raw trunk output; comparing a
    standardised-space vector against a raw-space one would produce a cosine
    with no meaning.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2:
        return None, None
    sc = StandardScaler().fit(feats)
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(feats), y)
    w_eff = lr.coef_.ravel() / np.maximum(sc.scale_, 1e-12)
    n = np.linalg.norm(w_eff)
    return (w_eff / n if n > 0 else None), (sc, lr)


def cos_sigma(a, b, F) -> float:
    """
    Covariance-weighted cosine: alignment as the data sees it.

    Why the ambient cosine is the wrong object. The score a model produces is
    `F @ w`, so the only part of `w` that matters is its component along
    directions the features actually vary in. Ambient cosine weights all 1024
    dimensions equally, and in 1024-d ANY two vectors are ~90 deg apart
    (random pairs: 90.00 +- 1.80). The statistic therefore has almost no
    dynamic range and cannot distinguish "misaimed head" from "random head".

    Under the feature covariance Sigma the right quantity is

        cos_Sigma(a, b) = a'Sigma b / sqrt(a'Sigma a * b'Sigma b)

    which for centred features is exactly the Pearson correlation between the
    two score vectors. Because Sigma is strongly anisotropic, this has full
    dynamic range: two ambient-orthogonal directions can still produce highly
    correlated scores.
    """
    Fc = F - F.mean(axis=0, keepdims=True)
    sa, sb = Fc @ a, Fc @ b
    na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
    return float(np.clip((sa @ sb) / max(na * nb, 1e-12), -1.0, 1.0))


def subspace_angle(a, b, F, var: float = 0.95) -> tuple:
    """Ambient angle after projecting both onto the top-k PCA subspace of F."""
    from sklearn.decomposition import PCA
    k = min(F.shape[0] - 1, F.shape[1])
    pca = PCA(n_components=k).fit(F)
    kk = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var) + 1)
    V = pca.components_[:kk]
    pa, pb = V @ a, V @ b
    na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
    c = float(np.clip((pa @ pb) / max(na * nb, 1e-12), -1.0, 1.0))
    return c, kk


def cosang(a, b) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.clip(a @ b, -1.0, 1.0))


def main() -> None:
    import torch
    from scipy import stats
    from shortcut_reliance import score
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-root", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--source", default="XR_WRIST", choices=ALL_ANATOMIES)
    ap.add_argument("--backbones", nargs="+",
                    default=["densenet121", "swin_tiny_patch4_window7_224"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(M.SEEDS))
    ap.add_argument("--skip-graz", action="store_true")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", default=None,
                    help="explicit output path; default readout_angle_{SRC}.csv")
    ap.add_argument("--gpu", type=int, default=0)
    cfg = ap.parse_args()

    SRC = cfg.source
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({"script": "readout_angle.py v1", "source": SRC,
                         "note": "frozen-encoder oracle readout + head geometry"})
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    # v2: cos_sigma / subspace columns did not exist in the v1 schema, and
    # CsvAppender correctly refused to widen a file mid-run. New file, not a
    # silently widened old one.
    out = Path(cfg.out) if cfg.out else (
        Path(cfg.out_dir) / f"readout_angle_{SRC}.csv")
    res = CsvAppender(out, prov)
    index = load_index(cfg)
    ck_dir = Path(cfg.ckpt_root) / "source_models"

    # ---- Stage A ceilings, for the gap decomposition ----------------------
    a = pd.read_csv(Path(M.DEFAULT_RESULTS) / "source_models" /
                    "anatomy_transfer_arms_merged.csv", comment="#", low_memory=False)
    ceil = (a[a.condition == "ceiling"]
            .groupby(["target", "backbone"]).auc_study.mean())
    # GRAZ initial-only ceiling, PER BACKBONE, read from the trained-subset run.
    # Hardcoding 0.9571 (the DenseNet+Swin mean) would silently apply a
    # two-backbone average to a third backbone.
    GRAZ_CEILING = {}
    gpath = Path(cfg.out_dir) / "graz_initial_only_ceiling.csv"
    if gpath.exists():
        gc = pd.read_csv(gpath, comment="#")
        gc = gc[gc.score_label == "label_abnormal"]
        GRAZ_CEILING = gc.groupby("backbone").zs_auc_study.mean().to_dict()
        print(f"[graz-ceiling] per backbone: "
              f"{ {k: round(v,4) for k,v in GRAZ_CEILING.items()} }")
    else:
        print(f"[graz-ceiling] {gpath.name} not found")

    # ---- pools -------------------------------------------------------------
    pools = {}
    for t in [x for x in ALL_ANATOMIES if x != SRC]:
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

    rows, heads = [], {}
    for bb in cfg.backbones:
        for seed in cfg.seeds:
            block_start = len(rows)
            model, _ = load_source(ckpt_path(ck_dir, SRC, bb, seed), bb, device)
            w_src = model.head.weight.detach().float().cpu().numpy().ravel()
            w_src = w_src / np.linalg.norm(w_src)
            t0 = time.time()
            for t, (tr, va) in pools.items():
                ftr, _, ytr = extract(model, tr, device, cfg, P.load_standard)
                fva, zva, yva = extract(model, va, device, cfg, P.load_standard)
                st = va.study.to_numpy()
                zs = score(zva, yva, st)["auc_study"]
                w_or, _ = fit_direction(ftr, ytr)
                if w_or is None:
                    continue
                orc = score(fva @ w_or, yva, st)["auc_study"]
                if t.startswith("GRAZ"):
                    c = GRAZ_CEILING.get(bb, np.nan)
                    if not np.isfinite(c):
                        print(f"  !! no GRAZ ceiling for {bb}; run "
                              f"stratify_scores.py train --subset initial_only "
                              f"--backbones {bb}. Skipping the GRAZ cell.")
                        continue
                else:
                    c = ceil.get((t, bb), np.nan)
                gap = c - zs
                cs = cosang(w_src, w_or)
                cs_sig = cos_sigma(w_src, w_or, fva)
                cs_sub, k_sub = subspace_angle(w_src, w_or, fva)
                row = {"source": SRC, "target": t, "backbone": bb, "seed": seed,
                       "n_train_images": len(tr),
                       "n_train_studies": int(tr.study.nunique()),
                       "zeroshot": zs, "oracle_readout": orc, "ceiling": c,
                       "gap": gap,
                       "readout_recoverable": orc - zs,
                       "encoder_residual": c - orc,
                       "readout_frac": (orc - zs) / gap if gap >= GAP_GATE else np.nan,
                       "cos_src_oracle": cs,
                       "angle_deg": float(np.degrees(np.arccos(cs))),
                       "cos_sigma": cs_sig,
                       "angle_sigma_deg": float(np.degrees(np.arccos(cs_sig))),
                       "cos_subspace": cs_sub,
                       "angle_subspace_deg": float(np.degrees(np.arccos(cs_sub))),
                       "k_subspace": k_sub,
                       "feat_dim": int(w_src.shape[0]),
                       "head_shape": str(tuple(
                           model.head.weight.shape)),
                       "minutes": (time.time() - t0) / 60.0}
                rows.append(row)
                heads[(bb, seed, t)] = (w_or, fva, yva, st, zs, orc, gap)
                print(f"  [{bb[:8]} s{seed}] {t:<14s} zs={zs:.4f} "
                      f"oracle={orc:.4f} ceil={c:.4f}  "
                      f"readout={row['readout_frac'] if np.isfinite(row['readout_frac']) else float('nan'):.3f}  "
                      f"amb={row['angle_deg']:.1f} "
                      f"SIG={row['angle_sigma_deg']:.1f} "
                      f"sub={row['angle_subspace_deg']:.1f} (k={k_sub})",
                      flush=True)

            # ---- Q3: leave-one-target-out, NO held-out labels -------------
            ts = [t for (b2, s2, t) in heads if b2 == bb and s2 == seed]
            for t in ts:
                others = [heads[(bb, seed, o)][0] for o in ts if o != t]
                if len(others) < 2:
                    continue
                w_avg = np.mean(others, axis=0)
                w_avg = w_avg / np.linalg.norm(w_avg)
                _, fva, yva, st, zs, orc, gap = heads[(bb, seed, t)]
                loto = score(fva @ w_avg, yva, st)["auc_study"]
                rows.append({"source": SRC, "target": t, "backbone": bb,
                             "seed": seed, "analysis": "LOTO",
                             "zeroshot": zs, "loto_auc": loto,
                             "oracle_readout": orc, "gap": gap,
                             "loto_gain": loto - zs,
                             "loto_frac_of_oracle":
                                 (loto - zs) / (orc - zs) if (orc - zs) > 0 else np.nan,
                             "cos_avg_oracle": cosang(w_avg, heads[(bb, seed, t)][0])})
            # flush this (backbone, seed) block before touching the next model
            if len(rows) > block_start:
                res.write(rows[block_start:])
                print(f"  [flush] wrote {len(rows) - block_start} rows "
                      f"({bb}, seed {seed})", flush=True)
            del model
            torch.cuda.empty_cache()

    # ================================ analysis ============================
    d = pd.DataFrame(rows)
    main_d = d[d.get("analysis").isna()] if "analysis" in d else d
    g = main_d[main_d.gap >= GAP_GATE]

    print("\n" + "=" * 78)
    print("Q2. GAP DECOMPOSITION  (frozen-encoder ORACLE readout)")
    print("=" * 78)
    t2 = (g.groupby("target")[["zeroshot", "oracle_readout", "ceiling",
                               "readout_recoverable", "encoder_residual",
                               "readout_frac", "angle_deg"]].mean())
    print(t2.round(4).sort_values("readout_frac", ascending=False).to_string())
    xr = [t for t in t2.index if t.startswith("XR_")]
    print(f"\n  cross-anatomy mean readout fraction : {t2.loc[xr,'readout_frac'].mean():.4f}")
    if "GRAZ_initial" in t2.index:
        print(f"  GRAZ initial-only                   : "
              f"{t2.loc['GRAZ_initial','readout_frac']:.4f}")
    print(f"\n  Compare the n=500 BUDGET probe (§2.19b): cross-anatomy 0.545, "
          f"GRAZ 0.202.")
    print("  If oracle >> budget, more labels help. If oracle ~= budget, 500")
    print("  studies already saturates the readout.")

    print("\n" + "=" * 78)
    print("Q1. DOES HEAD MISALIGNMENT PREDICT RECOVERABILITY?")
    print("=" * 78)
    print(f"  head shape (must be (1, d) for a single-logit BCE net): "
          f"{g.head_shape.iloc[0]}  feat_dim {int(g.feat_dim.iloc[0])}")
    print(f"  PCA subspace retained for 95% variance: "
          f"{g.k_subspace.mean():.0f} of {int(g.feat_dim.iloc[0])} dims\n")
    print(f"  {'metric':<22}{'mean':>9}{'sd':>8}{'rho vs readout_frac':>22}{'p':>10}")
    for col, lab in (("angle_deg", "ambient angle"),
                     ("angle_sigma_deg", "Sigma-weighted angle"),
                     ("angle_subspace_deg", "subspace angle")):
        r_, p_ = stats.spearmanr(g[col], g.readout_frac)
        print(f"  {lab:<22}{g[col].mean():>9.2f}{g[col].std():>8.2f}"
              f"{r_:>22.3f}{p_:>10.4g}")
    print("\n  The AMBIENT row is expected to be flat: random vectors in this many")
    print("  dimensions sit at 90.0 +- 1.8 deg, so it has no dynamic range. If the")
    print("  Sigma-weighted row also has a small sd, misalignment genuinely does")
    print("  not vary across cells. If its sd is large, use IT as the metric.\n")
    rho, p = stats.spearmanr(g.angle_sigma_deg, g.readout_frac)
    rho2, p2 = stats.spearmanr(g.angle_sigma_deg, g.readout_recoverable)
    print(f"  PRIMARY -- Sigma-weighted:")
    print(f"    Spearman(angle_sigma, readout_frac)        = {rho:+.3f}  p={p:.4g}  n={len(g)}")
    print(f"    Spearman(angle_sigma, readout_recoverable) = {rho2:+.3f}  p={p2:.4g}")
    print(f"    range: {g.angle_sigma_deg.min():.1f}-{g.angle_sigma_deg.max():.1f} deg")
    if abs(rho) >= PREREG_ANGLE_RHO and p < 0.05:
        print("\n  >>> G1: MECHANISM CONFIRMED BY DIRECT MEASUREMENT. The source")
        print("      head's direction predicts how much a re-aimed head buys.")
        print("      This is the headline figure for the chapter.")
    else:
        print("\n  >>> G1n: the angle does NOT predict recovery. The readout story")
        print("      is weaker than the AUC evidence suggests. Report it and do")
        print("      not go looking for a fourth statistic that agrees.")

    if "analysis" in d and (d.analysis == "LOTO").any():
        L = d[d.analysis == "LOTO"]
        print("\n" + "=" * 78)
        print("Q3. THE OBJECTIVE 3 GATE -- label-free conditional readout")
        print("=" * 78)
        t3 = L.groupby("target")[["zeroshot", "loto_auc", "oracle_readout",
                                  "loto_gain", "loto_frac_of_oracle",
                                  "cos_avg_oracle"]].mean()
        print(t3.round(4).sort_values("loto_frac_of_oracle",
                                      ascending=False).to_string())
        ok = int((L.loto_frac_of_oracle >= PREREG_LOTO_FRAC).sum())
        print(f"\n  cells reaching >= {PREREG_LOTO_FRAC:.0%} of the oracle gain: "
              f"{ok} / {len(L)}")
        if ok >= PREREG_LOTO_CELLS:
            print("\n  >>> G3: A LABEL-FREE CONDITIONAL READOUT IS VIABLE.")
            print("      Objective 3 gets a NEW METHOD with a mechanism behind")
            print("      it: condition the HEAD, not the trunk. This is what the")
            print("      gating ablation should have been.")
        else:
            print("\n  >>> G3n: target head directions are not predictable from")
            print("      other targets without labels. The finding is DIAGNOSTIC.")
            print("      Objective 3 becomes 'readout re-alignment vs encoder")
            print("      adaptation at matched label budgets' -- still a chapter,")
            print("      and this is the more likely outcome, not a failure.")

    print("\n" + "=" * 78)
    print("REPORTED REGARDLESS: how structured is the set of head directions?")
    print("=" * 78)
    for bb in cfg.backbones:
        W = np.array([v[0] for (b2, s2, t), v in heads.items() if b2 == bb])
        if len(W) < 3:
            continue
        from sklearn.decomposition import PCA
        pca = PCA().fit(W)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k95 = int(np.searchsorted(cum, 0.95) + 1)
        print(f"  {bb:<32s} {len(W)} heads, dim {W.shape[1]}, "
              f"{k95} components for 95% variance")
        print(f"     mean pairwise cosine between target heads: "
              f"{np.mean([cosang(W[i], W[j]) for i in range(len(W)) for j in range(i+1, len(W))]):.3f}")
    print("\n  Few components + high pairwise cosine = structured, and G3 has a")
    print("  chance. Many components + near-zero cosine = unstructured, and no")
    print("  fitting procedure will make G3 work.")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
