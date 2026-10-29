#!/usr/bin/env python3
"""Derive the Stage A tables from the outputs of train_source_models.py.

Merges the per-shard CSVs, checks that every expected cell is present (naming
any that are missing) and writes the merged arm table, the probe-budget curve
and the decomposition of the transfer gap.

Outputs (under --out, default: --results):
    anatomy_transfer_arms_merged.csv, anatomy_probe_curve.csv,
    anatomy_gap_decomposition.csv, anatomy_missing_cells.csv

Usage:
    python source_model_tables.py --results ~/diagnosis/results/source_models

CPU only.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ANATOMIES = ("XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND",
             "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST")
PRIMARY = "auc_study"          # MURA labels are per study


# ================================================================== io ======

def read_shards(results: Path, name: str) -> pd.DataFrame:
    parts = sorted(glob.glob(str(results / "*" / name))) or \
            sorted(glob.glob(str(results / name)))
    if not parts:
        raise SystemExit(f"no {name} found under {results}")
    frames = []
    for p in parts:
        d = pd.read_csv(p, comment="#", low_memory=False)
        d["_shard"] = Path(p).parent.name
        frames.append(d)
        print(f"  {Path(p).parent.name}/{name}: {len(d)} rows")
    return pd.concat(frames, ignore_index=True)


def write_csv(df: pd.DataFrame, path: Path, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# source_model_tables.py  {pd.Timestamp.now('UTC').isoformat()}\n")
        if note:
            fh.write(f"# {note}\n")
        df.to_csv(fh, index=False)


# ======================================================== completeness ======

def check_completeness(arms: pd.DataFrame, out: Path) -> bool:
    backbones = sorted(arms.backbone.unique())
    seeds = sorted(arms.seed.unique())
    sources = sorted(arms.source.unique())
    have = set(zip(arms.source, arms.target, arms.backbone, arms.seed,
                   arms.condition))

    missing: List[Dict] = []
    for s, bb, sd in itertools.product(sources, backbones, seeds):
        for t in ANATOMIES:
            cond0 = "ceiling" if s == t else "zeroshot"
            if (s, t, bb, sd, cond0) not in have:
                missing.append({"source": s, "target": t, "backbone": bb,
                                "seed": sd, "condition": cond0})
            if s != t and (s, t, bb, sd, "adabn") not in have:
                missing.append({"source": s, "target": t, "backbone": bb,
                                "seed": sd, "condition": "adabn"})

    print(f"\n  sources {len(sources)} | backbones {len(backbones)} | "
          f"seeds {len(seeds)}")
    n_exp = len(sources) * len(backbones) * len(seeds) * (len(ANATOMIES) + 6)
    print(f"  arm rows: {len(arms)} present, {n_exp} expected, "
          f"{len(missing)} missing")
    if missing:
        m = pd.DataFrame(missing)
        write_csv(m, out / "anatomy_missing_cells.csv")
        print("\n  !! MISSING CELLS (first 20) -- tables below are incomplete:")
        print(m.head(20).to_string(index=False))
        print(f"  full list -> {out/'anatomy_missing_cells.csv'}")
        return False
    print("  complete.")
    return True


# ======================================================= gap decomposition ==

def decompose(arms: pd.DataFrame) -> pd.DataFrame:
    """
    ceiling  = model trained on T, evaluated on T's valid  (source == target)
    zeroshot = model trained on S, evaluated on T's valid
    """
    ceil = (arms[arms.condition == "ceiling"]
            .rename(columns={"target": "tgt"})
            .set_index(["tgt", "backbone", "seed"])
            [[PRIMARY, "auc_image", "bacc_50"]]
            .rename(columns={PRIMARY: "ceiling_auc_study",
                             "auc_image": "ceiling_auc_image",
                             "bacc_50": "ceiling_bacc_50"}))

    zs = arms[arms.condition == "zeroshot"].copy()
    # zeroshot rows carry NaN placeholders for these; the real values come from
    # the adabn rows on the join, so drop the placeholders to avoid a collision.
    zs = zs.drop(columns=[c for c in ("n_bn_modules", "adabn_batches",
                                      "adabn_mean_drift") if c in zs.columns])
    ab = (arms[arms.condition == "adabn"]
          .set_index(["source", "target", "backbone", "seed"])
          [[PRIMARY, "auc_image", "bacc_50", "n_bn_modules",
            "adabn_batches", "adabn_mean_drift"]]
          .rename(columns={PRIMARY: "adabn_auc_study",
                           "auc_image": "adabn_auc_image",
                           "bacc_50": "adabn_bacc_50"}))

    d = zs.join(ceil, on=["target", "backbone", "seed"])
    d = d.join(ab, on=["source", "target", "backbone", "seed"])

    d["gap_auc_study"] = d["ceiling_auc_study"] - d[PRIMARY]
    d["gap_auc_image"] = d["ceiling_auc_image"] - d["auc_image"]

    # only these two can move AUC at all
    d["adabn_delta_auc"] = d["adabn_auc_study"] - d[PRIMARY]

    # thresholds move balanced accuracy only
    d["prior_delta_bacc"] = d["bacc_prior"] - d["bacc_50"]
    d["oracle_delta_bacc"] = d["bacc_oracle"] - d["bacc_50"]
    d["gap_bacc"] = d["ceiling_bacc_50"] - d["bacc_50"]

    d = d.rename(columns={PRIMARY: "zeroshot_auc_study",
                          "auc_image": "zeroshot_auc_image",
                          "bacc_50": "zeroshot_bacc_50"})
    return d


def frac(num: pd.Series, den: pd.Series, min_gap: float = 0.02) -> pd.Series:
    """Recovery fraction, undefined where the gap is too small to be meaningful."""
    out = num / den.replace(0, np.nan)
    return out.where(den > min_gap)


# =========================================================== probe curve ====

def probe_curve(probes: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    cell = ["source", "target", "backbone", "seed"]
    p = (probes.groupby(cell + ["n_studies"], as_index=False)
         .agg(probe_auc_study=("auc_study", "mean"),
              probe_auc_study_sd=("auc_study", "std"),
              probe_auc_image=("auc_image", "mean"),
              n_reps=("rep", "count"),
              n_probe_images=("n_probe_images", "mean"),
              pool_dropped_leak=("pool_studies_dropped_leak", "mean")))

    keys = dec.set_index(cell)[["zeroshot_auc_study", "ceiling_auc_study",
                                "gap_auc_study", "norm_family"]]
    p = p.join(keys, on=cell)
    p["probe_delta_auc"] = p["probe_auc_study"] - p["zeroshot_auc_study"]
    p["recovery_frac"] = frac(p["probe_delta_auc"], p["gap_auc_study"])
    p["beats_ceiling"] = (p["probe_auc_study"] >= p["ceiling_auc_study"]).astype(int)
    return p


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


# ================================================================== main ====

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(Path.home() / "diagnosis/results/source_models"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else res

    print("=" * 78)
    print("STAGE A ANALYSIS")
    print("=" * 78)
    print("\n--- reading shards ---")
    arms = read_shards(res, "anatomy_transfer_arms.csv")
    probes = read_shards(res, "anatomy_transfer_probes.csv")
    try:
        rec = read_shards(res, "adabn_receipts.csv")
    except SystemExit:
        rec = pd.DataFrame()

    n0 = len(arms)
    arms = arms.drop_duplicates(["source", "target", "backbone", "seed", "condition"])
    if len(arms) != n0:
        print(f"  dropped {n0-len(arms)} duplicate arm rows")
    probes = probes.drop_duplicates(["source", "target", "backbone", "seed",
                                     "n_studies", "rep"])
    write_csv(arms, out / "anatomy_transfer_arms_merged.csv")

    print("\n--- completeness ---")
    complete = check_completeness(arms, out)

    # ---------------------------------------------------------------- AdaBN
    print("\n" + "=" * 78)
    print("ADABN RECEIPTS -- did the intervention actually execute?")
    print("=" * 78)
    if len(rec):
        r = rec.groupby(["backbone", "norm_family"], as_index=False).agg(
            cells=("mean_drift", "size"),
            bn_modules=("n_bn_modules", "mean"),
            batches=("n_batches", "mean"),
            mean_drift=("mean_drift", "mean"))
        print(r.to_string(index=False))
        bn = r[r.norm_family == "batchnorm"]
        if len(bn) and (bn.mean_drift.fillna(0) < 1e-6).any():
            print("  !! BatchNorm backbone shows ~0 drift -- AdaBN was inert. "
                  "Do not report it as tested.")
        elif len(bn):
            print(f"  BatchNorm backbones: statistics genuinely moved "
                  f"(mean relative drift {bn.mean_drift.mean():.4f}).")
        print("  LayerNorm backbones have no BatchNorm; applicable=0 is a "
              "DEFINED no-op, not a silent one.")

    # -------------------------------------------------------- decomposition
    dec = decompose(arms)
    write_csv(dec, out / "anatomy_gap_decomposition.csv")

    print("\n" + "=" * 78)
    print("GAP DECOMPOSITION  (study-level AUC; cross-anatomy pairs only)")
    print("=" * 78)
    print(f"  cells: {len(dec)}")
    print(f"  ceiling  AUC  {dec.ceiling_auc_study.mean():.4f}")
    print(f"  zeroshot AUC  {dec.zeroshot_auc_study.mean():.4f}")
    print(f"  GAP           {dec.gap_auc_study.mean():.4f}"
          f"   [{boot_ci(dec.gap_auc_study)[0]:.4f}, "
          f"{boot_ci(dec.gap_auc_study)[1]:.4f}]")
    print("\n  what moves AUC:")
    lo, hi = boot_ci(dec.adabn_delta_auc)
    print(f"    adabn   (label-free)  dAUC = {dec.adabn_delta_auc.mean():+.4f}"
          f"  [{lo:+.4f}, {hi:+.4f}]   "
          f"{100*frac(dec.adabn_delta_auc, dec.gap_auc_study).mean():+.1f}% of gap")
    print("\n  what moves balanced accuracy only (a threshold cannot change AUC):")
    print(f"    prior   (label-free)  dBACC = {dec.prior_delta_bacc.mean():+.4f}")
    print(f"    oracle  (USES LABELS) dBACC = {dec.oracle_delta_bacc.mean():+.4f}"
          f"   {100*frac(dec.oracle_delta_bacc, dec.gap_bacc).mean():+.1f}% of bacc gap")

    print("\n--- by normalisation family (the architectural BatchNorm test) ---")
    byn = dec.groupby("norm_family").agg(
        cells=("gap_auc_study", "size"),
        ceiling=("ceiling_auc_study", "mean"),
        zeroshot=("zeroshot_auc_study", "mean"),
        gap=("gap_auc_study", "mean"),
        adabn_dAUC=("adabn_delta_auc", "mean")).round(4)
    print(byn.to_string())
    print("  If the gap were a BatchNorm-statistic effect, LayerNorm backbones "
          "should show a smaller gap.")

    print("\n--- by backbone ---")
    print(dec.groupby(["backbone_short", "norm_family"]).agg(
        ceiling=("ceiling_auc_study", "mean"),
        zeroshot=("zeroshot_auc_study", "mean"),
        gap=("gap_auc_study", "mean"),
        adabn_dAUC=("adabn_delta_auc", "mean")).round(4).to_string())

    # ---------------------------------------------------------- probe curve
    pc = probe_curve(probes, dec)
    write_csv(pc, out / "anatomy_probe_curve.csv")

    print("\n" + "=" * 78)
    print("LABEL BUDGET -- re-fitting ONLY the readout on frozen features")
    print("=" * 78)
    g = pc.groupby("n_studies").agg(
        cells=("probe_auc_study", "size"),
        probe_auc=("probe_auc_study", "mean"),
        dAUC=("probe_delta_auc", "mean"),
        recovery=("recovery_frac", "mean"),
        beats_ceiling=("beats_ceiling", "mean")).round(4)
    zs_mean = dec.zeroshot_auc_study.mean()
    ce_mean = dec.ceiling_auc_study.mean()
    print(f"  zero-shot {zs_mean:.4f}   ceiling {ce_mean:.4f}   "
          f"gap {ce_mean-zs_mean:.4f}\n")
    print(g.to_string())

    print("\n--- smallest budget reaching a recovery threshold ---")
    for thr in (0.5, 0.8, 0.9, 1.0):
        hit = g[g.recovery >= thr]
        print(f"  >= {int(thr*100):3d}% of gap : "
              f"{('n=' + str(int(hit.index[0]))) if len(hit) else 'not reached'}")

    print("\n--- probe recovery by normalisation family ---")
    print(pc.groupby(["norm_family", "n_studies"]).recovery_frac
          .mean().unstack(0).round(3).to_string())

    # ------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    ab = dec.adabn_delta_auc.mean()
    best_n = g.recovery.idxmax() if g.recovery.notna().any() else None
    best_r = g.recovery.max() if best_n is not None else np.nan
    print(f"  AUC gap to close                     {ce_mean-zs_mean:+.4f}")
    print(f"  adabn   (0 labels)  recovers         {ab:+.4f}"
          f"  ({100*ab/max(1e-9, ce_mean-zs_mean):+.1f}%)")
    if best_n is not None:
        bd = g.loc[best_n, "dAUC"]
        print(f"  probe@{int(best_n):<4d}(n labels)  recovers   {bd:+.4f}"
              f"  ({100*best_r:+.1f}%)")
    print()
    if best_n is not None and best_r > 0.5 and abs(ab) < 0.02:
        print("  => Re-fitting the readout recovers the gap; label-free")
        print("     statistic adaptation does not. The failure is in the")
        print("     READOUT DIRECTION, not the encoder and not BatchNorm.")
        print("     Objective 3 becomes: how few labels, and can anything")
        print("     label-free beat this bar?")
    elif best_n is not None and best_r <= 0.5:
        print("  => Re-fitting the readout does NOT recover the gap either.")
        print("     The readout-direction hypothesis is WRONG; the encoder")
        print("     itself is implicated. Re-plan before building methods.")
    else:
        print("  => Mixed. Inspect anatomy_probe_curve.csv per backbone.")
    if not complete:
        print("\n  NOTE: cells were missing; treat every number above as "
              "provisional until anatomy_missing_cells.csv is empty.")
    print("=" * 78)
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
