#!/usr/bin/env python3
"""Harm magnitude, harm margins and worst losses on the external arm.

On the external cells (GRAZPEDWRI-DX initial examinations, MFXR and FracAtlas
hand; three backbones; 1,345 cells) writes, per policy:

    harm_magnitude.csv    cells, harmed cells, harm rate, size of the losses
                             (median, quartiles, 90th percentile, worst) and mean
                             AUC gain, pooled and per budget
    harm_with_margin.csv  cells whose AUC falls by more than 0, 0.01 and 0.02
    worst_loss_cells.csv  the ten largest losses under the cross-validated
                             weight and under the fixed-penalty refit

Usage:
    python external_harm_analyses.py \
        --b13 ../results/transfer/readout_shrinkage_XR_WRIST.csv \
        --b14 ../results/transfer/fracatlas_budget_cells_XR_WRIST_XR_HAND.csv \
        --out-dir <dir>

CPU only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from birs_schedule import (PAPER_BACKBONES, TAU_PRIMARY,  # noqa: E402
                                       apply_schedule, baseline_rows,
                                       fit_schedule, load_grid,
                                       per_budget_stats, split_arms)

MARGINS = [0.0, 0.01, 0.02]
NAMED = ["shrink_nested", "refit_nestedC", "refit_C1"]
WORST_FOR = ["shrink_nested", "refit_C1"]
TOP_N = 10
Q5_COLUMNS = ["rank", "target", "backbone", "seed", "budget", "resample",
              "setting", "lam", "C", "chosen_C", "chosen_lam", "auc_study",
              "zeroshot_auc", "delta_vs_zeroshot", "harm", "loss", "lam0_auc"]
CELL = ["target", "backbone", "seed", "budget", "resample"]


def policies(paths, backbones):
    """Held-out-arm rows for every policy, as {name: frame}, in output order."""
    grid = load_grid(paths, backbones)
    cal, test = split_arms(grid)
    sched, _ = fit_schedule(per_budget_stats(cal), TAU_PRIMARY)
    oracle, _ = fit_schedule(per_budget_stats(test), TAU_PRIMARY)
    threshold = max(b for b, lam in sched.items() if lam == 0.0)
    abstain = {b: (0.0 if b <= threshold else 1.0) for b in sched}
    print(f"[q] schedule {sched}\n[q] oracle   {oracle}\n[q] abstain at <= {threshold}")

    out = {f"schedule_tau{TAU_PRIMARY:g}": apply_schedule(test, sched),
           "oracle_schedule": apply_schedule(test, oracle),
           f"threshold_only_abstain{threshold}": apply_schedule(test, abstain)}
    named = baseline_rows(paths, backbones, sorted(test.target.unique()))
    for s in NAMED:
        out[s] = named[named.setting == s].reset_index(drop=True)
    for lam in sorted(test.lam.unique()):
        sub = test[np.isclose(test.lam, lam)].reset_index(drop=True)
        out[str(sub.setting.iloc[0])] = sub

    # The first inline computation kept only cells present for EVERY policy.
    # This script compares policies directly, which is the same thing only
    # while every policy covers the same cells. Stop if a rerun ever breaks that.
    counts = {name: tuple(sorted(d.groupby("budget").size().items()))
              for name, d in out.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit("!! policies cover different cells; per-budget counts:\n"
                         + "\n".join(f"   {k}: {v}" for k, v in counts.items()))
    return out, test


def magnitude(d):
    """One q1 row: counts over all cells, loss sizes over harmed cells."""
    loss = -d.loc[d.harm == 1, "delta_vs_zeroshot"]
    row = {"n_cells": int(len(d)), "n_harmed": int(len(loss)),
           "harm_rate": float(d.harm.mean())}
    if len(loss):
        q25, q75 = loss.quantile(0.25), loss.quantile(0.75)
        row.update(median_loss=loss.median(), q25_loss=q25, q75_loss=q75,
                   iqr_loss=q75 - q25, p90_loss=loss.quantile(0.90),
                   worst_loss=loss.max())
    else:
        row.update(median_loss=np.nan, q25_loss=np.nan, q75_loss=np.nan,
                   iqr_loss=np.nan, p90_loss=np.nan, worst_loss=np.nan)
    row["mean_gain"] = float(d.delta_vs_zeroshot.mean())
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b13", required=True)
    ap.add_argument("--b14", required=True)
    ap.add_argument("--backbones", nargs="+", default=PAPER_BACKBONES)
    ap.add_argument("--out-dir", required=True)
    cfg = ap.parse_args()
    out_dir = Path(cfg.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [cfg.b13, cfg.b14]

    pol, test = policies(paths, cfg.backbones)

    q1 = []
    for name, d in pol.items():
        q1.append({"policy": name, "budget": "pooled", **magnitude(d)})
        for b in sorted(d.budget.unique()):
            q1.append({"policy": name, "budget": int(b),
                       **magnitude(d[d.budget == b])})
    pd.DataFrame(q1).to_csv(out_dir / "harm_magnitude.csv", index=False)

    q2 = []
    for name, d in pol.items():
        loss = -d.delta_vs_zeroshot
        for m in MARGINS:
            n = int((loss > m).sum())
            q2.append({"policy": name, "margin": m, "n_cells": int(len(d)),
                       "n_harmed": n, "harm_rate": n / len(d)})
    pd.DataFrame(q2).to_csv(out_dir / "harm_with_margin.csv", index=False)

    # q5 needs the full b13/b14 columns (C, chosen_C, chosen_lam), not the
    # harmonised grid, so it reads the named rows again with every column.
    full = pd.concat([pd.read_csv(p, comment="#") for p in paths], ignore_index=True)
    full = full[full.backbone.isin(cfg.backbones)
                & full.target.isin(sorted(test.target.unique()))]
    lam0 = (full[full.setting == "shrink_lam0"][CELL + ["auc_study"]]
            .rename(columns={"auc_study": "lam0_auc"}))
    q5 = []
    for s in WORST_FOR:
        d = full[(full.setting == s) & (full.harm == 1)].copy()
        d["loss"] = -d.delta_vs_zeroshot
        d = d.sort_values("loss", ascending=False).head(TOP_N)
        d = d.merge(lam0, on=CELL, how="left")
        d.insert(0, "rank", range(1, len(d) + 1))
        q5.append(d[Q5_COLUMNS])
    q5 = pd.concat(q5, ignore_index=True)
    if not np.allclose(q5.lam0_auc, q5.zeroshot_auc, atol=1e-12):
        raise SystemExit("!! lambda = 0 AUC differs from the zero-shot AUC in a q5 cell")
    q5.to_csv(out_dir / "worst_loss_cells.csv", index=False)

    print(f"[q] wrote q1 ({len(q1)} rows), q2 ({len(q2)} rows), q5 ({len(q5)} rows) "
          f"to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
