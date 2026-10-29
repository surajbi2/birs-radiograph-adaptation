#!/usr/bin/env python3
"""Every policy on the left-out anatomy of the leave-one-anatomy-out check.

For each MURA anatomy, the schedule is refitted on the other five anatomies
(birs_schedule.fit_schedule, tau = 0.05, non-decreasing) and applied
to the anatomy left out; every other policy is scored on exactly the same cells,
so the comparison is paired.

Input:  results/transfer/readout_shrinkage_XR_WRIST.csv
Output: results/transfer/harm_analyses/loo_all_policies.csv

CPU only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from birs_schedule import (PAPER_BACKBONES, TAU_PRIMARY,  # noqa: E402
                                       fit_schedule, load_grid,
                                       per_budget_stats)

RES = Path.home() / "diagnosis/results/transfer"
B13 = RES / "readout_shrinkage_XR_WRIST.csv"
OUT = RES / "harm_analyses/loo_all_policies.csv"
KEY = ["target", "backbone", "seed", "budget", "resample"]
ABSTAIN_UPTO = 50


def main() -> int:
    raw = pd.read_csv(B13, comment="#")
    raw = raw[raw.backbone.isin(PAPER_BACKBONES) & (raw.budget > 0)]
    raw = raw[raw.target.astype(str).str.match(r"^XR_[A-Z]+$")]
    raw["lam"] = raw["lam"].astype(float)

    # the calibration grid exactly as b16 builds it
    grid = load_grid([B13], PAPER_BACKBONES)
    cal = grid[grid.target.astype(str).str.match(r"^XR_[A-Z]+$")]

    def by_setting(d, s):
        return d[d.setting == s].set_index(KEY)

    rows = []
    for t in sorted(cal.target.unique()):
        sched, _ = fit_schedule(per_budget_stats(cal[cal.target != t]),
                                TAU_PRIMARY, True)
        d = raw[raw.target == t]
        lam_at = d.budget.map(sched)

        pol = {
            "loo_schedule": d[np.isclose(d.lam, lam_at)
                              & d.setting.str.startswith("shrink_lam")],
            "no_adaptation": d[d.setting == "shrink_lam0"],
            "refit_C1": d[d.setting == "refit_C1"],
            "refit_tuned": d[d.setting == "refit_nestedC"],
            "cv_lambda": d[d.setting == "shrink_nested"],
            "abstain50_lam1": pd.concat([
                d[(d.setting == "shrink_lam0") & (d.budget <= ABSTAIN_UPTO)],
                d[(d.setting == "shrink_lam1") & (d.budget > ABSTAIN_UPTO)]]),
        }
        pol = {k: v.set_index(KEY) for k, v in pol.items()}
        for k, v in pol.items():
            if v.index.duplicated().any():
                raise SystemExit(f"!! duplicate cells for {t}/{k}")
        common = None
        for v in pol.values():
            common = v.index if common is None else common.intersection(v.index)
        sizes = {k: len(v) for k, v in pol.items()}
        if len(set(sizes.values())) != 1:
            print(f"[pair] {t}: unequal cell sets {sizes}; using intersection "
                  f"{len(common)}")

        zs = pol["no_adaptation"].loc[common, "zeroshot_auc"]
        for k, v in pol.items():
            v = v.loc[common]
            assert np.allclose(v.zeroshot_auc, zs)
            loss = -v.delta_vs_zeroshot[v.harm == 1]
            rows.append({
                "left_out_anatomy": t, "policy": k,
                "schedule": ("; ".join(f"n={b}:{sched[b]:g}" for b in sorted(sched))
                             if k == "loo_schedule" else ""),
                "n_cells": len(v), "n_harmed": int(v.harm.sum()),
                "harm_rate": float(v.harm.mean()),
                "mean_gain": float(v.delta_vs_zeroshot.mean()),
                "median_loss_harmed": float(loss.median()) if len(loss) else np.nan,
                "worst_loss": float(loss.max()) if len(loss) else np.nan,
                "zeroshot_auc": float(zs.mean()),
            })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.drop(columns="schedule").round(4).to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
