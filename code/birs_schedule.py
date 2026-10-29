#!/usr/bin/env python3
"""Fit the budget-indexed shrinkage schedule (BIRS) and test it on the external
datasets.

For each budget, chooses the weight lam on the grid that maximises the mean AUC
gain over the transferred model subject to the harm rate (share of cells below
the transferred model) being at most tau, on the six MURA anatomies only. Ties
go to the smaller lam, a budget with no feasible non-zero lam gets lam = 0, and
the schedule is made non-decreasing by a running maximum. The schedule is then
applied unchanged to GRAZPEDWRI-DX, MFXR and FracAtlas and compared with the
other policies, the cap sensitivity (tau grid) and an oracle schedule fitted on
the external arm itself.

Inputs:
    --b13  readout_shrinkage_XR_WRIST.csv
    --b14  fracatlas_budget_cells_XR_WRIST_XR_HAND.csv (optional)

Output:
    birs_schedule.csv (path set by --out)

Usage:
    python birs_schedule.py --b13 <csv> --b14 <csv>

CPU only. Also imported for ``fit_schedule``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PAPER_BACKBONES = ["densenet121", "convnext_tiny", "swin_tiny_patch4_window7_224"]
TAU_GRID = [0.0, 0.05, 0.10, 0.20]
TAU_PRIMARY = 0.05
LAM_RE = re.compile(r"^shrink_lam(?P<lam>[0-9.]+)$")

# Columns shared by the b13 and b14 schemas. b14 carries extra cohort columns
# that have no counterpart in b13; taking the intersection keeps the concat
# honest instead of filling NaNs.
COMMON = ["source", "target", "backbone", "seed", "budget", "resample",
          "setting", "lam", "auc_study", "zeroshot_auc", "delta_vs_zeroshot",
          "harm"]

ROW_COLUMNS = ["arm", "target", "budget", "lam", "tau", "rule", "n_cells",
               "harm_rate", "mean_gain", "median_gain"]


def load_grid(paths, backbones):
    """Fixed-lambda rows from every supplied csv, harmonised and concatenated."""
    frames = []
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        d = pd.read_csv(p, comment="#")
        missing = [c for c in COMMON if c not in d.columns]
        if missing:
            raise SystemExit(f"!! {p.name} is missing {missing}; it was not "
                             f"written by readout_shrinkage/b14.")
        d = d[d.backbone.isin(backbones)]
        d = d[d.setting.astype(str).str.match(LAM_RE)]
        if not len(d):
            print(f"[skip] {p.name}: no shrink_lam* rows")
            continue
        d = d[COMMON].copy()
        d["src_file"] = p.name
        frames.append(d)
        print(f"[load] {p.name}: {len(d)} fixed-lambda rows, "
              f"targets {sorted(d.target.unique())}")
    if not frames:
        raise SystemExit("!! no usable rows in any input")
    g = pd.concat(frames, ignore_index=True)
    g["lam"] = g["lam"].astype(float)
    g["budget"] = g["budget"].astype(int)
    return g


def split_arms(g):
    """Anatomy targets calibrate; cross-dataset targets test.

    A MURA anatomy target is one whose name is an XR_* region. Everything else
    -- GRAZ_initial, MFXR, FRAC_XR_HAND -- is a dataset change and belongs to
    the test arm. The test never informs the fit.
    """
    is_anat = g.target.astype(str).str.match(r"^XR_[A-Z]+$")
    cal, test = g[is_anat].copy(), g[~is_anat].copy()
    if not len(cal):
        raise SystemExit("!! no anatomy targets to calibrate on")
    if not len(test):
        raise SystemExit("!! no cross-dataset targets to test on; run b13 with "
                         "--with-mfxr and/or supply the b14 csv")
    print(f"\n[arms] calibrate on {sorted(cal.target.unique())}")
    print(f"[arms] test on      {sorted(test.target.unique())}")
    return cal, test


def per_budget_stats(d):
    """harm rate and mean gain for every (budget, lambda)."""
    return (d.groupby(["budget", "lam"])
              .agg(n_cells=("harm", "size"), harm_rate=("harm", "mean"),
                   mean_gain=("delta_vs_zeroshot", "mean"),
                   median_gain=("delta_vs_zeroshot", "median"))
              .reset_index())


def fit_schedule(stats, tau, monotone=True):
    """lambda*(budget) = argmax mean_gain subject to harm_rate <= tau.

    Ties on gain are broken toward the SMALLER lambda: between two lambdas with
    equal expected benefit, the more conservative one carries less risk of a
    bad site, and the constraint is a cap not a target.

    Falls back to lambda = 0 when no grid value satisfies the constraint, which
    is the honest answer -- at that budget nothing on the grid is safe enough,
    so do not adapt.
    """
    out = {}
    for b, g in stats.groupby("budget"):
        ok = g[g.harm_rate <= tau + 1e-12]
        if not len(ok):
            out[int(b)] = 0.0
            continue
        best = ok.mean_gain.max()
        out[int(b)] = float(ok[ok.mean_gain >= best - 1e-12].lam.min())
    raw = dict(sorted(out.items()))
    if not monotone:
        return raw, raw
    run, mono = -np.inf, {}
    for b in sorted(raw):
        run = max(run, raw[b])
        mono[b] = float(run)
    return mono, raw


def apply_schedule(d, sched):
    """Rows selected by a budget -> lambda map. One lambda per budget."""
    keep = [d[(d.budget == b) & (np.isclose(d.lam, lam))]
            for b, lam in sched.items()]
    keep = [k for k in keep if len(k)]
    return pd.concat(keep, ignore_index=True) if keep else d.iloc[0:0]


def arm_summary(d, label):
    if not len(d):
        return None
    return {"rule": label, "n_cells": int(len(d)),
            "harm_rate": float(d.harm.mean()),
            "mean_gain": float(d.delta_vs_zeroshot.mean()),
            "median_gain": float(d.delta_vs_zeroshot.median())}


def baseline_rows(paths, backbones, test_targets):
    """The named settings b13/b14 already evaluated, on the test arm only."""
    frames = []
    for p in paths:
        if p is None or not Path(p).exists():
            continue
        d = pd.read_csv(p, comment="#")
        d = d[d.backbone.isin(backbones) & d.target.isin(test_targets)]
        d = d[d.setting.isin(["refit_C1", "refit_nestedC", "shrink_nested"])]
        if len(d):
            frames.append(d[["target", "budget", "setting", "harm",
                             "delta_vs_zeroshot"]])
    return pd.concat(frames, ignore_index=True) if frames else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b13", required=True)
    ap.add_argument("--b14", default=None)
    ap.add_argument("--backbones", nargs="+", default=PAPER_BACKBONES)
    ap.add_argument("--tau", type=float, default=TAU_PRIMARY)
    ap.add_argument("--tau-grid", nargs="+", type=float, default=TAU_GRID)
    ap.add_argument("--no-monotone", action="store_true")
    ap.add_argument("--out", default=None,
                    help="csv of every arm/rule/budget row; defaults next to "
                         "the b13 file")
    cfg = ap.parse_args()

    g = load_grid([cfg.b13, cfg.b14], cfg.backbones)
    cal, test = split_arms(g)
    budgets = [int(b) for b in sorted(g.budget.unique())]
    lams = [float(v) for v in sorted(g.lam.unique())]
    print(f"[grid] budgets {budgets}\n[grid] lambdas {lams}")

    cal_stats = per_budget_stats(cal)
    test_stats = per_budget_stats(test)

    print("\n" + "=" * 74)
    print("CALIBRATION ARM (anatomy): harm rate by budget and fixed lambda")
    print("=" * 74)
    print(cal_stats.pivot(index="budget", columns="lam",
                          values="harm_rate").round(4).to_string())
    print("\nmean AUC gain, same cells")
    print(cal_stats.pivot(index="budget", columns="lam",
                          values="mean_gain").round(4).to_string())

    # ---- schedules over tau -------------------------------------------------
    print("\n" + "=" * 74)
    print("FITTED SCHEDULES  lambda*(budget), by tau  (fit on anatomy only)")
    print("=" * 74)
    sched_by_tau = {}
    for tau in cfg.tau_grid:
        mono, raw = fit_schedule(cal_stats, tau, not cfg.no_monotone)
        sched_by_tau[tau] = mono
        star = " <- primary" if abs(tau - cfg.tau) < 1e-12 else ""
        print(f"  tau={tau:<5g} " +
              "  ".join(f"n={b}:{mono[b]:g}" for b in budgets) + star)
        if mono != raw:
            print(f"           unconstrained: " +
                  "  ".join(f"n={b}:{raw[b]:g}" for b in budgets)
                  + "   (monotonicity changed it -- report both)")

    sched = sched_by_tau[cfg.tau]
    if all(v == 0.0 for v in sched.values()):
        print(f"\n  !! at tau={cfg.tau:g} no grid lambda is safe at any budget: "
              "the prescription is to never adapt.\n"
              "     That is a reportable result, not a failure, but check the "
              "tau sensitivity above before quoting it.")

    # ---- evaluate on the held-out cross-dataset arm -------------------------
    rows, summaries = [], []
    s_test = apply_schedule(test, sched)
    summaries.append(arm_summary(s_test, f"schedule(tau={cfg.tau:g})"))

    zs = test[np.isclose(test.lam, 0.0)]
    summaries.append(arm_summary(zs, "zero-shot (lam=0)"))

    bl = baseline_rows([cfg.b13, cfg.b14], cfg.backbones,
                       sorted(test.target.unique()))
    if bl is not None:
        for s in ("refit_C1", "refit_nestedC", "shrink_nested"):
            sub = bl[bl.setting == s]
            if len(sub):
                summaries.append({"rule": s, "n_cells": int(len(sub)),
                                  "harm_rate": float(sub.harm.mean()),
                                  "mean_gain": float(sub.delta_vs_zeroshot.mean()),
                                  "median_gain": float(
                                      sub.delta_vs_zeroshot.median())})

    oracle, _ = fit_schedule(test_stats, cfg.tau, not cfg.no_monotone)
    summaries.append(arm_summary(apply_schedule(test, oracle),
                                 f"ORACLE schedule (not achievable)"))

    print("\n" + "=" * 74)
    print(f"TEST ARM (cross-dataset), tau={cfg.tau:g}")
    print("=" * 74)
    print(pd.DataFrame([s for s in summaries if s])
          .set_index("rule").round(4).to_string())
    print(f"\n  prescribed schedule: "
          + "  ".join(f"n={b}:{sched[b]:g}" for b in budgets))
    print(f"  oracle schedule:     "
          + "  ".join(f"n={b}:{oracle[b]:g}" for b in budgets))
    print("  The oracle is fitted ON the test arm and cannot be used; it is "
          "here\n  only to show how much the prescription leaves behind.")

    print("\nPER-TARGET, prescribed schedule")
    if len(s_test):
        print(s_test.groupby("target")
              .agg(n=("harm", "size"), harm=("harm", "mean"),
                   gain=("delta_vs_zeroshot", "mean")).round(4).to_string())

    # ---- leave-one-anatomy-out: does the schedule depend on one target? ----
    print("\n" + "=" * 74)
    print("LEAVE-ONE-ANATOMY-OUT (does the fit hang on a single target?)")
    print("=" * 74)
    stable = True
    loo = {}
    for t in sorted(cal.target.unique()):
        s_t, _ = fit_schedule(per_budget_stats(cal[cal.target != t]),
                              cfg.tau, not cfg.no_monotone)
        loo[t] = s_t
        same = s_t == sched
        stable &= same
        print(f"  drop {t:<12s} " +
              "  ".join(f"{s_t[b]:g}" for b in budgets)
              + ("" if same else "   <- DIFFERS"))
    print(f"\n  schedule is {'STABLE' if stable else 'NOT stable'} to dropping "
          "any single calibration target.")

    # ---- write ------------------------------------------------------------
    for arm, d in (("calibrate", cal), ("test", test)):
        st = per_budget_stats(d)
        for _, r in st.iterrows():
            rows.append({"arm": arm, "target": "ALL", "budget": int(r.budget),
                         "lam": float(r.lam), "tau": np.nan, "rule": "grid",
                         "n_cells": int(r.n_cells), "harm_rate": r.harm_rate,
                         "mean_gain": r.mean_gain,
                         "median_gain": r.median_gain})
    for tau, sc in sched_by_tau.items():
        for b, lam in sc.items():
            sub = test[(test.budget == b) & (np.isclose(test.lam, lam))]
            if not len(sub):
                continue
            rows.append({"arm": "test", "target": "ALL", "budget": int(b),
                         "lam": float(lam), "tau": float(tau),
                         "rule": "schedule", "n_cells": int(len(sub)),
                         "harm_rate": float(sub.harm.mean()),
                         "mean_gain": float(sub.delta_vs_zeroshot.mean()),
                         "median_gain": float(sub.delta_vs_zeroshot.median())})
    # Overall test-arm comparison: one row per rule, budget -1 meaning pooled
    # over budgets. Printed above; written here because any number the
    # paper may quote has to come from paper_numbers.json, not a log.
    for sm in summaries:
        if not sm:
            continue
        rows.append({"arm": "test", "target": "ALL", "budget": -1,
                     "lam": np.nan, "tau": float(cfg.tau), "rule": sm["rule"],
                     "n_cells": sm["n_cells"], "harm_rate": sm["harm_rate"],
                     "mean_gain": sm["mean_gain"],
                     "median_gain": sm["median_gain"]})
    # The prescribed and oracle schedules themselves, as lambda per budget.
    for name, sc in (("schedule_lam", sched), ("oracle_lam", oracle)):
        for b, lam in sc.items():
            rows.append({"arm": "summary", "target": "ALL", "budget": int(b),
                         "lam": float(lam), "tau": float(cfg.tau),
                         "rule": name, "n_cells": 0, "harm_rate": np.nan,
                         "mean_gain": np.nan, "median_gain": np.nan})
    # Per-target test records. The pooled zero cannot distinguish "holds on
    # every dataset" from "carried by one", so the split is persisted here
    # rather than left in the log, where the next run would lose it.
    if len(s_test):
        for _t, _g in s_test.groupby("target"):
            rows.append({"arm": "test", "target": str(_t), "budget": -1,
                         "lam": np.nan, "tau": float(cfg.tau),
                         "rule": "per_target", "n_cells": int(len(_g)),
                         "harm_rate": float(_g.harm.mean()),
                         "mean_gain": float(_g.delta_vs_zeroshot.mean()),
                         "median_gain": float(_g.delta_vs_zeroshot.median())})

    # Leave-one-out stability, so the paper cannot call the schedule stable
    # without the check itself having been recorded.
    # Per-drop schedules, machine-readable. The single loo_stable boolean
    # says the schedule moved but not WHERE, and the distinction matters:
    # if every drop keeps lambda = 0 over the same low budgets then the
    # abstain threshold is robust even when the ramp above it is not.
    def _abstain(sc):
        z = [b for b, lam in sc.items() if lam == 0.0]
        return max(z) if z else 0
    for t, sc in loo.items():
        for b, lam in sc.items():
            rows.append({"arm": "summary", "target": t, "budget": int(b),
                         "lam": float(lam), "tau": float(cfg.tau),
                         "rule": "loo_drop", "n_cells": 0,
                         "harm_rate": np.nan, "mean_gain": np.nan,
                         "median_gain": np.nan})
    _ab = [_abstain(sc) for sc in loo.values()] or [0]
    rows.append({"arm": "summary", "target": "ALL", "budget": int(min(_ab)),
                 "lam": np.nan, "tau": float(cfg.tau),
                 "rule": "loo_abstain_min", "n_cells": len(loo),
                 "harm_rate": float(min(_ab) == _abstain(sched)),
                 "mean_gain": np.nan, "median_gain": np.nan})
    rows.append({"arm": "summary", "target": "ALL", "budget": -1,
                 "lam": np.nan, "tau": float(cfg.tau), "rule": "loo_stable",
                 "n_cells": int(cal.target.nunique()),
                 "harm_rate": float(bool(stable)), "mean_gain": np.nan,
                 "median_gain": np.nan})

    out = Path(cfg.out) if cfg.out else \
        Path(cfg.b13).with_name("birs_schedule.csv")
    pd.DataFrame(rows)[ROW_COLUMNS].to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
