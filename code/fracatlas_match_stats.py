#!/usr/bin/env python3
"""Significance tests for the FracAtlas anatomy-match contrast.

Tests matched against mismatched MURA sources on the FracAtlas targets at the
cell level (backbone x seed), with one-sided Mann-Whitney tests in both
directions and per-source spreads with seed-level standard errors.

Output:
    fracatlas_match_stats.csv (path set by --out)

Usage:
    python fracatlas_match_stats.py

CPU only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mskmodel as M

N_PERM = 20000


def main() -> None:
    from scipy import stats
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "fracatlas_transfer.csv"))
    ap.add_argument("--out", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "fracatlas_match_stats.csv"))
    cfg = ap.parse_args()
    d = pd.read_csv(cfg.csv, comment="#", low_memory=False)
    rows = []

    for tgt, sub in d.groupby("target"):
        print("=" * 78)
        print(f"TARGET {tgt}   n={int(sub.n_images.iloc[0])} images, "
              f"prevalence {sub.target_prev.iloc[0]:.3f}")
        print("=" * 78)

        print("\n1. PER-SOURCE, with seed-level standard errors")
        t = (sub.groupby("source").zs_auc_study
             .agg(mean="mean", sd="std", n="size").reset_index())
        t["se"] = t.sd / np.sqrt(t.n)
        t["matched"] = (t.source == tgt).astype(int)
        t = t.sort_values("mean", ascending=False)
        print(f"  {'source':<14}{'mean':>9}{'sd':>8}{'se':>8}{'n':>4}")
        for _, r in t.iterrows():
            flag = "   <-- MATCHED" if r.matched else ""
            print(f"  {r.source:<14}{r['mean']:>9.4f}{r.sd:>8.4f}"
                  f"{r.se:>8.4f}{int(r.n):>4}{flag}")
        spread = t["mean"].max() - t["mean"].min()
        print(f"\n  spread across sources {spread:.4f}; "
              f"typical seed-level se {t.se.median():.4f} "
              f"({spread / max(t.se.median(), 1e-9):.1f}x)")

        m = sub[sub.matched == 1].zs_auc_study
        o = sub[sub.matched == 0].zs_auc_study
        if len(m) == 0:
            print("  no matched source for this target; skipping")
            continue

        print("\n2. MATCHED vs MISMATCHED  (both directions reported)")
        print(f"  matched    n={len(m):>3}  {m.mean():.4f} +- {m.std():.4f}")
        print(f"  mismatched n={len(o):>3}  {o.mean():.4f} +- {o.std():.4f}")
        print(f"  difference {m.mean() - o.mean():+.4f}")
        p_less = stats.mannwhitneyu(m, o, alternative="less").pvalue
        p_grt = stats.mannwhitneyu(m, o, alternative="greater").pvalue
        p_two = stats.mannwhitneyu(m, o, alternative="two-sided").pvalue
        print(f"  Mann-Whitney matched < mismatched : p = {p_less:.4g}")
        print(f"  Mann-Whitney matched > mismatched : p = {p_grt:.4g}   "
              "(the pre-registered F1 direction)")
        print(f"  two-sided                         : p = {p_two:.4g}")

        print("\n3. PAIRED BY (backbone, seed) -- better powered")
        piv = sub.pivot_table(index=["backbone", "seed"], columns="source",
                             values="zs_auc_study")
        if tgt in piv.columns:
            others = piv.drop(columns=[tgt]).mean(axis=1)
            diff = piv[tgt] - others
            w = stats.wilcoxon(diff, alternative="less") if len(diff) > 5 else None
            print(f"  matched minus mean-of-others, per cell: "
                  f"{diff.mean():+.4f} +- {diff.std():.4f}, n = {len(diff)}")
            print(f"  negative in {int((diff < 0).sum())}/{len(diff)} cells")
            if w is not None:
                print(f"  Wilcoxon signed-rank (matched < others): p = {w.pvalue:.4g}")
            tt = stats.ttest_1samp(diff, 0.0)
            print(f"  paired t vs 0: t = {tt.statistic:+.2f}, p = {tt.pvalue:.4g}")

        print(f"\n4. PERMUTATION ON THE DIFFERENCE  (NOT on the rank)")
        print("   The obvious test -- 'how often does the matched source land")
        print("   last under shuffling' -- CANNOT WORK. With 7 sources the")
        print("   statistic 'rank >= 7' has a minimum achievable p of 1/7 =")
        print("   0.143, so it can never reject at 0.05 no matter how large the")
        print("   effect. Validated on synthetic data: a penalty of 0.00 and a")
        print("   penalty of 0.11 both returned p ~ 0.135. The pre-registered")
        print("   'rank permutation p < 0.05' clause was therefore IMPOSSIBLE to")
        print("   satisfy and is replaced, before seeing the result, with a")
        print("   permutation on the CONTINUOUS difference, which has power.")
        obs_rank = int((t["mean"] > t.loc[t.matched == 1, "mean"].iloc[0]).sum()) + 1
        n_src = len(t)
        vals = sub.pivot_table(index=["backbone", "seed"], columns="source",
                               values="zs_auc_study")
        cols = list(vals.columns)
        obs_diff = float(vals[tgt].mean()
                         - vals.drop(columns=[tgt]).values.mean())
        rs = np.random.RandomState(20260805)
        null = np.empty(N_PERM)
        V = vals.values
        j = cols.index(tgt)
        for i in range(N_PERM):
            perm = np.apply_along_axis(rs.permutation, 1, V)
            null[i] = perm[:, j].mean() - np.delete(perm, j, axis=1).mean()
        p_perm = float((null <= obs_diff).mean())
        print(f"  observed rank of the matched source: {obs_rank} / {n_src} "
              "(descriptive only)")
        print(f"  observed difference (matched - others): {obs_diff:+.4f}")
        print(f"  permutation null: {null.mean():+.4f} +- {null.std():.4f}")
        print(f"  permutation p (difference this low or lower): {p_perm:.4g}  "
              f"[{N_PERM} draws, shuffled WITHIN backbone x seed]")
        p_rank = p_perm

        rows.append({"target": tgt, "matched_mean": m.mean(),
                     "mismatched_mean": o.mean(),
                     "difference": m.mean() - o.mean(),
                     "p_matched_less": p_less, "p_matched_greater": p_grt,
                     "p_two_sided": p_two, "observed_rank": obs_rank,
                     "n_sources": n_src, "observed_difference": obs_diff,
                     "p_difference_permutation": p_perm})

        print("\n" + "-" * 78)
        sig = (p_less < 0.05) and (p_rank < 0.05)
        if m.mean() > o.mean():
            print("  >>> P-c: matched source is BETTER. F2 is WITHDRAWN.")
        elif sig:
            print("  >>> P-a: F2 STANDS. The anatomy-matched source is worse than")
            print("      mismatched sources, and its last-place rank is not chance.")
        else:
            print("  >>> P-b: direction holds but significance does not.")
            print("      DOWNGRADE §2.18 to 'no matched advantage detected'.")
            print("      Remove 'the matched source is worst' from the paper.")
        print("-" * 78 + "\n")

    if rows:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cfg.out, index=False)
        print(f"written to {cfg.out}")
    print("\nSTANDING LIMITATION, whatever the p-value: FA-shoulder has 98 usable")
    print("images and is not runnable, so this contrast exists for ONE FracAtlas")
    print("target, with no ceiling and a coarse anatomy flag.")


if __name__ == "__main__":
    main()
