"""Build every number, table and figure of the paper from the result files.

Reads the result CSVs registered in ``INPUTS`` (relative to --results-root) and
the foundation-arm cells and summary under ``$FA_ROOT``, recomputes every value
the paper reports and writes them, each with a key and a note, to
``paper_numbers.json`` together with the tables (.tex and .csv) and figures.
Every input is recorded with its SHA-256. Values that a previous run wrote are
re-checked, so an unintended change in any input or computation is reported.

Usage:
    python make_paper_outputs.py --results-root ../results --out ./paper
    FA_ROOT=../foundation_arm python make_paper_outputs.py --results-root ../results --out ./paper
    python make_paper_outputs.py --check          # compare against a previous run only

CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import datetime as _dt
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

TOL = 1e-6

# ------------------ inputs ---------------------
# relative to --results-root; `required` False means the paper degrades
# gracefully without it (a figure is skipped, not silently faked).
INPUTS = {
    "probe_wrist":    ("transfer/readout_probe_XR_WRIST_merged.csv", True),
    "probe_hand":     ("transfer/readout_probe_XR_HAND.csv", True),
    "probe_humerus":  ("transfer/readout_probe_XR_HUMERUS.csv", True),
    "angle":          ("transfer/readout_angle_XR_WRIST.csv", True),
    "diagonal":       ("transfer/appearance_diagonal.csv", True),
    "zeroshot":       ("transfer/graz_zeroshot.csv", True),
    "graz_ceiling":   ("transfer/graz_initial_only_ceiling.csv", True),
    "fracatlas":      ("transfer/fracatlas_transfer.csv", True),
    "frac_stats":     ("transfer/fracatlas_match_stats.csv", True),
    "source_models":        ("source_models/anatomy_transfer_arms_merged.csv", True),
    "step1":          ("source_models/joint_encoder_vs_ceiling.csv", True),
    "src_prep":       ("transfer/source_preprocessing.csv", False),
    "cohorts":        ("transfer/cohort_counts.csv", True),
    "bitdepth":       ("transfer/bitdepth_audit.csv", False),
    "creg_oracle":    ("transfer/oracle_readout_cv_XR_WRIST.csv", False),
    "creg_probe":     ("transfer/readout_penalty_sweep_XR_WRIST.csv", False),
    "shrink":         ("transfer/readout_shrinkage_XR_WRIST.csv", False),
    "frac_budget":    ("transfer/fracatlas_budget_cells_XR_WRIST_XR_HAND.csv", False),
    "mfxr":           ("transfer/mfxr_external_XR_WRIST.csv", False),
    "lam_sched":      ("transfer/birs_schedule.csv", False),
    "graz_labels":    ("transfer/graz_label_crosstab.csv", False),
    "rr_harm":        ("transfer/harm_analyses/harm_magnitude.csv", False),
    "rr_margin":      ("transfer/harm_analyses/harm_with_margin.csv", False),
    "rr_loao":        ("transfer/harm_analyses/loo_all_policies.csv", False),
    "rr_worst":       ("transfer/harm_analyses/worst_loss_cells.csv", False),
    "l2sp":           ("transfer/l2sp_comparator_XR_WRIST.csv", False),
    "shortcut":       ("layerwise/shortcut_reliance_matrix.csv", True),
    "layerwise":      ("layerwise/layerwise_cells.csv", False),
    "label_audit":    ("transfer/graz_label_audit.csv", False),
    "easy":           ("transfer/graz_indomain_audit.csv", False),
    "strata":         ("transfer/score_strata.csv", False),
    "confounds":      ("transfer/graz_confound_audit.csv", False),
    "bootstrap":      ("transfer/bootstrap_ci.csv", False),
}

# What each input is for. Printed when a file is missing, so the failure names
# the result that would be lost rather than only the filename.
SUPPLIES = {
    "probe_wrist":  "label-budget curves and break-even, wrist source (Fig 1, Table 2)",
    "probe_hand":   "label-budget curves, hand source",
    "probe_humerus": "label-budget curves, humerus source",
    "angle":        "oracle decomposition, encoder residual (Table 1, Fig 2)",
    "diagonal":     "the diagonal null of the projection instrument (Fig 3)",
    "zeroshot":     "the preprocessing arms, including the histogram-matching null",
    "graz_ceiling": "GRAZ initial-exam ceiling per backbone",
    "fracatlas":    "anatomy match across a dataset change (Table 4)",
    "frac_stats":   "the FracAtlas within-target permutation test",
    "source_models":      "target ceilings and Stage A zero-shot anchors",
    "step1":        "the anatomy-gating null (joint encoder vs the 7 specialists)",
    "src_prep":     "the source training pool (cross-check against cohort_counts)",
    "cohorts":      "cohort flow accounting: exclusions, leakage removal, source pool",
    "bitdepth":     "evidence for the bit-depth conversion choice in Methods",
    "creg_oracle":  "PRIMARY oracle decomposition at the cross-validated penalty",
    "creg_probe":   "PRIMARY budget curves at the cross-validated penalty",
    "shortcut":     "AUC dependence of the projection delta, i.e. why calipers",
    "layerwise":    "depth-wise fine-tuning, i.e. the evidence that a reference is not a bound",
    "label_audit":  "appearance separability, cast detectability",
    "easy":         "metadata-only and text-geometry baselines on GRAZ",
    "strata":       "GRAZ initial vs follow-up composition (Fig 4)",
    "confounds":    "age bands, native-threshold sensitivity, C3 projection",
    "bootstrap":    "patient-clustered intervals (bootstrap_intervals.py)",
}

BACKBONE_LABEL = {
    "densenet121": "DenseNet121 (CNN, BatchNorm)",
    "convnext_tiny": "ConvNeXt-T (CNN, LayerNorm)",
    "swin_tiny_patch4_window7_224": "Swin-T (Transformer, LayerNorm)",
}
# colourblind-safe (Okabe-Ito)
CB = {"densenet121": "#0072B2", "convnext_tiny": "#D55E00",
      "swin_tiny_patch4_window7_224": "#009E73",
      "graz": "#CC79A7", "grey": "#666666", "light": "#BBBBBB"}


# ------------------------------------------------------------- machinery ----

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "NO_GIT"


class Numbers:
    """Every scalar the paper may quote, at full precision."""

    def __init__(self) -> None:
        self.d: Dict[str, object] = {}

    def set(self, key: str, value, note: str = "", unit_of_aggregation: str = ""):
        if key in self.d:
            raise KeyError(f"duplicate key {key!r} — names must be unique")
        v = (float(value) if isinstance(value, (int, float, np.floating,
                                                np.integer)) else value)
        self.d[key] = {"value": v, "note": note,
                       "unit_of_aggregation": unit_of_aggregation}
        return v


def style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.1,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"  [fig] {name}.pdf / .png")


# --------------------------------------------------------------- loading ----

def load(root: Path) -> tuple:
    """Load every registered input.

    The nominal relative path is tried first. If it is absent, the file is
    looked up by basename anywhere under the results root, because these
    files are written by several scripts and do not all land in the same
    subdirectory. A search that finds exactly one match is used and the
    RESOLVED path goes into provenance, so the JSON always records where the
    bytes actually came from. Two or more matches is an error, not a coin
    toss: silently picking one would be the worst possible outcome.
    """
    data, prov, missing = {}, {}, []
    for key, (rel, required) in INPUTS.items():
        p = root / rel
        note = ""
        if not p.exists():
            hits = sorted(q for q in root.rglob(Path(rel).name) if q.is_file())
            if len(hits) > 1:
                print(f"!! {key}: {len(hits)} files named {Path(rel).name} "
                      f"under the results root:")
                for q in hits:
                    print(f"     {q.relative_to(root)}")
                print("   Refusing to guess. Remove or rename the duplicates.")
                raise SystemExit(2)
            if len(hits) == 1:
                p = hits[0]
                note = f"  [found at {p.relative_to(root)}]"
            else:
                (missing.append((key, rel)) if required
                 else print(f"  [optional] absent: {rel}"))
                continue
        data[key] = pd.read_csv(p, comment="#", low_memory=False)
        prov[key] = {"path": str(p), "relative": str(p.relative_to(root)),
                     "nominal": rel, "resolved_by_search": bool(note),
                     "sha256": sha256(p), "rows": int(len(data[key]))}
        print(f"  [ok] {rel}  ({len(data[key])} rows){note}")
    return data, prov, missing


def ceilings(source_models: pd.DataFrame) -> pd.Series:
    return (source_models[source_models.condition == "ceiling"]
            .groupby(["target", "backbone"]).auc_study.mean())


def graz_ceiling_map(data: Dict[str, pd.DataFrame]) -> tuple:
    """GRAZ initial-exam ceiling per backbone.

    graz_initial_only_ceiling is the primary source. Some backbones were added to the study
    after that file was first written, so any backbone missing from it falls
    back to the `ceiling` column of the angle CSV, which measures the same
    quantity on the same subset. The substitution is RECORDED, never silent:
    a mixed provenance that is written down is fine, one that is not is the
    exact failure this whole file exists to prevent.
    """
    primary: Dict[str, float] = {}
    if "graz_ceiling" in data:
        g = data["graz_ceiling"]
        g = g[g.get("score_label", "label_abnormal") == "label_abnormal"]
        primary = g.groupby("backbone").zs_auc_study.mean().to_dict()

    fallback: Dict[str, float] = {}
    if "angle" in data:
        a = data["angle"]
        a = a[a["analysis"].isna()] if "analysis" in a else a
        gz = a[a.target.astype(str).str.startswith("GRAZ")]
        if len(gz):
            fallback = gz.groupby("backbone").ceiling.mean().to_dict()

    out, src = {}, {}
    for bb in set(primary) | set(fallback):
        if bb in primary:
            out[bb], src[bb] = primary[bb], "graz_initial_only_ceiling"
        else:
            out[bb], src[bb] = fallback[bb], "readout_angle (FALLBACK)"
    for bb, s in sorted(src.items()):
        if "FALLBACK" in s:
            print(f"  [note] GRAZ ceiling for {bb} taken from the angle CSV "
                  f"(absent from graz_initial_only_ceiling)")
    return out, src


def probe_cells(data: Dict[str, pd.DataFrame], ceil: pd.Series,
                graz_ceil: Dict[str, float]) -> pd.DataFrame:
    """One row per (source, target, backbone, seed) with zero-shot / n=500 / gap."""
    frames = []
    for key, src in (("probe_wrist", "XR_WRIST"), ("probe_hand", "XR_HAND"),
                     ("probe_humerus", "XR_HUMERUS")):
        if key not in data:
            continue
        d = data[key].copy()
        if "source" not in d.columns:
            d["source"] = src
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    out = []
    for (s, t, bb, sd), g in d.groupby(["source", "target", "backbone", "seed"]):
        z = g.loc[g.budget == 0, "auc_study"]
        p5 = g.loc[g.budget == g.budget.max(), "auc_study"]
        if z.empty or p5.empty:
            continue
        zs, pv = float(z.mean()), float(p5.mean())
        c = (graz_ceil.get(bb, np.nan) if str(t).startswith("GRAZ")
             else ceil.get((t, bb), np.nan))
        if not np.isfinite(c):
            continue
        out.append({"source": s, "target": t, "backbone": bb, "seed": sd,
                    "zeroshot": zs, "probe_max": pv, "ceiling": c,
                    "gap": c - zs, "gain": pv - zs,
                    "recovery": (pv - zs) / (c - zs) if (c - zs) >= 0.10 else np.nan,
                    "arm": "dataset" if str(t).startswith("GRAZ") else "anatomy"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures ---

def fig_budget(creg_probe, out: Path, N: Numbers) -> None:
    """Budget curves at the CV-chosen penalty, wrist source.

    This used to draw from the readout_probe probe files, which are the SUPERSEDED
    fixed-penalty run, and to pool three sources while the caption claimed the
    wrist source alone. The figure therefore contradicted Table 3: it put the
    dataset gain at +0.054 crossing near 100 where the table says +0.081
    crossing at 50, and it showed a POSITIVE mean at 10 studies where the text
    says re-fitting hurts there.

    No gap gate here. The 0.10 gate exists because a recovery FRACTION is
    unstable on a small gap. This panel plots a raw AUC GAIN, which is well
    defined at any gap, and gating it would silently drop forearm from a figure
    whose companion table reports forearm.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _, d = _split_creg(creg_probe)
    zs = (creg_probe[creg_probe.budget == 0]
          .groupby(["target", "backbone"]).auc_study.mean())
    budgets = sorted(b for b in d.budget.unique() if b > 0)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, arm, title in ((axes[0], "anatomy", "Anatomy shift (within MURA)"),
                           (axes[1], "dataset", "Dataset shift (MURA wrist to GRAZPEDWRI-DX)")):
        sub = d[d.target.str.startswith("GRAZ") == (arm == "dataset")]
        curves = []
        for (t, bb), g in sub.groupby(["target", "backbone"]):
            if (t, bb) not in zs.index:
                continue
            z = zs.loc[(t, bb)]
            y = [g.loc[g.budget == b, "auc_study"].mean() - z for b in budgets]
            ax.plot(budgets, y, color=CB["light"], lw=0.7, alpha=0.9, zorder=1)
            curves.append(y)
        if curves:
            m = np.nanmean(np.array(curves, dtype=float), axis=0)
            col = CB["graz"] if arm == "dataset" else CB["grey"]
            ax.plot(budgets, m, color=col, lw=2.0, zorder=3,
                    label=f"mean of {len(curves)} target-backbone curves")
            # Renamed from budget_curves_* / budget_mean_gain_*. Those names
            # held the same figure drawn from the fixed-penalty probe over three
            # sources. This figure is the nested sweep over the wrist source, so
            # the quantity behind the name changed. Reusing the name would have
            # been a silent redefinition, and would have tripped the drift guard
            # below every run for a reason that is not drift.
            N.set(f"nest_budget_curves_{arm}_n", float(len(curves)),
                  "curves averaged in the figure: one per (target, backbone), "
                  "seeds and resamples averaged within each. Wrist source, "
                  "CV-chosen penalty. NOT the same count as the per-cell totals "
                  "quoted in the text.")
            N.set(f"nest_budget_mean_gain_{arm}_max", m[-1],
                  f"mean AUC gain over zero-shot at n={budgets[-1]} labelled studies",
                  "(target, backbone) curve, wrist source, CV penalty")
        ax.axhline(0, color="k", lw=0.6, ls=(0, (4, 3)), zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        # A log axis puts minor ticks at 20, 30, 40 ... which fall between the
        # budgets we actually sampled and appear as an uneven number of marks
        # per interval: one between 10 and 25, four between 50 and 100. They
        # imply budgets that were never run, so remove them.
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlabel("labelled target studies")
        ax.set_title(title)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("study AUC gain over zero-shot")
    fig.tight_layout()
    save(fig, out, "fig_label_budget_curves")
    plt.close(fig)


def fig_decomposition(ang: pd.DataFrame, out: Path, N: Numbers,
                      creg: pd.DataFrame = None) -> None:
    """Gap decomposition. Uses the cross-validated penalty when the sweep is
    present, so the figure cannot disagree with the table built from the same
    rows. Plotted at C = 1 the elbow bar reads 0.052 against a table value of
    0.028, and forearm 0.054 against 0.003."""
    import matplotlib.pyplot as plt
    if creg is not None:
        _, m = _split_creg(creg)
        if not len(m):
            m = None
    else:
        m = None
    if m is None:
        m = ang[ang.get("analysis").isna()] if "analysis" in ang else ang
    t = (m.groupby("target")
         .agg(zeroshot=("zeroshot", "mean"),
              oracle=("oracle_readout", "mean"),
              ceiling=("ceiling", "mean"),
              readout=("readout_recoverable", "mean"),
              encoder=("encoder_residual", "mean"))
         .sort_values("encoder"))
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    y = np.arange(len(t))
    # Not "irreducible": the reference is not an upper bound (fine-tuned
    # configurations can beat it).
    # The residual is what a refitted linear head on a FROZEN encoder does not
    # recover, which is a statement about one method, not about all methods.
    ax.barh(y, t.readout, color=CB["grey"], label="recovered by refitting the readout")
    ax.barh(y, t.encoder, left=t.readout, color=CB["graz"],
            label="encoder residual (not recovered by a linear readout)")
    ax.set_yticks(y)
    # "Graz" is never defined in the text; spell the dataset out on the axis.
    ax.set_yticklabels([("GRAZPEDWRI-DX (dataset)" if str(s).startswith("GRAZ")
                         else str(s).replace("XR_", "").title())
                        for s in t.index])
    # "ceiling" is the internal name; the paper calls it the reference model,
    # because it is not an upper bound.
    ax.set_xlabel("study AUC gap, zero-shot to reference model")
    # The legend used to sit at lower right, on top of the forearm row: its
    # opaque frame covered the end of the shortest bar and made the grey
    # segment look unfilled. Put it outside the data area entirely, so no
    # future ordering of the bars can hide one behind it.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=1, frameon=False, handlelength=1.4, fontsize=7)
    # barh centres bars on the tick, so the first and last extend half a bar
    # height beyond it and the axes line clips them. Pad the limits.
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.margins(x=0.02)
    for i, (_, r) in enumerate(t.iterrows()):
        ax.text(r.readout + r.encoder + 0.004, i, f"{r.encoder:.3f}",
                va="center", fontsize=6.5, color=CB["graz"])
    fig.tight_layout()
    save(fig, out, "fig_gap_decomposition")
    plt.close(fig)

    an = m[~m.target.astype(str).str.startswith("GRAZ")]
    gz = m[m.target.astype(str).str.startswith("GRAZ")]
    if len(gz):
        e_g = N.set("encoder_residual_graz", gz.encoder_residual.mean(),
                    "ceiling - oracle, GRAZ initial-exam only",
                    "cell (backbone x seed) of ONE dataset pair; n=1 on the shift axis")
        e_a = N.set("encoder_residual_anatomy", an.encoder_residual.mean(),
                    "ceiling - oracle, cross-anatomy mean", "cell, 6 targets")
        worst = an.groupby("target").encoder_residual.mean().max()
        N.set("encoder_residual_worst_anatomy", worst, "largest cross-anatomy residual",
              "target")
        N.set("encoder_residual_ratio_to_mean", e_g / e_a, "GRAZ / cross-anatomy mean")
        N.set("encoder_residual_ratio_to_worst", e_g / worst,
              "GRAZ / WORST anatomy — the conservative statistic to quote")
        cons = gz.groupby("backbone").encoder_residual.mean()
        N.set("encoder_residual_graz_exceeds_all_anatomy_backbones",
              bool((cons > an.groupby("backbone").encoder_residual.mean().max()).all()),
              "does GRAZ exceed every cross-anatomy backbone mean")
        N.set("encoder_residual_n_backbones", int(gz.backbone.nunique()))
        N.set("encoder_residual_n_seeds", int(gz.seed.nunique()))
        # robustness: is a larger residual just what a higher ceiling gives?
        from scipy import stats
        a = an.dropna(subset=["ceiling", "encoder_residual"])
        rho, p = stats.spearmanr(a.ceiling, a.encoder_residual)
        N.set("ceiling_vs_residual_rho_within_anatomy", rho,
              "cell-level; NEGATIVE means higher ceilings go with SMALLER residuals, "
              "i.e. against the confound", "cell")
        N.set("ceiling_vs_residual_p_within_anatomy", p)


def fig_diagonal(diag: pd.DataFrame, out: Path, N: Numbers) -> None:
    import matplotlib.pyplot as plt
    t = diag.groupby("anatomy").delta_study.agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    y = np.arange(len(t))
    ax.errorbar(t["mean"], y, xerr=t["std"], fmt="o", ms=3.2, lw=0,
                elinewidth=0.8, color=CB["densenet121"], capsize=2)
    ax.axvline(0, color="k", lw=0.7, ls=(0, (4, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("XR_", "").title() for s in t.index])
    ax.set_xlabel("projection $\\Delta$ study AUC at ZERO domain shift")
    # Rows are sorted by mean, so the most negative anatomy is at the bottom
    # and its error bar runs into the bottom-left corner where this label used
    # to sit. The top rows sit close to zero, so the upper left is clear.
    ax.text(0.02, 0.94, "zero is not the null", transform=ax.transAxes,
            fontsize=6.5, color=CB["grey"], va="top")
    ax.set_ylim(-0.7, len(t) - 0.3)
    fig.tight_layout()
    save(fig, out, "fig_diagonal_null")
    plt.close(fig)
    N.set("diagonal_null_mean", diag.delta_study.mean(),
          "mean projection delta with NO domain shift", "cell (anatomy x backbone x seed)")
    N.set("diagonal_null_sd", diag.delta_study.std(ddof=1))
    N.set("diagonal_null_n_negative", int((diag.delta_study < 0).sum()))
    N.set("diagonal_null_n_cells", int(len(diag)))
    if "perm_delta_study_mean" in diag:
        N.set("permutation_null_mean", diag.perm_delta_study_mean.mean(),
              "how much the permutation null absorbs — near zero means it does no work")


def fig_graz_composition(strata, easy, out: Path, N: Numbers) -> None:
    """Follow-up composition. Skipped cleanly if the audit CSVs are absent."""
    if strata is None:
        print("  [skip] fig4 — score_strata.csv absent")
        return
    import matplotlib.pyplot as plt
    s = strata[strata.get("score_label", "label_abnormal") == "label_abnormal"]
    # Restrict to the PRIMARY preprocessing arm. Pooling all four arms here
    # while the caption quotes the standard arm makes the figure disagree with
    # its own caption by 0.008 AUC.
    if "prep" in s.columns:
        s = s[s.prep == "standard"]
    piv = s.pivot_table(index="stratum", columns="arm", values="auc_study",
                        aggfunc="mean")
    piv.index = [{"all": "full pool", "meta_initial_exam=1": "initial exams",
                  "meta_initial_exam!=1": "follow-up exams"}.get(i, i)
                 for i in piv.index]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    piv.plot.bar(ax=ax, color=[CB["grey"], CB["graz"]], width=0.7, rot=0,
                 legend=piv.shape[1] > 1)
    # 0.5, not 0, is the floor of this axis on purpose: an AUC of 0.5 is
    # chance, so the region below it is not performance a classifier can
    # meaningfully occupy and giving it half the panel would flatten the very
    # difference the figure exists to show. A truncated bar axis still has to
    # be declared rather than left for the reader to notice, so the label says
    # where it starts and why.
    ax.set_ylabel("study AUC (axis starts at 0.5, chance)")
    ax.set_xlabel("")
    ax.set_ylim(0.5, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=6.5, color=CB["grey"])
    fig.tight_layout()
    save(fig, out, "fig_graz_composition")
    plt.close(fig)


# ================================================================ numbers ====
# Everything below emits scalars only. Figures are above. The split is
# deliberate: a figure that fails to render must not take the numbers with it.

def _sklearn_version() -> str:
    """Recorded here because the run-time provenance headers do not capture it,
    and the oracle and probe heads are scikit-learn logistic regressions."""
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "NOT_INSTALLED"


def tkey(t: str) -> str:
    """Stable short name for a target, used in every JSON key."""
    t = str(t)
    if t.startswith("GRAZ"):
        return "graz"
    return t.replace("XR_", "").lower()


def numbers_oracle(ang: pd.DataFrame, out: Path, N: Numbers) -> None:
    """Table 1. Per-target zero-shot / oracle / ceiling / residual.

    The readout FRACTION is gated at gap >= 0.10 per cell, so the number of
    contributing cells differs by target. That count is emitted beside every
    fraction; a fraction resting on 2 cells must not be read like one resting
    on 15.
    """
    m = ang[ang["analysis"].isna()] if "analysis" in ang else ang
    rows = []
    for t, g in m.groupby("target"):
        k = tkey(t)
        n_gated = int(g.readout_frac.notna().sum())
        rec = {
            "target": t,
            "zeroshot": g.zeroshot.mean(),
            "oracle": g.oracle_readout.mean(),
            "ceiling": g.ceiling.mean(),
            "gap": g.ceiling.mean() - g.zeroshot.mean(),
            "readout_recoverable": g.readout_recoverable.mean(),
            "encoder_residual": g.encoder_residual.mean(),
            "readout_frac": g.readout_frac.mean(),
            "n_cells": int(len(g)),
            "n_cells_gated": n_gated,
        }
        rows.append(rec)
        N.set(f"oracle_{k}_zeroshot", rec["zeroshot"],
              "frozen source encoder, source head, no target labels", "cell (backbone x seed)")
        N.set(f"oracle_{k}_oracle", rec["oracle"],
              "frozen encoder, linear head fitted on ALL target training data", "cell")
        N.set(f"oracle_{k}_ceiling", rec["ceiling"],
              "target-trained reference model; a reference point, NOT an upper bound", "cell")
        N.set(f"oracle_{k}_gap", rec["gap"], "ceiling - zeroshot", "cell")
        N.set(f"oracle_{k}_encoder_residual", rec["encoder_residual"],
              "ceiling - oracle: the AUC no amount of target labelling buys back", "cell")
        N.set(f"oracle_{k}_n_cells", rec["n_cells"])
        N.set(f"oracle_{k}_n_cells_gated", n_gated,
              "cells with gap >= 0.10; only these contribute to the readout fraction")
        # The AUC means above are over ALL cells; the recovery fraction is a
        # mean of per-cell ratios over the GATED cells only. Printing both in
        # one table row makes the fraction non-derivable from its own columns:
        # for finger the row implies 0.186 while the fraction reads 0.393.
        # Emit the gated-subset AUC means as well, so the published table can
        # be built from one population throughout.
        gg = g[g.readout_frac.notna()]
        if n_gated:
            N.set(f"oracle_{k}_readout_frac", rec["readout_frac"],
                  "MEAN OF PER-CELL RATIOS over gap >= 0.10 cells", "gated cell")
            N.set(f"oracle_{k}_zeroshot_gated", gg.zeroshot.mean(),
                  "zero shot, GATED cells only", "gated cell")
            N.set(f"oracle_{k}_oracle_gated", gg.oracle_readout.mean(),
                  "oracle readout, GATED cells only", "gated cell")
            N.set(f"oracle_{k}_ceiling_gated", gg.ceiling.mean(),
                  "reference model, GATED cells only", "gated cell")
            N.set(f"oracle_{k}_gap_gated", (gg.ceiling - gg.zeroshot).mean(),
                  "gap, GATED cells only", "gated cell")
            N.set(f"oracle_{k}_encoder_residual_gated", gg.encoder_residual.mean(),
                  "encoder residual, GATED cells only", "gated cell")
            N.set(f"oracle_{k}_readout_frac_sd", gg.readout_frac.std(ddof=1),
                  "SD of the per-cell recovery fraction across gated cells; the "
                  "cells are 3 backbones x 5 seeds and are NOT independent draws",
                  "gated cell")
            if "seed" in gg.columns and gg.seed.nunique() > 1:
                N.set(f"oracle_{k}_readout_frac_seed_sd",
                      gg.groupby("seed").readout_frac.mean().std(ddof=1),
                      "SD across SEED means, which isolates seed variance from "
                      "backbone variance", "seed")
            rec.update({"zeroshot_gated": gg.zeroshot.mean(),
                        "oracle_gated": gg.oracle_readout.mean(),
                        "ceiling_gated": gg.ceiling.mean(),
                        "readout_frac_sd": gg.readout_frac.std(ddof=1)})
    tab = pd.DataFrame(rows).sort_values("encoder_residual", ascending=False)
    tab.to_csv(out / "table_oracle_decomposition.csv", index=False)
    print("  [tab] table_oracle_decomposition.csv")

    an = m[~m.target.astype(str).str.startswith("GRAZ")]
    N.set("oracle_frac_anatomy_pooled", an.readout_frac.mean(),
          "cross-anatomy oracle readout fraction, gap >= 0.10", "cell")
    N.set("oracle_frac_anatomy_n", int(an.readout_frac.notna().sum()))
    gz = m[m.target.astype(str).str.startswith("GRAZ")]
    N.set("oracle_frac_dataset_pooled", gz.readout_frac.mean(),
          "GRAZ oracle readout fraction; ONE dataset pair, n = 1 on the shift axis", "cell")
    N.set("oracle_frac_dataset_n", int(gz.readout_frac.notna().sum()))


def probe_long(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for key, src in (("probe_wrist", "XR_WRIST"), ("probe_hand", "XR_HAND"),
                     ("probe_humerus", "XR_HUMERUS")):
        if key not in data:
            continue
        d = data[key].copy()
        if "source" not in d.columns:
            d["source"] = src
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    return d.groupby(["source", "target", "backbone", "seed", "budget"],
                     as_index=False).auc_study.mean()


def numbers_budgets(data: Dict[str, pd.DataFrame], out: Path, N: Numbers) -> None:
    """Table 2. Break-even budgets, and how often re-fitting HURTS.

    Break-even is the smallest budget at which the mean beats zero-shot AND
    stays above it at every larger budget. A single lucky crossing that is
    given back at the next budget is not a break-even point.
    """
    g = probe_long(data)
    budgets = sorted(b for b in g.budget.unique() if b > 0)

    zs = (g[g.budget == 0].rename(columns={"auc_study": "zs"})
          .drop(columns="budget"))
    m = g[g.budget > 0].merge(zs, on=["source", "target", "backbone", "seed"])
    m["harm"] = m.auc_study < m.zs
    for b in budgets:
        s = m[m.budget == b]
        N.set(f"refit_harmful_frac_at_{b}", s.harm.mean(),
              f"fraction of cells where re-fitting the readout on {b} labelled "
              f"studies is WORSE than leaving the source head alone",
              "cell (source x target x backbone x seed), all cells, no gap gate")
    N.set("refit_harmful_n_cells_per_budget", int(len(m[m.budget == budgets[0]])))

    rows = []
    for (src, t), gg in g.groupby(["source", "target"]):
        base = gg.loc[gg.budget == 0, "auc_study"].mean()
        curve = {b: gg.loc[gg.budget == b, "auc_study"].mean() for b in budgets}
        be = None
        for i, b in enumerate(budgets):
            if all(curve[c] > base for c in budgets[i:]):
                be = b
                break
        rows.append({"source": src, "target": t, "zeroshot": base,
                     "break_even_studies": be if be is not None else np.nan,
                     "never": be is None,
                     **{f"auc_at_{b}": curve[b] for b in budgets}})
        if src == "XR_WRIST":
            k = tkey(t)
            N.set(f"breakeven_wrist_{k}", (float(be) if be is not None else -1.0),
                  "smallest labelled-study budget at which re-fitting beats "
                  "zero-shot and stays ahead; -1 means it never crosses over",
                  "(target) pooled over backbones and seeds")
            N.set(f"budget500_gain_wrist_{k}", curve[budgets[-1]] - base,
                  f"mean AUC change at {budgets[-1]} labelled studies", "(target)")
    tab = pd.DataFrame(rows).sort_values(["source", "target"])
    tab.to_csv(out / "table_break_even.csv", index=False)
    print("  [tab] table_break_even.csv")
    N.set("budget_max_studies", int(budgets[-1]))
    N.set("budget_grid", ", ".join(str(b) for b in budgets))


def numbers_gating_null(step1: pd.DataFrame, N: Numbers) -> None:
    """Null 1. A joint all-anatomy encoder does not lose to the specialists,
    so there is no concept conflict for anatomy conditioning to route around."""
    from scipy import stats
    for bb, g in step1.groupby("backbone"):
        b = bb.split("_")[0]
        per = g.groupby("anatomy").delta_auc_study.mean()
        N.set(f"gating_null_delta_{b}", per.mean(),
              "joint all-anatomy encoder minus that anatomy's own specialist",
              "anatomy (means over seeds)")
        n_pos, n_ns_worse = 0, 0
        for a, gg in g.groupby("anatomy"):
            d = gg.delta_auc_study.values
            t, p = stats.ttest_1samp(d, 0.0)
            worse = bool(p < 0.05 and d.mean() < 0)
            n_pos += int(d.mean() >= 0)
            n_ns_worse += int(not worse)
            N.set(f"gating_null_{tkey(a)}_delta_{b}", d.mean(), "", "anatomy")
            N.set(f"gating_null_{tkey(a)}_p_{b}", p,
                  "one-sample t test against zero across seeds", "anatomy")
        # The defensible wording is "NOT SIGNIFICANTLY WORSE", not "matches or
        # beats". Only 4 of 7 have a positive mean; 6 of 7 are not significantly
        # worse. Both counts are emitted so the sentence cannot drift.
        N.set(f"gating_null_n_positive_mean_{b}", n_pos, "of 7 anatomies")
        N.set(f"gating_null_n_not_significantly_worse_{b}", n_ns_worse,
              "of 7 anatomies; THIS is the count the manuscript may quote")
    N.set("gating_null_n_anatomies", int(step1.anatomy.nunique()))
    N.set("gating_null_n_seeds", int(step1.seed.nunique()))


def numbers_prep_arms(zshot: pd.DataFrame, N: Numbers) -> None:
    """Null 2. Making GRAZ look like MURA makes transfer worse.

    NOTE ON SCOPE. These arms were run on the FULL GRAZ pool, not on
    initial exams only. They are a within-arm contrast so the comparison
    holds, but the absolute values are inflated by follow-up composition,
    which is stated wherever they are reported.
    """
    z = zshot[zshot.score_label == "label_abnormal"]
    piv = z.pivot_table(index="prep", columns="backbone",
                        values="zs_auc_study", aggfunc="mean")
    if "standard" not in piv.index:
        return
    for prep in piv.index:
        for bb in piv.columns:
            b = bb.split("_")[0]
            N.set(f"prep_{prep}_{b}_auc", piv.loc[prep, bb],
                  "MURA-wrist to GRAZ zero-shot under this preprocessing arm; "
                  "FULL GRAZ POOL, not initial-exam only", "backbone (5 seeds)")
            if prep != "standard":
                N.set(f"prep_{prep}_{b}_delta", piv.loc[prep, bb] - piv.loc["standard", bb],
                      "change from the standard arm", "backbone")
    N.set("prep_n_arms", int(len(piv.index)))


def numbers_graz_audit(data: Dict[str, pd.DataFrame], N: Numbers) -> None:
    """The dataset-composition audit. All on the GRAZ validation split."""
    st = data.get("strata")
    if st is not None:
        base = st[(st.arm == "zeroshot") &
                  (st.get("score_label", "label_abnormal") == "label_abnormal")]
        name = {"all": "full_pool", "meta_initial_exam=1": "initial",
                "meta_initial_exam!=1": "followup"}
        # The strata file spans four preprocessing arms. The paper's primary
        # arm is `standard`; the other three are controls reported separately.
        # Pooling all four gives a different level (0.849 vs 0.857 on the full
        # pool), so BOTH are emitted under distinct names and the text
        # must say which it is quoting. Silently pooling primary with controls
        # is the same class of error as pooling two backbone counts.
        for tag, s in (("", base[base.prep == "standard"]),
                       ("_allpreps", base)):
            if not len(s):
                continue
            agg = s.groupby("stratum").agg(n=("n", "mean"),
                                           prev=("prevalence", "mean"),
                                           auc=("auc_study", "mean"))
            nbb = int(s.backbone.nunique())
            for k, v in name.items():
                if k not in agg.index:
                    continue
                if not tag:
                    # counts and prevalences are arm-independent, emit once
                    N.set(f"graz_{v}_n_images", agg.loc[k, "n"],
                          "GRAZ validation split, image count")
                    N.set(f"graz_{v}_prevalence_image", agg.loc[k, "prev"],
                          "image-level abnormal rate on this stratum")
                if not tag and "auc_image" in s.columns:
                    ai = s.groupby("stratum").auc_image.mean()
                    N.set(f"graz_{v}_zeroshot_image", ai.loc[k],
                          "IMAGE-level zero-shot AUC. The operating point in "
                          "the paper is image-level, so it must be compared "
                          "against this rather than the study-level figure.",
                          "stratum")
                N.set(f"graz_{v}_zeroshot{tag}", agg.loc[k, "auc"],
                      "MURA-wrist zero-shot on this stratum"
                      + ("" if tag else "; STANDARD preprocessing arm only"),
                      "stratum (backbone x seed)")
                N.set(f"graz_{v}_zeroshot{tag}_n_backbones", nbb,
                      "backbones contributing; the zeroshot stage was not re-run "
                      "with every backbone used elsewhere in the paper")
            if {"all", "meta_initial_exam=1"} <= set(agg.index):
                N.set(f"graz_fullpool_inflation{tag}",
                      agg.loc["all", "auc"] - agg.loc["meta_initial_exam=1", "auc"],
                      "how much evaluating on the full pool inflates the transfer number")
            if not tag and {"all", "meta_initial_exam!=1"} <= set(agg.index):
                N.set("graz_followup_image_share",
                      agg.loc["meta_initial_exam!=1", "n"] / agg.loc["all", "n"],
                      "share of GRAZ validation IMAGES that are follow-up exams")

    ez = data.get("easy")
    if ez is not None:
        for _, r in ez.iterrows():
            a = str(r["analysis"]).replace("::", "_").replace("label_", "")
            N.set(f"graz_easy_{a}", r["auc_grouped"],
                  "patient-grouped CV, no pixels unless the name says otherwise",
                  "full dataset")

    la = data.get("label_audit")
    if la is not None:
        pick = {"has_cast": "cast_detectability",
                "has_metal": "metal_detectability",
                "label_abnormal": "appearance_to_label",
                "dataset (GRAZ vs MURA-wrist)": "dataset_separability",
                "label_abnormal | has_cast=1": "label_given_cast"}
        for a, k in pick.items():
            row = la[la.analysis == a]
            if len(row):
                r = row.iloc[0]
                N.set(f"graz_{k}_auc", r["auc_grouped"],
                      "five appearance descriptors, patient-grouped CV, full coverage",
                      "image")
                N.set(f"graz_{k}_prevalence", r["prevalence"], "", "image")
                N.set(f"graz_{k}_n", int(r["n"]), "", "image")


def numbers_confounds(cf: pd.DataFrame, N: Numbers) -> None:
    """C1 to C3. All on GRAZ initial exams only."""
    age = cf[cf.analysis == "C1_age_band"]
    for _, r in age.iterrows():
        b = str(r["band"]).replace("-", "to")
        N.set(f"confound_age_{b}_auc", r["auc"],
              "GRAZ initial-exam zero-shot within this age band", "band")
        N.set(f"confound_age_{b}_n", int(r["n"]), "", "band")

    if len(age) > 2:
        from scipy import stats as _st
        order = ["0-6", "6-10", "10-13", "13-16", "16-19"]
        a = age.set_index("band").reindex([b for b in order if b in set(age.band)])
        a = a.dropna(subset=["auc"])
        if len(a) > 2:
            rho, p = _st.spearmanr(range(len(a)), a.auc.to_numpy())
            N.set("confound_age_trend_rho", rho,
                  "Spearman between age band order and zero-shot AUC. A skeletal "
                  "maturity account predicts a POSITIVE value, older children "
                  "transferring better from an adult trained encoder.", "band")
            N.set("confound_age_trend_p", p, "n = 5 bands; this cannot reject")
            N.set("confound_age_auc_min", float(a.auc.min()))
            N.set("confound_age_auc_max", float(a.auc.max()))
            N.set("confound_age_n_bands", float(len(a)))

    c2 = cf[cf.analysis == "C2_native_threshold"]
    if len(c2):
        r = c2.iloc[0]
        N.set("clinical_sensitivity_native", r["sensitivity"],
              "sensitivity of the transferred model at its OWN decision boundary "
              "(logit >= 0) on GRAZ initial exams", "pooled cells")
        N.set("clinical_specificity_native", r["specificity"],
              "specificity at the same boundary", "pooled cells")
        N.set("clinical_call_rate_native", r["call_rate"],
              "fraction called positive", "pooled cells")
        N.set("clinical_true_prevalence", r["true_prevalence"], "", "pooled cells")
        N.set("clinical_undercall", r["true_prevalence"] - r["call_rate"],
              "positive: the model UNDER-calls, the opposite of the "
              "abnormality-detector-meets-fracture-label prediction")

    c3 = cf[cf.analysis == "C3_projection"]
    if len(c3):
        r = c3.iloc[0]
        for c, nm in (("cross_raw", "anatomy_raw"), ("cross_resid", "anatomy_projected"),
                      ("graz_raw", "graz_raw"), ("graz_resid", "graz_projected")):
            N.set(f"confound_c3_{nm}", r[c],
                  "zero-shot AUC before/after projecting the appearance component out "
                  "of both sides", "pooled cells")
        N.set("confound_c3_advantage_before", r["cross_raw"] - r["graz_raw"])
        N.set("confound_c3_advantage_after", r["cross_resid"] - r["graz_resid"],
              "the cross-anatomy advantage barely shrinks, so a shared-acquisition "
              "shortcut does not explain it")


def numbers_fracatlas(fa: pd.DataFrame, out: Path, N: Numbers) -> None:
    """Anatomy match does not survive a dataset change."""
    hand = fa[fa.target.astype(str).str.contains("HAND", case=False, na=False)]
    if not len(hand):
        hand = fa
    per = (hand.groupby(["source", "matched"]).zs_auc_study
           .agg(["mean", "std", "size"]).reset_index()
           .sort_values("mean", ascending=False))
    per.to_csv(out / "table_fracatlas_sources.csv", index=False)
    print("  [tab] table_fracatlas_sources.csv")
    for _, r in per.iterrows():
        N.set(f"fracatlas_source_{tkey(r['source'])}", r["mean"],
              "MURA source anatomy evaluated on the FracAtlas hand target; "
              "evaluation only, no ceiling exists (no patient IDs)",
              "cell (backbone x seed)")
    N.set("fracatlas_n_sources", int(per.source.nunique()))
    rank = per["mean"].rank(ascending=False, method="min")
    matched_rank = -1
    if "matched" in per.columns:
        sel = rank[per["matched"].astype(bool).values]
        if len(sel):
            matched_rank = int(sel.min())
    N.set("fracatlas_matched_rank", matched_rank,
          "rank of the anatomy-matched source among all sources, 1 = best")
    N.set("fracatlas_range_low", per["mean"].min(),
          "every source is poor here; the contrast is between bad and less bad")
    N.set("fracatlas_range_high", per["mean"].max())


def numbers_shortcut(sc: pd.DataFrame, N: Numbers) -> None:
    """Why every delta comparison has to be AUC-caliper matched."""
    from scipy import stats
    cross = sc[sc.source != sc.target]
    rho, p = stats.spearmanr(cross.zs_auc_study, cross.delta_study)
    N.set("projection_delta_vs_auc_rho", rho,
          "projection delta is strongly dependent on the AUC level of the cell, "
          "which is why raw deltas may not be compared across cells",
          "cell (source x target x backbone x seed)")
    N.set("projection_delta_vs_auc_p", p)
    N.set("projection_delta_n_cross_cells", int(len(cross)))
    N.set("projection_delta_cross_mean", cross.delta_study.mean(), "", "cell")


def numbers_gaps(source_models: pd.DataFrame, N: Numbers) -> None:
    """Wrist-source zero-shot and ceiling per target, the Stage A anchors."""
    zs = source_models[(source_models.condition == "zeroshot") & (source_models.source == "XR_WRIST")]
    for t, g in zs.groupby("target"):
        if t == "XR_WRIST":
            continue
        N.set(f"stagea_wrist_{tkey(t)}_zeroshot", g.auc_study.mean(),
              "Stage A zero-shot anchor, wrist source, all backbones",
              "target (backbone x seed)")
    N.set("stagea_n_backbones", int(source_models.backbone.nunique()))
    N.set("stagea_n_seeds", int(source_models.seed.nunique()))


def numbers_diagonal_per_anatomy(diag: pd.DataFrame, out: Path, N: Numbers) -> None:
    t = diag.groupby("anatomy").delta_study.agg(["mean", "std", "size"])
    t.to_csv(out / "supp_diagonal_per_anatomy.csv")
    for a, r in t.iterrows():
        N.set(f"diagonal_{tkey(a)}_mean", r["mean"],
              "projection delta at ZERO domain shift for this anatomy", "anatomy")


def numbers_datasets(data: Dict[str, pd.DataFrame], out: Path, N: Numbers) -> None:
    """Cohort counts for the Methods section.

    MURA training counts are read from the angle run, so they are already
    NET of source-patient removal: patients shared with the wrist source have
    been dropped from every target pool. That is the pool the models actually
    saw, which is the only pool worth reporting.
    """
    rows = []
    if "source_models" in data:
        c = data["source_models"][data["source_models"].condition == "ceiling"]
        v = c.groupby("target")[["n_studies", "n_images", "target_prev"]].first()
        for t, r in v.iterrows():
            k = tkey(t)
            N.set(f"data_{k}_val_studies", r["n_studies"], "MURA official validation split")
            N.set(f"data_{k}_val_images", r["n_images"], "MURA official validation split")
            # source_models's target_prev is IMAGE-level. It was previously labelled
            # "study-level", which is wrong: for elbow it is 0.4946 per image
            # against 0.4177 per study. MURA labels are assigned per study, so
            # the two are not interchangeable in the text.
            N.set(f"data_{k}_val_prevalence_image", r["target_prev"],
                  "IMAGE-level abnormal rate; for the study-level rate use the "
                  "cohort_counts keys")
            rows.append({"cohort": t, "val_studies": r["n_studies"],
                         "val_images": r["n_images"],
                         "val_prevalence_image": r["target_prev"]})
    if "angle" in data:
        a = data["angle"]
        a = a[a["analysis"].isna()] if "analysis" in a else a
        tr = a.groupby("target")[["n_train_images", "n_train_studies"]].first()
        for t, r in tr.iterrows():
            k = tkey(t)
            N.set(f"data_{k}_train_studies", r["n_train_studies"],
                  "training pool after removing patients shared with the wrist source")
            N.set(f"data_{k}_train_images", r["n_train_images"],
                  "training pool after removing patients shared with the wrist source")
    if "graz_ceiling" in data:
        g = data["graz_ceiling"]
        g = g[g.get("score_label", "label_abnormal") == "label_abnormal"].iloc[0]
        N.set("data_graz_train_prevalence", g["train_prev"], "GRAZ initial exams only")
        N.set("data_graz_val_studies", g["n_studies"], "GRAZ initial exams only")
        N.set("data_graz_val_images", g["n_images"], "GRAZ initial exams only")
        N.set("data_graz_val_prevalence_image", g["target_prev"],
              "image-level abnormal rate, GRAZ initial exams only")
    if "fracatlas" in data:
        f = data["fracatlas"]
        h = f[f.target.astype(str).str.contains("HAND", case=False, na=False)]
        if len(h):
            r = h.iloc[0]
            N.set("data_fracatlas_hand_images", r["n_images"],
                  "FracAtlas hand pool; evaluation only, no ceiling (no patient IDs)")
            N.set("data_fracatlas_hand_prevalence", r["target_prev"], "fracture rate")
    if rows:
        pd.DataFrame(rows).to_csv(out / "supp_cohorts.csv", index=False)


def numbers_source_pool(sp: pd.DataFrame, N: Numbers) -> None:
    """What the transferred models were actually trained on.

    Without this the Methods section describes a source model whose training
    pool has no provenance path at all, which is worse than the section being
    thin.
    """
    for src in sorted(sp.source.unique()) if "source" in sp.columns else ["XR_WRIST"]:
        s = sp[sp.source == src] if "source" in sp.columns else sp
        k = tkey(src)
        for col, suffix, note in (
                ("n_train_images", "images", "images in the source training pool"),
                ("n_train_studies", "studies", "studies in the source training pool"),
                ("train_prev", "prevalence", "abnormal rate of the source training pool")):
            if col in s.columns:
                v = s[col].dropna()
                if len(v):
                    N.set(f"data_{k}_source_{suffix}", v.iloc[0], note)
                    if v.nunique() > 1:
                        raise SystemExit(
                            f"!! {col} is not constant across rows for source {src} "
                            f"({v.nunique()} distinct values). The source pool must be "
                            f"one number; refusing to pick one.")


def numbers_not_a_bound(lw: pd.DataFrame, ceil: pd.Series, N: Numbers) -> None:
    """Evidence that a target-trained reference model is not an upper bound.

    Backbone-MATCHED throughout. Comparing a fine-tune pooled over backbones
    against a single backbone's reference is the same unit-mixing error the
    statistics section promises to police, and it is easy to make here because
    the depth grid is not balanced across backbones.
    """
    if "depth" not in lw.columns or not len(ceil):
        return
    best = None
    for (t, bb, d), g in lw.groupby(["target", "backbone", "depth"]):
        if (t, bb) not in ceil.index or d == 0:
            continue
        delta = g.auc_study.mean() - ceil[(t, bb)]
        if best is None or delta > best[0]:
            best = (delta, t, bb, d, g.auc_study.mean(), ceil[(t, bb)], len(g))
    if best is None:
        return
    delta, t, bb, d, ft, ref, n = best
    N.set("not_a_bound_delta", delta,
          "largest amount by which depth-wise fine-tuning exceeds the target's own "
          "reference model, BACKBONE-MATCHED", "backbone x seed cells")
    N.set("not_a_bound_finetune_auc", ft, f"{t}, {bb}, depth {d}", "cell")
    N.set("not_a_bound_reference_auc", ref, f"{t}, {bb} reference model", "cell")
    N.set("not_a_bound_depth", float(d))
    N.set("not_a_bound_n_cells", float(n))
    N.set("not_a_bound_target", -1.0, f"target is {t}, backbone {bb}; see note")
    n_exceed = 0
    for (tt, bbx, dd), g in lw.groupby(["target", "backbone", "depth"]):
        if (tt, bbx) in ceil.index and dd > 0 and g.auc_study.mean() > ceil[(tt, bbx)]:
            n_exceed += 1
    N.set("not_a_bound_n_configs_exceeding", float(n_exceed),
          "how many (target, backbone, depth) configurations beat their own reference")


LINEAR_STAGES = ["raw", "readable", "descriptors", "full", "preleak",
                 "leak_removed", "postleak"]
# `initial_only` and `followup_only` are SIBLING subsets of the same parent,
# not further steps in the chain. Ranking them in a sequence and demanding a
# non-increasing count would be checking the wrong arithmetic: what must hold
# is that the two branches partition the parent, not that one contains the
# other.
BRANCH_STAGES = ["initial_only", "followup_only"]
STAGE_ORDER = LINEAR_STAGES + BRANCH_STAGES

# The cohort file carries the stage in two places: a coarse `stage` column and
# a fine-grained suffix on `cohort_id` (`..._preleak`, `..._postleak`,
# `..._initial`). The suffix is the one that distinguishes rows, so it wins,
# and the base id becomes the grouping key. Any disagreement with the `stage`
# column is printed rather than resolved silently.
SUFFIX_ALIAS = {
    "raw": "raw", "readable": "readable", "descriptors": "descriptors",
    "full": "full", "preleak": "preleak", "postleak": "postleak",
    "leak_removed": "leak_removed", "leakremoved": "leak_removed",
    "initial": "initial_only", "initial_only": "initial_only",
    "followup": "followup_only", "followup_only": "followup_only",
    "follow_up": "followup_only",
}


def ckey(c: str) -> str:
    """Key for a cohort id. Unlike tkey it does NOT collapse anything to
    'graz', because two cohorts such as GRAZ_train and GRAZ_valid would then
    map to the same key and collide."""
    return str(c).strip().replace("XR_", "").lower()


def _split_stage_suffix(cid: str) -> tuple:
    """Return (base_id, stage) if the id ends in a known stage token."""
    low = str(cid).strip().lower()
    for suf in sorted(SUFFIX_ALIAS, key=len, reverse=True):
        if low.endswith("_" + suf):
            return str(cid)[: -(len(suf) + 1)], SUFFIX_ALIAS[suf]
    return str(cid).strip(), None


def _cohort_frame(cc: pd.DataFrame) -> pd.DataFrame:
    """Normalise the cohort file into (cohort, stage, ...) regardless of layout.

    Two layouts are accepted, because both were described:
      A. an explicit `stage` column, cohort ids carrying no stage suffix
      B. no `stage` column, the stage baked into the cohort id as a suffix
    Anything else is a hard stop: guessing at a schema risks a wrong
    denominator.
    """
    if len(cc.columns) == 1 and "\t" in str(cc.columns[0]):
        raise SystemExit(
            "!! cohort_counts.csv parsed as a single column and that column name "
            "contains tab characters.\n"
            "   The file is tab-separated, or it was round-tripped through a "
            "spreadsheet.\n"
            f"   column found: {cc.columns[0]!r}\n"
            "   Check with `head -3` and re-export as comma-separated.")

    cols = {c.strip().lower(): c for c in cc.columns}
    idc = (cols.get("cohort_id") or cols.get("cohort") or cols.get("name")
           or cols.get("id"))
    if idc is None:
        raise SystemExit(
            "!! cohort_counts.csv has no `cohort_id` (or `cohort`/`name`/`id`) column.\n"
            f"   columns found: {list(cc.columns)}")
    d = cc.rename(columns={c: c.strip().lower() for c in cc.columns})
    d = d.rename(columns={idc.strip().lower(): "cohort"}).copy()
    d["cohort"] = d["cohort"].astype(str).str.strip()
    d["cohort_id_raw"] = d["cohort"]

    col_stage = (d["stage"].astype(str).str.strip().str.lower()
                 if "stage" in d.columns else None)
    split = d["cohort"].map(_split_stage_suffix)
    d["cohort"] = [b for b, _ in split]
    suf_stage = pd.Series([s for _, s in split], index=d.index)

    if suf_stage.notna().any():
        d["stage"] = suf_stage.where(suf_stage.notna(),
                                     col_stage if col_stage is not None else None)
        if col_stage is not None:
            mism = d[suf_stage.notna() & (suf_stage != col_stage)]
            if len(mism):
                print("  [cohorts] id suffix and `stage` column differ on "
                      f"{len(mism)} row(s); the SUFFIX is used. Reconciliation:")
                rec = (mism.assign(from_suffix=suf_stage[mism.index],
                                   from_column=col_stage[mism.index])
                       [["cohort_id_raw", "from_suffix", "from_column"]])
                for _, r in rec.iterrows():
                    print(f"            {r.cohort_id_raw:<38} "
                          f"suffix={r.from_suffix:<14} column={r.from_column}")
    elif col_stage is not None:
        d["stage"] = col_stage
    else:
        pat = "|".join(STAGE_ORDER)
        m = d["cohort"].str.extract(rf"_({pat})$", expand=False)
        if m.isna().any():
            bad = d.loc[m.isna(), "cohort"].tolist()[:5]
            raise SystemExit(
                "!! cohort_counts.csv has no `stage` column and these cohort ids "
                f"do not end in a known stage: {bad}\n"
                f"   known stages: {STAGE_ORDER}")
        d["stage"] = m
        d["cohort"] = d["cohort"].str.replace(rf"_({pat})$", "", regex=True)

    unknown = sorted(set(d["stage"]) - set(STAGE_ORDER))
    if unknown:
        raise SystemExit(f"!! unknown stage(s) in cohort_counts.csv: {unknown}\n"
                         f"   known stages: {STAGE_ORDER}")
    if "n_images" not in d.columns:
        raise SystemExit("!! cohort_counts.csv has no `n_images` column.\n"
                         f"   columns found: {list(cc.columns)}")
    if d.duplicated(["cohort", "stage"]).any():
        dup = d[d.duplicated(["cohort", "stage"], keep=False)][["cohort", "stage"]]
        raise SystemExit("!! duplicate (cohort, stage) rows:\n"
                         f"{dup.to_string(index=False)}")
    return d


def numbers_bitdepth(bd: pd.DataFrame, N: Numbers) -> None:
    """Evidence for the percentile rescaling choice.

    A direct 8-bit cast does not saturate: no image in the sample had more than 0.41 per cent of pixels
    driven to white. What varies between images is how much of the 16-bit range
    they occupy, and that variation is the actual reason for rescaling.
    """
    N.set("bitdepth_n_sampled", float(len(bd)), "random sample of 16-bit images")
    N.set("bitdepth_span_median", float(bd.span.median()),
          "median fraction of the 16-bit range the stored intensities occupy")
    N.set("bitdepth_span_min", float(bd.span.min()),
          "worst case; a direct cast would leave the rest of the 8-bit scale unused")
    N.set("bitdepth_crushed_median", float(bd.crushed.median()),
          "median fraction of pixels a direct cast sends to pure black")
    N.set("bitdepth_crushed_max", float(bd.crushed.max()))
    N.set("bitdepth_saturated_max", float(bd.saturated.max()),
          "largest fraction driven to pure white ANYWHERE in the sample; the "
          "manuscript originally claimed saturation and this refutes it")
    N.set("bitdepth_absdiff_median", float(bd.mean_abs_diff.median()),
          "median absolute difference between the two conversions, in 8-bit levels")
    N.set("bitdepth_frac_differ_gt10", float((bd.mean_abs_diff > 10).mean()),
          "fraction of images where the two conversions differ by >10 levels")
    from scipy import stats as _st
    N.set("bitdepth_corr_span_absdiff",
          float(_st.pearsonr(bd.span, bd.mean_abs_diff)[0]),
          "the disagreement tracks how much of the range is used, not damage")
    N.set("bitdepth_corr_crushed_absdiff",
          float(_st.pearsonr(bd.crushed, bd.mean_abs_diff)[0]),
          "near zero, so crushing is not what drives the disagreement")


def numbers_cohorts(cc: pd.DataFrame, out: Path, N: Numbers) -> None:
    """Cohort flow: raw -> readable -> descriptors -> leak removed -> initial only.

    This is the CONSORT-style accounting for the paper. Two Methods numbers
    that previously had no file behind them come from here: the FracAtlas
    exclusion count with its label breakdown, and the per-target leakage
    removal rate.
    """
    d = _cohort_frame(cc)
    d.sort_values(["cohort", "stage"]).to_csv(out / "supp_cohort_flow.csv",
                                              index=False)
    print("  [tab] supp_cohort_flow.csv")

    # The linear chain must not grow as filtering proceeds. A rise means the
    # stages were written in the wrong order or a filter was applied twice.
    rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    for coh, g in d.groupby("cohort"):
        lin = g[g.stage.isin(LINEAR_STAGES)].sort_values(
            "stage", key=lambda s: s.map(rank))
        n = lin["n_images"].to_numpy(dtype=float)
        if len(n) > 1 and (np.diff(n) > 0).any():
            raise SystemExit(
                f"!! cohort {coh}: image count INCREASES along the filtering "
                f"chain.\n{lin[['stage', 'n_images']].to_string(index=False)}")

        # Branch subsets must partition whatever they were cut from.
        br = g[g.stage.isin(BRANCH_STAGES)]
        if not len(br):
            continue
        parent = float(lin["n_images"].iloc[-1]) if len(lin) else None
        if parent is None:
            print(f"  [cohorts] {coh}: branch subsets present with no linear "
                  f"parent row, partition not checked")
            continue
        for _, r in br.iterrows():
            if float(r["n_images"]) > parent:
                raise SystemExit(
                    f"!! cohort {coh}: the {r['stage']} subset has "
                    f"{float(r['n_images']):.0f} images but its parent has "
                    f"{parent:.0f}. A subset cannot exceed what it came from.")
        if set(br.stage) == set(BRANCH_STAGES):
            tot = float(br["n_images"].sum())
            if abs(tot - parent) > 0.5:
                raise SystemExit(
                    f"!! cohort {coh}: initial_only + followup_only = {tot:.0f} "
                    f"but the parent stage has {parent:.0f} images. The two "
                    f"subsets do not partition the pool. If that is intentional "
                    f"(some images belong to neither), say so and the check will "
                    f"be relaxed; otherwise one of the three numbers is wrong.")
            N.set(f"cohort_{ckey(coh)}_partition_verified", 1.0,
                  "initial_only + followup_only equals the parent image count")

    for _, r in d.iterrows():
        k = f"cohort_{ckey(r['cohort'])}_{r['stage']}"
        N.set(f"{k}_images", r["n_images"], "cohort flow accounting")
        for col, nm in (("n_studies", "studies"), ("n_patients", "patients"),
                        ("prevalence_image", "prevalence_image"),
                        ("prevalence_study", "prevalence_study")):
            if col in d.columns and pd.notna(r.get(col)):
                N.set(f"{k}_{nm}", r[col], "cohort flow accounting")

    # ---- FracAtlas exclusions -------------------------------------------
    # Selected by the `dataset` column when present. Matching on the cohort id
    # string is a guess about naming; the dataset column is a fact.
    if "dataset" in d.columns:
        fr = d[d.dataset.astype(str).str.lower().str.contains("frac", na=False)]
    else:
        fr = d[d.cohort.str.lower().str.startswith("frac")]
    for coh, g in fr.groupby("cohort"):
        gs = g.set_index("stage")
        if not {"raw", "readable"} <= set(gs.index):
            continue
        raw, rdb = float(gs.loc["raw", "n_images"]), float(gs.loc["readable", "n_images"])
        k = ckey(coh)
        N.set(f"fracatlas_{k}_raw_images", raw, "before any exclusion")
        N.set(f"fracatlas_{k}_analysed_images", rdb, "after excluding undecodable files")
        exc = raw - rdb
        N.set(f"fracatlas_{k}_excluded_images", exc, "could not be decoded")
        if "n_excluded" in gs.columns and pd.notna(gs.loc["readable", "n_excluded"]):
            stated = float(gs.loc["readable", "n_excluded"])
            if abs(stated - exc) > 0.5:
                raise SystemExit(
                    f"!! {coh}: n_excluded says {stated:.0f} but raw - readable "
                    f"is {exc:.0f}. The flow does not add up.")
        n0 = gs.loc["readable"].get("excluded_label_0")
        n1 = gs.loc["readable"].get("excluded_label_1")
        if pd.notna(n0) and pd.notna(n1):
            if abs(float(n0) + float(n1) - exc) > 0.5:
                raise SystemExit(
                    f"!! {coh}: excluded_label_0 + excluded_label_1 "
                    f"({float(n0):.0f} + {float(n1):.0f}) does not equal the "
                    f"{exc:.0f} images excluded.")
            N.set(f"fracatlas_{k}_excluded_label_0", n0,
                  "excluded images carrying the negative label")
            N.set(f"fracatlas_{k}_excluded_label_1", n1,
                  "excluded images carrying the positive label; a non-zero value "
                  "would make the corruption class-correlated")

    # ---- Source-patient leakage removal ----------------------------------
    # The pair may be labelled (preleak, postleak) or, if the file uses the
    # coarse stage names, (readable, leak_removed). Both mean the same thing:
    # the pool before and after dropping patients shared with the source.
    PAIRS = [("preleak", "postleak"), ("preleak", "leak_removed"),
             ("readable", "leak_removed"), ("raw", "leak_removed")]
    pcts = []
    for coh, g in d.groupby("cohort"):
        gs = g.set_index("stage")
        pre = post = None
        for a, b in PAIRS:
            if a in gs.index and b in gs.index:
                pre, post = float(gs.loc[a, "n_images"]), float(gs.loc[b, "n_images"])
                break
        if pre is None:
            continue
        if pre <= 0:
            raise SystemExit(f"!! {coh}: pre-removal image count is {pre}")
        pct = (pre - post) / pre
        pcts.append(pct)
        N.set(f"leak_removal_pct_{ckey(coh)}", pct,
              "fraction of the target training pool dropped because those "
              "patients also appear in the source training set")
    if pcts:
        N.set("leak_removal_pct_min", float(np.min(pcts)))
        N.set("leak_removal_pct_max", float(np.max(pcts)))
        N.set("leak_removal_pct_mean", float(np.mean(pcts)))
        N.set("leak_removal_n_targets", float(len(pcts)))

    # ---- Source pool, and a cross-check against source_preprocessing ------------------
    n_src = 0
    for coh, g in d.groupby("cohort"):
        if "source" not in coh.lower():
            continue
        n_src += 1
        gs = g[g.stage.isin(LINEAR_STAGES)].sort_values(
            "stage", key=lambda s: s.map(rank))
        if not len(gs):
            gs = g.sort_values("stage", key=lambda s: s.map(rank))
        r = gs.iloc[-1]
        k = ckey(coh)
        N.set(f"data_{k}_images", r["n_images"], "source training pool")
        for col, nm in (("n_studies", "studies"), ("n_patients", "patients"),
                        ("prevalence_image", "prevalence_image"),
                        ("prevalence_study", "prevalence_study")):
            if col in d.columns and pd.notna(r.get(col)):
                N.set(f"data_{k}_{nm}", r[col], "source training pool")

    # A cohort file that supplies none of the three things it exists for is a
    # schema mismatch, not an empty result. Fail with the ids so the mismatch
    # is visible, rather than quietly emitting a shorter Methods section.
    print(f"  [cohorts] {d.cohort.nunique()} cohorts, "
          f"{len(fr.cohort.unique())} FracAtlas, {len(pcts)} leak pairs, "
          f"{n_src} source")
    if not len(fr) or not pcts or not n_src:
        missing_kind = ", ".join(
            k for k, ok in (("FracAtlas raw/readable", len(fr)),
                            ("preleak/postleak pairs", len(pcts)),
                            ("source pool", n_src)) if not ok)
        raise SystemExit(
            f"!! cohort_counts.csv supplied no rows for: {missing_kind}\n"
            f"   cohort ids seen: {sorted(d.cohort.unique())}\n"
            f"   stages seen:     {sorted(d.stage.unique())}\n"
            f"   datasets seen:   "
            f"{sorted(d.dataset.astype(str).unique()) if 'dataset' in d.columns else 'n/a'}\n"
            "   These are the numbers the file exists to provide. Fix the file "
            "or the selection rule; do not proceed without them.")


def crosscheck_source_pool(N: Numbers) -> None:
    """If both source_preprocessing and cohort_counts describe the source pool, they must
    agree. Two files that can answer the same question are a liability unless
    the disagreement is made to fail."""
    vals = {k: v["value"] for k, v in N.d.items()}
    pairs = [(k, k.replace("_source_images", "_images"))
             for k in vals if k.endswith("_source_images")]
    for a, b in pairs:
        if b in vals and abs(float(vals[a]) - float(vals[b])) > 0.5:
            raise SystemExit(
                f"!! source pool disagreement: {a} = {vals[a]} but {b} = {vals[b]}.\n"
                "   source_preprocessing.csv and cohort_counts.csv must agree.")


def _req(g, key):
    """Fetch a value that a table cannot be correct without.

    Returning a default here is how Table 4 shipped with n = 0 on every
    backbone row and 204 on the pooled row. A table that cannot be built is a
    failure, not a zero.
    """
    v = g(key)
    if v is None:
        raise SystemExit(
            f"!! table generation needs '{key}' and it is not in paper_numbers.json.\n"
            "   Re-run make_paper_outputs.py end to end; do not hand-fill the table.")
    return v


def _f(v, d=3):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.{d}f}"


# LaTeX float placement for every generated table. IEEEtran rejects the
# Elsevier `pos=` key, so this must match the target venue's class file.
# Change it here; every generated table uses it.
TABLE_POS = "[!t]"


def emit_latex_tables(N: Numbers, out: Path) -> None:
    """Write the paper's tables as .tex from the same values as the text.

    Tables are generated, never typed, so they cannot disagree with the text.
    """
    v = {k: d["value"] for k, d in N.d.items()}
    g = v.get
    # Prefer the cross-validated penalty when the sweep has been run. The
    # fixed C = 1 numbers stay in the JSON as a documented sensitivity arm.
    NESTED = any(k.startswith("nest_") and k.endswith("_encoder_residual")
                 for k in v)
    MURA = [("shoulder", "Shoulder"), ("humerus", "Humerus"), ("elbow", "Elbow"),
            ("hand", "Hand"), ("finger", "Finger"), ("forearm", "Forearm")]
    ALL = MURA + [("graz", "GRAZPEDWRI-DX")]

    L = [r"\begin{table}" + TABLE_POS, r"\centering",
         r"\caption{Cohorts. For the six MURA targets the training pool is given "
         r"after removal of patients shared with the wrist source, which is the "
         r"pool the models actually saw; the percentage removed is shown. The "
         r"wrist row is the source pool itself. GRAZPEDWRI-DX figures are for "
         r"initial examinations only. FracAtlas is evaluation only and has no "
         r"patient identifiers, so it has no training pool and no study "
         r"grouping. Prevalence is study level except for FracAtlas, which is "
         r"image level.}",
         r"\label{tab:cohorts}", r"{\footnotesize",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"\multirow{2}{*}{Cohort} & \multicolumn{3}{c}{Training pool} & "
         r"\multicolumn{3}{c}{Evaluation pool} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r" & Studies & Images & Removed & Studies & Images & Prevalence \\",
         r"\midrule"]
    for k, lab in (("elbow", "MURA elbow"), ("finger", "MURA finger"),
                   ("forearm", "MURA forearm"), ("hand", "MURA hand"),
                   ("humerus", "MURA humerus"), ("shoulder", "MURA shoulder")):
        rem = next((g(kk) for kk in
                    (f"leak_removal_pct_target_{k}_train",
                     f"leak_removal_pct_{k}_train", f"leak_removal_pct_{k}")
                    if g(kk) is not None), None)
        # NOTE: the percent string is built OUTSIDE the f-string. Python 3.11
        # and earlier reject a backslash inside an f-string expression, and
        # this file must run on the same interpreter as the rest of the study.
        rem_s = "--" if rem is None else ("%.1f" % (100 * rem)) + r"\%"
        L.append(
            f"{lab} & {int(_req(g, f'data_{k}_train_studies'))} & "
            f"{int(_req(g, f'data_{k}_train_images'))} & "
            f"{rem_s} & "
            f"{int(_req(g, f'data_{k}_val_studies'))} & "
            f"{int(_req(g, f'data_{k}_val_images'))} & "
            f"{_f(_req(g, f'cohort_mura_{k}_valid_raw_prevalence_study'))} \\\\")
    L += [r"\midrule",
          f"MURA wrist (source) & {int(_req(g,'data_source_wrist_studies'))} & "
          f"{int(_req(g,'data_source_wrist_images'))} & -- & "
          f"{int(_req(g,'data_wrist_val_studies'))} & "
          f"{int(_req(g,'data_wrist_val_images'))} & "
          f"{_f(g('cohort_mura_wrist_valid_raw_prevalence_study'))} \\\\",
          f"GRAZPEDWRI-DX & {int(_req(g,'data_graz_train_studies'))} & "
          f"{int(_req(g,'data_graz_train_images'))} & -- & "
          f"{int(_req(g,'data_graz_val_studies'))} & "
          f"{int(_req(g,'data_graz_val_images'))} & "
          # study level, to match the caption and every other row of this table.
          # This used to print graz_initial_prevalence_image (0.426), which is
          # the image level rate, under a caption that says study level.
          f"{_f(_req(g,'cohort_graz_valid_initial_only_prevalence_study'))} \\\\",
          f"FracAtlas hand & -- & -- & -- & -- & "
          f"{int(_req(g,'fracatlas_frac_hand_analysed_images'))} & "
          f"{_f(g('data_fracatlas_hand_prevalence'))} \\\\",
          r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (out / "table_cohorts.tex").write_text("\n".join(L) + "\n")

    P = "nest" if NESTED else "oracle"
    cap = (r"\caption{Decomposition of the transfer gap for each target. "
           r"Zero shot is the wrist source model applied unchanged. Oracle is a "
           r"logistic head refitted on frozen source features using all target "
           r"training labels, with the L2 penalty chosen by patient grouped "
           r"cross validation inside that target's own training pool; the "
           r"validation set is never used for selection. Reference is a model "
           r"trained on the target from the start and is a comparison point "
           r"rather than an upper bound. The AUC, gap and residual columns are "
           r"means over all 15 backbone by seed cells. The recovery fraction is "
           r"a mean of PER-CELL ratios over only the $n_g$ cells whose gap "
           r"reaches 0.10 AUC, so it is a different population and is not equal "
           r"to the ratio of the column means in its own row; forearm has no "
           r"qualifying cell. SD is across those $n_g$ cells, which are 3 "
           r"backbones crossed with 5 seeds and are not independent draws.}"
           if NESTED else
           r"\caption{Decomposition of the transfer gap for each target, at a "
           r"fixed readout penalty of $C = 1$.}")
    L = [r"\begin{table}" + TABLE_POS, r"\centering", cap,
         r"\label{tab:oracle}", r"{\footnotesize",
         r"\begin{tabular}{lccccccc}", r"\toprule",
         r"Target & Zero shot & Oracle & Reference & Gap & "
         r"Encoder residual & Recovery (SD) & $n_g$ \\", r"\midrule"]
    # Sort on the value actually printed. Sorting on the all-cell residual
    # while printing the gated one makes the table look unsorted.
    def _sortkey(x):
        return -g(f"{P}_{x[0]}_encoder_residual", 0)
    for k, lab in sorted(ALL, key=_sortkey):
        ng = int(_req(g, f"{P}_{k}_n_gated" if NESTED else f"oracle_{k}_n_cells_gated"))
        if ng == 0:
            L.append(f"{lab} & {_f(_req(g, f'{P}_{k}_zeroshot'))} & "
                     f"{_f(_req(g, f'{P}_{k}_oracle' if NESTED else f'oracle_{k}_oracle'))} & "
                     f"{_f(_req(g, f'{P}_{k}_ceiling'))} & "
                     f"{_f(_req(g, f'{P}_{k}_gap'))} & "
                     f"{_f(_req(g, f'{P}_{k}_encoder_residual'))} & -- & 0 \\\\")
            continue
        z = _req(g, f"{P}_{k}_zeroshot")
        o = _req(g, f"{P}_{k}_oracle")
        c = _req(g, f"{P}_{k}_ceiling")
        fr = _req(g, f"{P}_{k}_readout_frac")
        sd = g(f"{P}_{k}_readout_frac_sd")
        # The printed fraction must be reconstructible from the printed AUC
        # columns. A large divergence means the columns and the fraction have
        # drifted onto different cell populations again, which is exactly the
        # error this rebuild exists to remove.
        # The printed columns are all-cell means; the fraction is a mean of
        # per-cell ratios over the gated subset. They are different
        # populations by design and the caption says so. What must hold is
        # that the GATED columns reproduce the fraction, which is checked
        # against the gated keys rather than the printed ones.
        zg = g(f"{P}_{k}_zeroshot_gated"); og = g(f"{P}_{k}_oracle_gated")
        cg = g(f"{P}_{k}_ceiling_gated")
        if None not in (zg, og, cg) and cg > zg:
            implied = (og - zg) / (cg - zg)
            if abs(implied - fr) > 0.05:
                raise SystemExit(
                    f"!! {k}: the gated columns imply a recovery fraction of "
                    f"{implied:.3f} but the reported fraction is {fr:.3f}.")
        sd_s = "" if sd is None or not np.isfinite(sd) else f" ({sd:.3f})"
        L.append(f"{lab} & {_f(z)} & {_f(o)} & {_f(c)} & "
                 f"{_f(_req(g, f'{P}_{k}_gap'))} & "
                 f"{_f(_req(g, f'{P}_{k}_encoder_residual'))} & "
                 f"{fr:.3f}{sd_s} & {ng} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (out / "table_oracle.tex").write_text("\n".join(L) + "\n")

    L = [r"\begin{table}" + TABLE_POS, r"\centering",
         r"\caption{Labelled target studies needed before refitting the readout "
         r"beats the transferred model. Break even is the smallest budget at "
         r"which the mean exceeds the zero shot value and continues to exceed it "
         r"at every larger budget. Gain at 500 is the mean AUC change at the "
         r"largest budget tested. Wrist source, pooled over three backbones and "
         r"five seeds.}", r"\label{tab:breakeven}", r"{\footnotesize",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Target & Zero shot & Break even (studies) & Gain at 500 \\", r"\midrule"]
    bk = "nest_breakeven_wrist_" if NESTED else "breakeven_wrist_"
    for k, lab in sorted(ALL, key=lambda x: (g(f"{bk}{x[0]}", 1e9)
                                             if (g(f"{bk}{x[0]}") or -1) > 0
                                             else 1e9)):
        be = g(f"nest_breakeven_wrist_{k}") if NESTED else g(f"breakeven_wrist_{k}", -1)
        bes = "never" if be is None or be < 0 else f"{int(be)}"
        L.append(f"{lab} & {_f(g(f'{P}_{k}_zeroshot'))} & {bes} & "
                 f"{g(f'nest_budget500_gain_wrist_{k}' if NESTED else f'budget500_gain_wrist_{k}'):+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (out / "table_break_even.tex").write_text("\n".join(L) + "\n")

    NESTED_BB = g("nest_recovery_anatomy") is not None
    PRE = "nest_recovery_anatomy" if NESTED_BB else "probe_recovery_anatomy"
    L = [r"\begin{table}" + TABLE_POS, r"\centering",
         r"\caption{Recovery of the transfer gap at a budget of 500 labelled "
         r"target studies, by backbone. The three backbones span three of the "
         r"four combinations of architecture family and normalisation. $n$ is "
         r"the number of contributing cells.}",
         r"\label{tab:backbones}", r"{\footnotesize",
         r"\begin{tabular}{llcc}", r"\toprule",
         r"Backbone & Family & Recovery & $n$ \\", r"\midrule"]
    for key, lab, fam in (
            ("densenet121", "DenseNet121", "CNN, batch norm"),
            ("convnext_tiny", "ConvNeXt-T", "CNN, layer norm"),
            ("swin_tiny_patch4_window7_224", "Swin-T", "Transformer, layer norm")):
        pre = "nest_recovery_anatomy" if NESTED_BB else "probe_recovery_anatomy"
        L.append(f"{lab} & {fam} & {_f(_req(g, f'{pre}_{key}'))} & "
                 f"{int(_req(g, f'{pre}_{key}_n'))} \\\\")
    L += [r"\midrule",
          f"Pooled & & {_f(_req(g, PRE))} & {int(_req(g, PRE + '_n'))} \\\\",
          f"Spread & & {_f(_req(g, PRE + '_backbone_spread'))} & \\\\",
          r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (out / "table_backbones.tex").write_text("\n".join(L) + "\n")

    L = [r"\begin{table}" + TABLE_POS, r"\centering",
         r"\caption{Zero shot AUC on the FracAtlas hand target by MURA source "
         r"anatomy. Evaluation only: FracAtlas has no patient identifiers, so no "
         r"reference model exists and no decomposition is possible. The anatomy "
         r"matched source is marked. Means over two backbones and five seeds.}",
         r"\label{tab:fracatlas}", r"{\footnotesize",
         r"\begin{tabular}{lc}", r"\toprule", r"Source anatomy & AUC \\",
         r"\midrule"]
    srcs = [(k, g(f"fracatlas_source_{k}")) for k in
            ("forearm", "humerus", "elbow", "wrist", "shoulder", "finger", "hand")
            if g(f"fracatlas_source_{k}") is not None]
    for k, val in sorted(srcs, key=lambda x: -x[1]):
        lab = k.capitalize() + (" (anatomy matched)" if k == "hand" else "")
        L.append(f"{lab} & {_f(val)} \\\\")
    L += [r"\midrule",
          f"Matched mean & {_f(g('fracatlas_matched_mean'))} \\\\",
          f"Mismatched mean & {_f(g('fracatlas_mismatched_mean'))} \\\\",
          f"Difference & {g('fracatlas_difference'):+.3f} \\\\",
          r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    (out / "table_fracatlas.tex").write_text("\n".join(L) + "\n")
    if NESTED:
        L = [r"\begin{table}" + TABLE_POS, r"\centering",
             # The old caption said the recovery fraction is "derivable from the
             # AUC columns of its own row". It is not: the column is a mean of
             # per-cell ratios, which is close to but not equal to the ratio of
             # the column means (humerus 0.986 against 0.988, finger 0.630
             # against 0.638). Say approximately, not derivable.
             r"\caption{The same decomposition "
             r"restricted to the gated cells, that is, those whose gap reaches "
             r"0.10 AUC. The recovery fraction is still a mean of per cell "
             r"ratios, so it is close to but not exactly the ratio of the AUC "
             r"columns of its own row; on this population the two agree to "
             r"within 0.008. Forearm has no qualifying "
             r"cell and is omitted. Compare with Table 2, whose AUC columns are "
             r"over all 15 cells.}",
             r"\label{tab:suppS1}", r"{\footnotesize",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Target & Zero shot & Oracle & Reference & Recovery & $n_g$ \\",
             r"\midrule"]
        for k, lab in ALL:
            if g(f"nest_{k}_zeroshot_gated") is None:
                continue
            L.append(f"{lab} & {_f(g(f'nest_{k}_zeroshot_gated'))} & "
                     f"{_f(g(f'nest_{k}_oracle_gated'))} & "
                     f"{_f(g(f'nest_{k}_ceiling_gated'))} & "
                     f"{_f(g(f'nest_{k}_readout_frac'))} & "
                     f"{int(g(f'nest_{k}_n_gated'))} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
        (out / "table_gated_supplement.tex").write_text("\n".join(L) + "\n")

        L = [r"\begin{table}" + TABLE_POS, r"\centering",
             r"\caption{Encoder residual against the strength of the readout "
             r"penalty. Smaller $C$ means stronger regularisation. The dataset "
             r"residual is flat across the range while the cross anatomy "
             r"residual is not, so the separation between the two kinds of "
             r"shift is not an artifact of the penalty. The last row is the "
             r"value used elsewhere in this paper, with $C$ chosen by patient "
             r"grouped cross validation inside each target's own training "
             r"pool. A fixed $C = 1$, which an earlier version of this "
             r"analysis used, sits near the bottom of the range.}",
             r"\label{tab:creg}", r"{\footnotesize",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Penalty $C$ & Cross anatomy mean & Cross anatomy worst & "
             r"Dataset & Ratio to worst \\", r"\midrule"]
        for c in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            t = f"C{c:g}".replace(".", "p")
            if g(f"creg_residual_dataset_{t}") is None:
                continue
            L.append(f"${c:g}$ & {_f(_req(g, f'creg_residual_anatomy_mean_{t}'))} & "
                     f"{_f(_req(g, f'creg_residual_anatomy_worst_{t}'))} & "
                     f"{_f(_req(g, f'creg_residual_dataset_{t}'))} & "
                     f"{_req(g, f'creg_ratio_worst_{t}'):.2f} \\\\")
        L += [r"\midrule",
              f"cross validated & {_f(_req(g, 'nest_residual_anatomy_mean'))} & "
              f"{_f(_req(g, 'nest_residual_anatomy_worst'))} & "
              f"{_f(_req(g, 'nest_residual_dataset'))} & "
              f"{_req(g, 'nest_ratio_to_worst'):.2f} \\\\",
              r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
        (out / "table_penalty_sweep.tex").write_text("\n".join(L) + "\n")

    print("  [tex] table_cohorts.tex, table_oracle.tex, table_break_even.tex, "
          "table_backbones.tex, table_fracatlas.tex"
          + (", table_penalty_sweep.tex" if NESTED else ""))


PAPER_BB = ["densenet121", "convnext_tiny", "swin_tiny_patch4_window7_224"]


def _split_creg(df: pd.DataFrame) -> tuple:
    """Split the cross-validated rows from the fixed-penalty grid.

    Also announces any backbone present in the file but not in the paper's
    reported set. readout_penalty_sweep was launched with the full backbone list and
    oracle_readout_cv with the paper's three, so the probe sweep carries a complete
    ResNet50 arm that the reported analysis does not use. Dropping an arm
    quietly is how backbone selection on outcome looks from outside, so the
    drop is printed every run, so the reason can be stated.
    """
    """Separate the cross-validated rows from the fixed-penalty grid.

    The nested rows carry a C value that also appears in the grid, so any
    aggregation keyed on C alone silently pools two populations and inflates
    the cell counts past the 15 that exist.
    """
    extra = sorted(set(df.backbone.unique()) - set(PAPER_BB))
    if extra:
        print(f"  [!] dropping backbone arm(s) not in the reported set: {extra}. "
              f"The manuscript must state why.")
    d = df[df.backbone.isin(PAPER_BB)].copy()
    setting = d["setting"].astype(str)
    return (d[setting.str.startswith("C=")],
            d[setting.isin(["nested", "nestedCV"])])


def numbers_oracle_nested(creg: pd.DataFrame, out: Path, N: Numbers) -> None:
    """PRIMARY oracle decomposition, at a penalty chosen by grouped CV inside
    the target training pool.

    The published run fixed C = 1, which the sweep shows to be the second
    worst of six penalties. It inflated the cross-anatomy encoder residual by
    roughly a factor of three while leaving the dataset residual unchanged, so
    every cross-anatomy number derived from it understated the separation.
    """
    grid, nest = _split_creg(creg)
    if not len(nest):
        print("  [skip] creg oracle has no nested rows")
        return

    rows = []
    for t, g in nest.groupby("target"):
        k = tkey(t)
        gg = g[g.gap >= 0.10]
        rec = {"target": t, "zeroshot": g.zeroshot.mean(),
               "oracle": g.oracle_readout.mean(), "ceiling": g.ceiling.mean(),
               "gap": g.gap.mean(), "residual": g.encoder_residual.mean(),
               "n_cells": int(len(g)), "n_gated": int(len(gg)),
               "C_median": float(g.C.median())}
        rows.append(rec)
        for nm, v, note in (
                ("zeroshot", rec["zeroshot"], "source head, no target labels"),
                ("oracle", rec["oracle"], "frozen encoder, head refitted on all "
                                          "target labels at the CV-chosen penalty"),
                ("ceiling", rec["ceiling"], "target-trained reference model"),
                ("gap", rec["gap"], "reference - zero shot"),
                ("encoder_residual", rec["residual"],
                 "reference - oracle at the CV-chosen penalty")):
            N.set(f"nest_{k}_{nm}", v, note, "cell (backbone x seed)")
        N.set(f"nest_{k}_C_median", rec["C_median"],
              "inverse regularisation strength chosen by grouped CV inside the "
              "target training pool; validation never used")
        N.set(f"nest_{k}_n_cells", float(rec["n_cells"]))
        N.set(f"nest_{k}_n_gated", float(rec["n_gated"]),
              "cells with gap >= 0.10")
        if len(gg):
            N.set(f"nest_{k}_readout_frac", gg.readout_frac.mean(),
                  "(oracle - zero shot) / gap, gated cells only", "gated cell")
            if gg.seed.nunique() > 1:
                N.set(f"nest_{k}_readout_frac_seed_sd",
                      gg.groupby("seed").readout_frac.mean().std(ddof=1),
                      "SD across SEED means, which separates seed variance from "
                      "backbone variance; the 15 cells are 3 backbones crossed "
                      "with 5 seeds and are not independent draws", "seed")
            N.set(f"nest_{k}_readout_frac_sd", gg.readout_frac.std(ddof=1),
                  "SD across gated cells, which are 3 backbones x 5 seeds and "
                  "are not independent draws", "gated cell")
            for nm, col in (("zeroshot_gated", "zeroshot"),
                            ("oracle_gated", "oracle_readout"),
                            ("ceiling_gated", "ceiling"),
                            ("gap_gated", "gap"),
                            ("encoder_residual_gated", "encoder_residual")):
                N.set(f"nest_{k}_{nm}", gg[col].mean(),
                      "GATED cells only, so the table row is self-consistent",
                      "gated cell")
    pd.DataFrame(rows).to_csv(out / "table_oracle_nested.csv", index=False)

    an = nest[~nest.target.astype(str).str.startswith("GRAZ")]
    gz = nest[nest.target.astype(str).str.startswith("GRAZ")]
    per_t = an.groupby("target").encoder_residual.mean()
    N.set("nest_residual_anatomy_mean", per_t.mean(),
          "mean encoder residual over the six cross-anatomy targets", "target")
    N.set("nest_residual_anatomy_worst", per_t.max(),
          f"worst single cross-anatomy target ({per_t.idxmax()})", "target")
    N.set("nest_residual_dataset", gz.encoder_residual.mean(),
          "encoder residual for the one cross-dataset pair", "cell")
    N.set("nest_ratio_to_mean", gz.encoder_residual.mean() / per_t.mean())
    N.set("nest_ratio_to_worst", gz.encoder_residual.mean() / per_t.max(),
          "the conservative headline: dataset residual over the WORST single "
          "cross-anatomy target, not over their mean")

    # Sensitivity across the fixed grid, which is what makes the claim robust
    # rather than merely surviving.
    ratios = []
    for c, g in grid.groupby("C"):
        a = g[~g.target.astype(str).str.startswith("GRAZ")].groupby("target").encoder_residual.mean()
        z = g[g.target.astype(str).str.startswith("GRAZ")].encoder_residual.mean()
        if len(a) and np.isfinite(z):
            ratios.append(z / a.max())
            N.set(f"creg_ratio_worst_C{c:g}".replace(".", "p"), z / a.max(),
                  f"ratio to worst cross-anatomy target at a fixed C = {c:g}")
            N.set(f"creg_residual_dataset_C{c:g}".replace(".", "p"), z,
                  f"dataset residual at a fixed C = {c:g}")
            N.set(f"creg_residual_anatomy_mean_C{c:g}".replace(".", "p"), a.mean(),
                  f"cross-anatomy mean residual at a fixed C = {c:g}", "target")
            N.set(f"creg_residual_anatomy_worst_C{c:g}".replace(".", "p"), a.max(),
                  f"worst cross-anatomy residual at a fixed C = {c:g}", "target")
    if ratios:
        N.set("creg_ratio_worst_min", float(np.min(ratios)),
              "smallest ratio over the whole penalty grid; the claim holds "
              "even here")
        N.set("creg_ratio_worst_max", float(np.max(ratios)))
        N.set("creg_n_penalties", float(len(ratios)))
    N.set("creg_published_C", 1.0, "the fixed penalty used in the first run")


def numbers_backbones_nested(creg_probe, creg_oracle, N: Numbers) -> None:
    """Per-backbone recovery at the 500-study budget, at the CV-chosen penalty.

    The published Table 4 was computed at C = 1 and also shipped with n = 0 on
    every backbone row. Both are fixed here.
    """
    _, po = _split_creg(creg_oracle)
    ceil = po.groupby(["target", "backbone"]).ceiling.mean().to_dict()
    pg, pn = _split_creg(creg_probe)
    zs = (creg_probe[creg_probe.budget == 0]
          [["backbone", "seed", "target", "auc_study"]]
          .rename(columns={"auc_study": "zs"}).drop_duplicates())
    m = (pn[pn.budget == pn.budget.max()].merge(zs, on=["backbone", "seed", "target"])
         .groupby(["backbone", "seed", "target"], as_index=False)
         .agg(auc=("auc_study", "mean"), zs=("zs", "first")))
    m["ceil"] = [ceil.get((t, b), np.nan) for t, b in zip(m.target, m.backbone)]
    m["gap"] = m.ceil - m.zs
    m = m[(m.gap >= 0.10) & (~m.target.astype(str).str.startswith("GRAZ"))]
    if not len(m):
        return
    m["rec"] = (m.auc - m.zs) / m.gap
    by = m.groupby("backbone").rec
    for bb, g in by:
        N.set(f"nest_recovery_anatomy_{bb}", g.mean(),
              f"{BACKBONE_LABEL.get(bb, bb)}, 500-study budget, CV penalty",
              "gated cell")
        N.set(f"nest_recovery_anatomy_{bb}_n", float(len(g)),
              "gated cells for this backbone; NOT the pooled count")
    N.set("nest_recovery_anatomy", m.rec.mean(), "pooled over backbones",
          "gated cell")
    N.set("nest_recovery_anatomy_n", float(len(m)))
    if m.seed.nunique() > 1:
        N.set("nest_recovery_anatomy_seed_sd",
              m.groupby("seed").rec.mean().std(ddof=1),
              "SD across seed means of the pooled recovery", "seed")
    N.set("nest_recovery_anatomy_backbone_spread",
          by.mean().max() - by.mean().min(),
          "largest minus smallest backbone mean")


def numbers_cell_separation(creg_oracle: pd.DataFrame, N: Numbers) -> None:
    """Cell-level residual ranges for the two shift types.

    Limitations currently concedes that one cross-anatomy cell reaches 0.137
    and that the claim is therefore only distributional. Under the published
    analysis that is false and the paper understates itself: the two groups of
    cells do not overlap at all.
    """
    _, nest = _split_creg(creg_oracle)
    if not len(nest):
        return
    an = nest[~nest.target.astype(str).str.startswith("GRAZ")]
    gz = nest[nest.target.astype(str).str.startswith("GRAZ")]
    for tag, d in (("anatomy", an), ("dataset", gz)):
        N.set(f"nest_residual_{tag}_cell_min", float(d.encoder_residual.min()),
              f"smallest single-cell encoder residual, {tag} shift", "cell")
        N.set(f"nest_residual_{tag}_cell_max", float(d.encoder_residual.max()),
              f"largest single-cell encoder residual, {tag} shift", "cell")
        N.set(f"nest_residual_{tag}_n_cells", float(len(d)))
    N.set("nest_residual_cells_overlap",
          1.0 if an.encoder_residual.max() >= gz.encoder_residual.min() else 0.0,
          "1 means the two groups of cells overlap. 0 means every dataset cell "
          "exceeds every anatomy cell, which is a stronger claim than the "
          "distributional one the Limitations currently makes.")


def numbers_backbone_coverage(data: Dict[str, pd.DataFrame], N: Numbers) -> None:
    """How many backbones each analysis actually used.

    Some analyses use two backbones rather than three; the counts are emitted
    so the text can state them exactly.
    """
    for key, label in (("fracatlas", "fracatlas"), ("step1", "gating"),
                       ("zeroshot", "prep_arms"), ("diagonal", "diagonal_null"),
                       ("shortcut", "projection"), ("strata", "graz_strata"),
                       ("source_models", "stage_a"), ("angle", "oracle"),
                       ("creg_oracle", "creg_oracle"), ("creg_probe", "creg_probe")):
        d = data.get(key)
        if d is None or "backbone" not in getattr(d, "columns", []):
            continue
        N.set(f"nbackbones_{label}", float(d.backbone.nunique()),
              "distinct backbones present in this input file")



def numbers_shrinkage(sh: "pd.DataFrame", out: Path, N: Numbers) -> None:
    """readout_shrinkage. Does shrinking toward the source head remove the harm?

    Three quantities the paper will quote, and nothing else:
      * harm rate per budget, for the unrestricted refit and for the shrunk
        readout, on the SAME budget samples;
      * the price of the guarantee -- how much AUC the shrunk readout gives up
        against the unrestricted refit at the largest budget, paired;
      * the lambda the nested CV actually chose, which is what tells a reader
        whether the procedure is doing anything or silently collapsing to
        lambda = 1.
    """
    d = sh[sh.backbone.isin(PAPER_BB)] if "backbone" in sh else sh
    m = d[d.budget > 0]
    if not len(m):
        print("  [shrink] no budget rows; skipped")
        return
    budgets = sorted(int(b) for b in m.budget.unique())

    for setting, tag in (("refit_nestedC", "refit"),
                         ("shrink_nested", "shrink")):
        g = m[m.setting == setting]
        if not len(g):
            continue
        N.set(f"shrink_harm_overall_{tag}", float(g.harm.mean()),
              f"fraction of cells below the transferred model, {setting}",
              "cell (target x backbone x seed x budget x resample)")
        for b in budgets:
            s = g[g.budget == b]
            if len(s):
                N.set(f"shrink_harm_{tag}_at_{b}", float(s.harm.mean()),
                      f"fraction of cells below the transferred model at {b} "
                      f"labelled studies, {setting}",
                      "cell (target x backbone x seed x resample)")

    key = ["target", "backbone", "seed", "resample"]
    big = m[m.budget == max(budgets)]
    a = big[big.setting == "shrink_nested"].groupby(key).auc_study.mean()
    b_ = big[big.setting == "refit_nestedC"].groupby(key).auc_study.mean()
    j = pd.concat([a.rename("shrink"), b_.rename("refit")], axis=1).dropna()
    if len(j):
        N.set("shrink_price_of_guarantee", float((j.refit - j.shrink).mean()),
              f"mean paired AUC given up by the shrunk readout against the "
              f"unrestricted refit at {max(budgets)} labelled studies; "
              f"positive means the guarantee costs something",
              f"paired cell, n={len(j)}")
        N.set("shrink_price_n_pairs", int(len(j)))

    sel = m[m.setting == "shrink_nested"]
    if len(sel) and "chosen_lam" in sel:
        for b in budgets:
            s = sel[sel.budget == b]
            if len(s):
                N.set(f"shrink_lambda_median_at_{b}",
                      float(s.chosen_lam.median()),
                      f"median lambda selected by nested CV at {b} labelled "
                      f"studies; 0 keeps the transferred model, 1 is the "
                      f"unrestricted refit", "cell")
    N.set("shrink_lambda_grid",
          ", ".join(f"{v:g}" for v in sorted(m.lam.dropna().unique())))
    out.mkdir(parents=True, exist_ok=True)
    m.groupby(["setting", "budget"]).agg(
        harm=("harm", "mean"), delta=("delta_vs_zeroshot", "mean"),
        n=("harm", "size")).reset_index().to_csv(
        out / "table_shrinkage.csv", index=False)
    print("  [tab] table_shrinkage.csv")


def numbers_frac_budget(fb: "pd.DataFrame", out: Path, N: Numbers) -> None:
    """fracatlas_budget_cells. Does the harmful regime replicate on FracAtlas?

    NO encoder residual is emitted here and none may be: FracAtlas has no
    patient column, so an in-domain ceiling would leak, and the leak would
    inflate the residual toward this paper's own conclusion. Only quantities
    defined against the transferred model are recorded.
    """
    d = fb[fb.backbone.isin(PAPER_BB)] if "backbone" in fb else fb
    m = d[d.budget > 0]
    z = d[d.budget == 0]
    if not len(m) or not len(z):
        print("  [frac-budget] no rows; skipped")
        return
    budgets = sorted(int(b) for b in m.budget.unique())

    N.set("frac_budget_zeroshot_auc", float(z.auc_study.mean()),
          "zero-shot study AUC of the transferred model on the FracAtlas "
          "held-out half", "mean over backbone x seed")
    N.set("frac_budget_eval_studies", int(z.n_eval_studies.iloc[0]))
    N.set("frac_budget_eval_images", int(z.n_eval_images.iloc[0]))
    N.set("frac_budget_eval_prevalence", float(z.eval_prevalence.iloc[0]),
          "fracture prevalence in the FracAtlas evaluation half")
    N.set("frac_budget_multiscan_dropped", int(z.n_multiscan_dropped.iloc[0]),
          "images dropped because they are flagged multiscan; the one "
          "repeat-imaging channel the FracAtlas metadata exposes")
    if "n_unreadable_dropped" in z.columns:
        N.set("frac_budget_unreadable_dropped",
              int(z.n_unreadable_dropped.iloc[0]),
              "truncated images dropped before the split; FracAtlas's "
              "unreadable files are almost all negatives, so leaving them in "
              "would put class-correlated corruption into the harm rate")

    N.set("frac_budget_anatomy", str(z.frac_anatomy.iloc[0]))

    for setting, tag in (("refit_C1", "c1"), ("refit_nestedC", "refit"),
                         ("shrink_nested", "shrink")):
        g = m[m.setting == setting]
        if not len(g):
            continue
        N.set(f"frac_harm_overall_{tag}", float(g.harm.mean()),
              f"fraction of FracAtlas cells below the transferred model, "
              f"{setting}", "cell (backbone x seed x budget x resample)")
        for b in budgets:
            s = g[g.budget == b]
            if len(s):
                N.set(f"frac_harm_{tag}_at_{b}", float(s.harm.mean()),
                      f"fraction of FracAtlas cells below the transferred "
                      f"model at {b} labelled studies, {setting}", "cell")
        curve = g.groupby("budget").delta_vs_zeroshot.mean()
        be = None
        for i, b in enumerate(budgets):
            if all(curve.get(c, -1.0) > 0 for c in budgets[i:]):
                be = b
                break
        N.set(f"frac_breakeven_{tag}", float(be) if be is not None else -1.0,
              "smallest labelled-study budget at which adaptation beats the "
              "transferred model on FracAtlas and stays ahead; -1 means it "
              "never crosses over", "(target) pooled over backbones and seeds")
    out.mkdir(parents=True, exist_ok=True)
    m.groupby(["setting", "budget"]).agg(
        harm=("harm", "mean"), delta=("delta_vs_zeroshot", "mean"),
        n=("harm", "size")).reset_index().to_csv(
        out / "table_fracatlas_budget.csv", index=False)
    print("  [tab] table_fracatlas_budget.csv")



def numbers_mfxr(mf: "pd.DataFrame", creg_oracle, out: Path, N: Numbers) -> None:
    """mfxr_external. MFXR as a SECOND cross-dataset target.

    The point of this block is the count, not any single value: the separation
    claim moves from one cross-dataset observation to two. Both are emitted,
    and the pooled dataset-arm mean is emitted only when both are present, so
    the paper can never quote an n=2 figure that was computed from n=1.

    MFXR is a case-control cohort (prevalence ~0.61). Absolute AUCs are not
    comparable to a screening setting, so the cohort descriptors are written
    alongside every absolute number and the note says so.
    """
    d = mf[mf.backbone.isin(PAPER_BB)] if "backbone" in mf else mf
    if not len(d):
        print("  [mfxr] no rows for the paper backbones; skipped")
        return

    N.set("mfxr_train_studies", int(d.n_train_studies.iloc[0]))
    N.set("mfxr_train_images", int(d.n_train_images.iloc[0]))
    N.set("mfxr_train_prevalence", float(d.train_prev.iloc[0]),
          "MFXR is a case-control collection, not a consecutive series")
    N.set("mfxr_eval_studies", int(d.n_eval_studies.iloc[0]))
    N.set("mfxr_eval_images", int(d.n_eval_images.iloc[0]))
    N.set("mfxr_eval_prevalence", float(d.eval_prev.iloc[0]),
          "MFXR is a case-control collection, not a consecutive series")
    N.set("mfxr_n_cells", int(len(d)), "backbone x seed cells")

    for col, key in (("zeroshot", "zeroshot"),
                     ("oracle_readout", "oracle"),
                     ("reference", "reference"),
                     ("gap", "gap"),
                     ("readout_recoverable", "readout_recoverable"),
                     ("encoder_residual", "encoder_residual")):
        N.set(f"mfxr_{key}", float(d[col].mean()),
              f"mean {col} for MURA {d.source.iloc[0]} -> MFXR; absolute AUCs "
              f"are from a case-control cohort and are not screening-comparable",
              "cell (backbone x seed)")
        N.set(f"mfxr_{key}_sd", float(d[col].std()), "", "cell (backbone x seed)")

    # The n=2 statement. Only emitted when the GRAZ arm is actually present.
    if creg_oracle is not None and len(creg_oracle):
        c = creg_oracle[creg_oracle.backbone.isin(PAPER_BB)]
        if "setting" in c.columns and (c.setting == "nested").any():
            c = c[c.setting == "nested"]
        gz = c[c.target.astype(str).str.startswith("GRAZ")]
        an = c[~c.target.astype(str).str.startswith("GRAZ")]
        if len(gz) and len(an):
            g_res = float(gz.encoder_residual.mean())
            m_res = float(d.encoder_residual.mean())
            a_by_t = an.groupby("target").encoder_residual.mean()
            N.set("dataset_residual_graz", g_res,
                  "encoder residual, MURA wrist -> GRAZ", "cell")
            N.set("dataset_residual_mfxr", m_res,
                  "encoder residual, MURA wrist -> MFXR", "cell")
            N.set("dataset_residual_mean_n2", 0.5 * (g_res + m_res),
                  "mean encoder residual across the TWO cross-dataset targets; "
                  "this key exists only when both are present", "(target), n=2")
            N.set("dataset_residual_min_n2", min(g_res, m_res),
                  "smaller of the two cross-dataset residuals; the "
                  "conservative figure to quote against the anatomy arm",
                  "(target), n=2")
            N.set("anatomy_residual_mean_for_n2", float(a_by_t.mean()),
                  "mean cross-anatomy encoder residual, recomputed here so the "
                  "ratio below uses one consistent source", "(target)")
            N.set("anatomy_residual_worst_for_n2", float(a_by_t.max()),
                  f"worst cross-anatomy target ({a_by_t.idxmax()})", "(target)")
            if a_by_t.max() > 0:
                N.set("dataset_vs_anatomy_ratio_n2",
                      min(g_res, m_res) / float(a_by_t.max()),
                      "smaller cross-dataset residual divided by the WORST "
                      "cross-anatomy residual; the most conservative form of "
                      "the separation claim, now on two dataset observations",
                      "(target), n=2")
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "table_mfxr.csv", index=False)
    print("  [tab] table_mfxr.csv")



def numbers_lambda_schedule(ls: "pd.DataFrame", out: Path, N: Numbers) -> None:
    """birs_schedule. The prescribed budget-indexed shrinkage schedule.

    Emits the schedule itself, its performance on the HELD-OUT cross-dataset
    arm against every baseline, and the leave-one-out stability flag. The
    oracle row is emitted too, deliberately: the gap between prescribed and
    oracle is what a reader needs in order to judge the prescription, and
    hiding it would flatter the method.

    Nothing here is computed from the calibration arm, because nothing quoted
    about performance may come from the data the schedule was fitted on.
    """
    if not len(ls):
        print("  [lam-sched] no rows; skipped")
        return

    # Take tau from the schedule_lam rows, which carry the PRIMARY tau.
    # ls also holds a tau-sensitivity sweep, and reading the first
    # non-null tau there picks whichever tau happened to be written
    # first (0.0), not the one the schedule was actually fitted under.
    _t = ls[ls.rule == "schedule_lam"].tau.dropna()
    if not len(_t):
        _t = ls[ls.rule.astype(str).str.startswith("schedule(")].tau.dropna()
    tau = float(_t.iloc[0]) if len(_t) else float("nan")
    N.set("lambda_schedule_tau", tau,
          "harm-rate cap the schedule was fitted under: the maximum tolerated "
          "probability that a deploying site ends up worse off")

    sc = ls[ls.rule == "schedule_lam"].sort_values("budget")
    if len(sc):
        N.set("lambda_schedule",
              ", ".join(f"n={int(r.budget)}:{r.lam:g}" for _, r in sc.iterrows()),
              "prescribed shrinkage coefficient by labelled-study budget, "
              "fitted on the cross-anatomy arm only")
        for _, r in sc.iterrows():
            N.set(f"lambda_schedule_at_{int(r.budget)}", float(r.lam),
                  f"prescribed lambda at {int(r.budget)} labelled studies; "
                  f"0 keeps the transferred model unchanged", "(budget)")
        zero_upto = [int(r.budget) for _, r in sc.iterrows() if r.lam == 0.0]
        if zero_upto:
            N.set("lambda_schedule_abstain_upto", max(zero_upto),
                  "largest budget at which the prescription is not to adapt "
                  "at all; the annotation threshold below which local "
                  "adaptation is not worth attempting", "(budget)")

    orc = ls[ls.rule == "oracle_lam"].sort_values("budget")
    if len(orc):
        N.set("lambda_schedule_oracle",
              ", ".join(f"n={int(r.budget)}:{r.lam:g}" for _, r in orc.iterrows()),
              "schedule fitted ON the test arm; NOT achievable, reported only "
              "to show what the prescription leaves behind")

    # Held-out arm, pooled over budgets. budget == -1 marks the pooled rows.
    pooled = ls[(ls.budget == -1) & (ls.rule != "loo_stable")]
    alias = {"zero-shot (lam=0)": "zeroshot", "refit_C1": "refit_c1",
             "refit_nestedC": "refit_nestedc", "shrink_nested": "shrink_nested",
             "ORACLE schedule (not achievable)": "oracle"}
    for _, r in pooled.iterrows():
        rule = str(r["rule"])
        key = alias.get(rule, "schedule" if rule.startswith("schedule") else None)
        if key is None:
            continue
        N.set(f"lamsched_test_harm_{key}", float(r.harm_rate),
              f"harm rate on the held-out cross-dataset arm, {rule}",
              "cell, pooled over budgets")
        N.set(f"lamsched_test_gain_{key}", float(r.mean_gain),
              f"mean AUC gain over the transferred model on the held-out "
              f"cross-dataset arm, {rule}", "cell, pooled over budgets")
        N.set(f"lamsched_test_n_{key}", int(r.n_cells))

    st = ls[ls.rule == "loo_stable"]
    if len(st):
        N.set("lambda_schedule_loo_stable", float(st.harm_rate.iloc[0]),
              "1 if the fitted schedule is unchanged by dropping any single "
              "calibration target, 0 otherwise; 0 means the schedule must be "
              "reported with that caveat", f"{int(st.n_cells.iloc[0])} targets")



    # Per-budget harm on the held-out arm at the primary tau. The pooled zero
    # cannot distinguish "safe at every budget" from "safe because it abstains
    # at the budgets where harm lives". Zero harm at 500, where lambda is
    # largest, is the non-trivial half of the claim and needs its own key.
    pb = ls[(ls.rule == "schedule") & (ls.tau.notna())]
    if len(pb):
        pb = pb[pb.tau.sub(tau).abs() < 1e-12]
        for _, r in pb.sort_values("budget").iterrows():
            N.set(f"lamsched_test_harm_at_{int(r.budget)}", float(r.harm_rate),
                  f"harm rate on the held-out cross-dataset arm at "
                  f"{int(r.budget)} labelled studies, prescribed lambda "
                  f"{r.lam:g}", "cell")
            N.set(f"lamsched_test_gain_at_{int(r.budget)}", float(r.mean_gain),
                  f"mean AUC gain on the held-out arm at {int(r.budget)} "
                  f"labelled studies", "cell")
    ab = ls[ls.rule == "loo_abstain_min"]
    if len(ab):
        N.set("lambda_schedule_abstain_upto_loo_min", int(ab.budget.iloc[0]),
              "smallest abstain threshold across all leave-one-out refits of "
              "the schedule; equal to the full-fit threshold means the "
              "threshold survives dropping any single calibration target, "
              "even where the ramp above it does not",
              f"{int(ab.n_cells.iloc[0])} leave-one-out refits")
        N.set("lambda_schedule_abstain_loo_robust", float(ab.harm_rate.iloc[0]),
              "1 if the abstain threshold is identical in every leave-one-out "
              "refit, 0 otherwise")

    # ---- Clopper-Pearson bounds on the harm rates -------------------------
    # A zero harm count needs an interval to say how precise it is; a one-sided
    # exact bound gives it. Exact rather than normal-approximate because the normal
    # interval is degenerate at a proportion of zero.
    def _cp_upper(k, n, alpha=0.05):
        if n <= 0:
            return float("nan")
        if k <= 0:                       # closed form; also avoids beta(0, n)
            return 1.0 - alpha ** (1.0 / n)
        if k >= n:
            return 1.0
        from scipy import stats as _st
        return float(_st.beta.ppf(1.0 - alpha, k + 1, n - k))

    def g(_k):
        """Read a value already set on N (Numbers exposes no getter)."""
        _v = N.d.get(_k)
        return _v["value"] if isinstance(_v, dict) else None

    for _key in ("schedule", "shrink_nested", "refit_nestedc", "refit_c1",
                 "oracle", "zeroshot"):
        _h = g(f"lamsched_test_harm_{_key}")
        _n = g(f"lamsched_test_n_{_key}")
        if _h is None or _n is None or not np.isfinite(_h) or _n <= 0:
            continue
        _k = int(round(_h * _n))
        N.set(f"lamsched_test_harm_k_{_key}", _k,
              f"number of held-out cells scoring below the transferred model, "
              f"{_key}", f"of {int(_n)} cells")
        N.set(f"lamsched_test_harm_ci95u_{_key}", _cp_upper(_k, int(_n)),
              f"one-sided exact (Clopper-Pearson) 95% upper bound on the harm "
              f"rate for {_key}; at k=0 this is the strongest statement the "
              f"data supports", f"of {int(_n)} cells")

    # Budget 500 separately: it is the budget at which lambda = 1 and the
    # readout is fully refitted, so a zero there is the non-trivial half of
    # the claim and deserves its own interval.
    _h500 = g("lamsched_test_harm_at_500")
    _pb500 = ls[(ls.rule == "schedule") & (ls.budget == 500)]
    if _h500 is not None and len(_pb500):
        _n500 = int(_pb500.n_cells.iloc[0])
        _k500 = int(round(_h500 * _n500))
        N.set("lamsched_test_harm_k_at_500", _k500, "", f"of {_n500} cells")
        N.set("lamsched_test_harm_ci95u_at_500", _cp_upper(_k500, _n500),
              "one-sided exact 95% upper bound on the harm rate at 500 "
              "labelled studies, where the prescribed lambda is 1 and the "
              "readout is refitted without shrinkage", f"of {_n500} cells")

    lo = ls[ls.rule == "loo_drop"]
    if len(lo):
        N.set("lambda_schedule_loo_n_refits", int(lo.target.nunique()))
    out.mkdir(parents=True, exist_ok=True)
    ls.to_csv(out / "table_lambda_schedule.csv", index=False)
    print("  [tab] table_lambda_schedule.csv")



def emit_tables_6_to_9(N: Numbers, out: Path) -> None:
    """LaTeX for the four tables carrying the new results.

    readout_shrinkage/14/15/16 write their tables as CSV only; this function turns them
    into LaTeX, so no table is typed by hand.

    Every value comes from N (i.e. paper_numbers.json), never from re-reading
    a csv, so these tables cannot drift away from the prose.

    Float placement: `pos=htbp` is Elsevier cas-sc syntax; TABLE_POS below
    switches the whole set. The older generators (table0-table5, tableS1)
    hard-code `pos=htbp`.
    """
    BUDGETS = [10, 25, 50, 100, 200, 500]

    v = {k: d["value"] for k, d in N.d.items()}
    g = v.get

    def pct(x, d=1):
        return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"

    def begin(caption, label, colspec, header):
        return [r"\begin{table}" + TABLE_POS, r"\centering",
                r"\caption{" + caption + r"}", r"\label{" + label + r"}",
                r"{\footnotesize", r"\begin{tabular}{" + colspec + r"}",
                r"\toprule", header, r"\midrule"]

    END = [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]

    # ---------------------------------------------------- table 6: shrinkage --
    if g("shrink_harm_overall_shrink") is not None:
        L = begin(
            r"Refitting the readout against shrinking it toward the source "
            r"head, on the same budget samples. Harm is the percentage of "
            r"cells scoring below the transferred model. $\lambda$ is selected "
            r"by patient-grouped cross-validation inside the budget sample; "
            r"its median is shown to expose what the selector actually does. "
            r"Cross-anatomy and cross-dataset targets pooled, three backbones, "
            r"five seeds, five resamples.",
            "tab:shrinkage", "lccc",
            r"Budget (studies) & Refit harm (\%) & Shrunk harm (\%) & "
            r"Median $\lambda$ \\")
        for b in BUDGETS:
            hr, hs = g(f"shrink_harm_refit_at_{b}"), g(f"shrink_harm_shrink_at_{b}")
            if hr is None and hs is None:
                continue
            lam = g(f"shrink_lambda_median_at_{b}")
            L.append(f"{b} & {pct(hr)} & {pct(hs)} & "
                     f"{'--' if lam is None else f'{lam:g}'} " + r"\\")
        L += [r"\midrule",
              f"All budgets & {pct(g('shrink_harm_overall_refit'))} & "
              f"{pct(g('shrink_harm_overall_shrink'))} & -- " + r"\\"]
        L += END
        (out / "table_shrinkage.tex").write_text("\n".join(L) + "\n")

    # ------------------------------------------- table 7: FracAtlas replication --
    if g("frac_harm_overall_refit") is not None:
        anat = g("frac_budget_anatomy") or "hand"
        L = begin(
            r"Replication of the harmful regime on FracAtlas "
            + str(anat).replace("_", r"\_") +
            r". No reference model and no encoder residual are reported: "
            r"FracAtlas carries no patient identifier, so an in-domain ceiling "
            r"would leak. Residual within-dataset leakage inflates the adapted "
            r"arm only, which biases against the harm reported here. "
            + f"{int(g('frac_budget_unreadable_dropped') or 0)} truncated and "
            + f"{int(g('frac_budget_multiscan_dropped') or 0)} multiscan images "
            r"were removed before the split.",
            "tab:fracbudget", "lccc",
            r"Budget (studies) & $C{=}1$ harm (\%) & Refit harm (\%) & "
            r"Shrunk harm (\%) \\")
        for b in BUDGETS:
            row = [g(f"frac_harm_{t}_at_{b}") for t in ("c1", "refit", "shrink")]
            if all(x is None for x in row):
                continue
            L.append(f"{b} & " + " & ".join(pct(x) for x in row) + r" \\")
        L += [r"\midrule",
              "All budgets & "
              + " & ".join(pct(g(f"frac_harm_overall_{t}"))
                           for t in ("c1", "refit", "shrink")) + r" \\"]
        L += END
        (out / "table_fracatlas_budget.tex").write_text("\n".join(L) + "\n")

    # ------------------------------------ table 8: MFXR and the n=2 separation --
    if g("mfxr_encoder_residual") is not None:
        L = begin(
            r"MFXR as a second cross-dataset target, and the resulting "
            r"separation between shift types. MFXR is a case-control cohort "
            + f"(prevalence {_f(g('mfxr_eval_prevalence'))}), so its absolute "
            r"AUCs are not comparable to a screening setting; the "
            r"decomposition terms are within-target differences and are. "
            + f"Means over {int(g('mfxr_n_cells') or 0)} backbone-by-seed "
            r"cells, standard deviation in parentheses.",
            "tab:mfxr", "lcc",
            r"Quantity & Value & SD \\")
        for key, lab in (("zeroshot", "Zero-shot AUC"),
                         ("oracle", "Oracle readout AUC"),
                         ("reference", "Reference model AUC"),
                         ("gap", r"Gap $G$"),
                         ("readout_recoverable", r"Readout recoverable $R$"),
                         ("encoder_residual", r"Encoder residual $E$")):
            val, sd = g(f"mfxr_{key}"), g(f"mfxr_{key}_sd")
            if val is None:
                continue
            L.append(f"{lab} & {_f(val)} & "
                     f"{'--' if sd is None else _f(sd)} " + r"\\")
        if g("dataset_residual_graz") is not None:
            L += [r"\midrule",
                  r"\multicolumn{3}{l}{\textit{Encoder residual by shift type}} \\",
                  f"\\quad Cross-dataset: GRAZPEDWRI-DX & "
                  f"{_f(g('dataset_residual_graz'))} & -- \\\\",
                  f"\\quad Cross-dataset: MFXR & "
                  f"{_f(g('dataset_residual_mfxr'))} & -- \\\\",
                  f"\\quad Cross-anatomy, mean & "
                  f"{_f(g('anatomy_residual_mean_for_n2'))} & -- \\\\",
                  f"\\quad Cross-anatomy, worst target & "
                  f"{_f(g('anatomy_residual_worst_for_n2'))} & -- \\\\",
                  f"\\quad Ratio, smaller dataset residual to worst anatomy & "
                  f"{_f(g('dataset_vs_anatomy_ratio_n2'), 2)}$\\times$ & -- \\\\"]
        L += END
        (out / "table_mfxr.tex").write_text("\n".join(L) + "\n")

    # -------------------------------------------- table 9: the lambda schedule --
    if g("lambda_schedule") is not None:
        tau = g("lambda_schedule_tau")
        n_t = g("lamsched_test_n_schedule")
        L = begin(
            r"The prescribed shrinkage schedule and its performance on the "
            r"held-out cross-dataset arm. $\lambda$ is fitted on the "
            r"cross-anatomy targets alone, by maximising mean AUC gain subject "
            + f"to a harm rate of at most {pct(tau, 0)}\\%, then applied "
            r"unchanged to GRAZPEDWRI-DX, MFXR and FracAtlas, which the fit "
            + f"never sees ({int(n_t or 0)} cells). The oracle schedule is "
            r"fitted on the held-out arm itself and is not achievable; it is "
            r"shown so the cost of the prescription is visible.",
            "tab:lamsched", "lccc",
            r"Budget (studies) & Prescribed $\lambda$ & Harm (\%) & "
            r"Mean AUC gain \\")
        for b in BUDGETS:
            lam = g(f"lambda_schedule_at_{b}")
            if lam is None:
                continue
            gain = g(f"lamsched_test_gain_at_{b}")
            L.append(f"{b} & {lam:g} & {pct(g(f'lamsched_test_harm_at_{b}'))} & "
                     f"{'--' if gain is None else f'{gain:+.4f}'} " + r"\\")
        L += [r"\midrule",
              r"\multicolumn{4}{l}{\textit{Pooled over budgets, held-out arm}} \\",
              r"Policy & & Harm (\%) & Mean AUC gain \\"]
        for key, lab in (("schedule", r"Prescribed schedule"),
                         ("shrink_nested", r"Cross-validated $\lambda$"),
                         ("refit_nestedc", r"Refit, tuned penalty"),
                         ("refit_c1", r"Refit, $C{=}1$"),
                         ("zeroshot", r"No adaptation"),
                         ("oracle", r"Oracle schedule (not achievable)")):
            h, gn = g(f"lamsched_test_harm_{key}"), g(f"lamsched_test_gain_{key}")
            if h is None:
                continue
            L.append(f"\\quad {lab} & & {pct(h)} & {gn:+.4f} " + r"\\")
        L += END
        (out / "table_lambda_schedule.tex").write_text("\n".join(L) + "\n")

    made = [f for f in ("table_shrinkage.tex", "table_fracatlas_budget.tex",
                        "table_mfxr.tex", "table_lambda_schedule.tex")
            if (out / f).exists()]
    print("  [tex] " + (", ".join(made) if made else "none (no new-result keys)"))



def fig_lambda_schedule(lam_sched, out: Path, N: Numbers) -> None:
    """Figure of the BIRS schedule and its results on the external datasets.

    Two stacked panels against labelled budget, single-column width:
      top    harm rate for the prescribed schedule and for the policies it
             replaces, so the zero line is visible against what it replaces;
      bottom mean AUC gain for the same policies, so the cost of the guarantee
             is visible in the same glance.

    The abstain region (budgets where the prescribed lambda is 0) is shaded,
    and the prescribed lambda is annotated at each budget. Those two
    annotations are the figure's actual content: the schedule is a decision
    rule, and a reader should be able to read the rule off the picture.

    Everything is drawn from the b16 csv, which is the same source as Table 9,
    so figure and table cannot disagree.
    """
    import matplotlib.pyplot as plt

    d = lam_sched
    sched = d[d.rule == "schedule_lam"].sort_values("budget")
    if not len(sched):
        print("  [fig] lambda schedule: no schedule_lam rows; skipped")
        return
    tau = float(sched.tau.iloc[0])
    budgets = [int(b) for b in sched.budget]
    lam_at = {int(r.budget): float(r.lam) for _, r in sched.iterrows()}

    # Per-budget curves. The schedule's own curve comes from the rule=schedule
    # rows at the primary tau; the comparison policies come from the grid rows
    # on the test arm at the lambda each policy implies.
    sch = d[(d.rule == "schedule") & (d.tau.sub(tau).abs() < 1e-12)
            ].sort_values("budget")
    grid = d[(d.arm == "test") & (d.rule == "grid")]

    def grid_curve(lam, col):
        s = grid[np.isclose(grid.lam, lam)].sort_values("budget")
        return [float(s.loc[s.budget == b, col].iloc[0])
                if (s.budget == b).any() else np.nan for b in budgets]

    series = [
        ("Prescribed schedule",
         [float(sch.loc[sch.budget == b, "harm_rate"].iloc[0])
          if (sch.budget == b).any() else np.nan for b in budgets],
         [float(sch.loc[sch.budget == b, "mean_gain"].iloc[0])
          if (sch.budget == b).any() else np.nan for b in budgets],
         "o-", 2.0),
        (r"Full refit ($\lambda=1$)", grid_curve(1.0, "harm_rate"),
         grid_curve(1.0, "mean_gain"), "s--", 1.0),
        (r"Fixed $\lambda=0.25$", grid_curve(0.25, "harm_rate"),
         grid_curve(0.25, "mean_gain"), "^:", 1.0),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True)
    x = np.arange(len(budgets))

    abstain = [i for i, b in enumerate(budgets) if lam_at.get(b, 0.0) == 0.0]
    for ax in (ax1, ax2):
        if abstain:
            ax.axvspan(-0.5, max(abstain) + 0.5, color="0.92", zorder=0)

    for lab, harm, _gain, style, lw in series:
        ax1.plot(x, [100 * h if h == h else np.nan for h in harm], style,
                 label=lab, linewidth=lw, markersize=4)
    ax1.axhline(100 * tau, color="0.4", linewidth=0.6, linestyle="-.")
    ax1.text(len(budgets) - 1, 100 * tau, f"  $\\tau$={100*tau:g}%",
             va="bottom", ha="right", fontsize=6, color="0.35")
    ax1.set_ylabel("Harm rate (%)")
    ax1.legend(loc="upper right", fontsize=6)

    for lab, _harm, gain, style, lw in series:
        ax2.plot(x, gain, style, label=lab, linewidth=lw, markersize=4)
    ax2.axhline(0.0, color="0.4", linewidth=0.6)
    ax2.set_ylabel("Mean AUC gain")
    ax2.set_xlabel("Labelled target studies")
    ax2.set_xticks(x)
    # The rule itself, in the tick labels: outside the data area, so no
    # plotted point (e.g. a negative gain at the smallest budget) can cover it.
    ax2.set_xticklabels([f"{b}\n$\\lambda$={lam_at[b]:g}" for b in budgets])
    if abstain:
        ax1.text(max(abstain) / 2.0, ax1.get_ylim()[1], "abstain",
                 ha="center", va="top", fontsize=6, color="0.35")

    fig.tight_layout(pad=0.3)
    save(fig, out, "fig_lambda_schedule")
    plt.close(fig)



def numbers_label_audit(gx: "pd.DataFrame", fb, out: Path, N: Numbers) -> None:
    """b19. GRAZPEDWRI-DX label disagreement and the FracAtlas budget-pool size.

    (1) The GRAZPEDWRI-DX label disagreement. The dataset is scored on "any
    pathological finding", the same concept as MURA, while the paper's other
    cross-dataset targets are fracture-only. The disagreement is one-sided:
    abnormal is a superset of fracture, so the count of fracture-without-
    abnormal is emitted too and a reader can check that it is zero.

    (2) The FracAtlas budget-pool size, read from the b14 per-cell records
    rather than a log line.
    """
    if gx is not None and len(gx):
        r = gx.iloc[0]
        n = int(r.n_images)
        N.set("graz_label_total_images", n, "images in the full GRAZ collection")
        N.set("graz_label_disagree_images", int(r.n_disagree),
              "images where label_abnormal and label_fracture differ")
        N.set("graz_label_disagree_frac", float(r.frac_disagree),
              "fraction of the full GRAZ collection where the two labels differ")
        N.set("graz_label_abnormal_not_fracture", int(r.abnormal_not_fracture),
              "abnormal but not fracture; the whole of the disagreement")
        N.set("graz_label_fracture_not_abnormal", int(r.fracture_not_abnormal),
              "fracture but not abnormal; zero means abnormal is a superset")
        print(f"  [labels] GRAZ disagreement {int(r.n_disagree)}/{n} "
              f"({100*float(r.frac_disagree):.3f}%)")
    if gx is not None and len(gx) and "frac_pool_images" in gx.columns:
        _pi = gx.frac_pool_images.iloc[0]
        if _pi == _pi:                                   # not NaN
            N.set("frac_budget_train_pool_images", int(_pi),
                  "images in the FracAtlas budget pool, after the multiscan "
                  "and unreadable exclusions")
            N.set("frac_budget_train_pool_studies", int(gx.frac_pool_studies.iloc[0]),
                  "studies in the FracAtlas budget pool")
def numbers_harm_analyses(hm, mg, lo, wc, out: Path, N: Numbers) -> None:
    """b20. Harm magnitude, harm margins, leave-one-anatomy-out and worst cells.

    All four inputs are re-analyses of stored b13/b14/b16 per-cell records
    (results/transfer/harm_analyses/); nothing here fits
    anything. Policy names are mapped once so the same policy has the same
    key stem in every family:
      schedule, oracle_schedule, abstain50 (lam=0 for n<=50, lam=1 above),
      cv_lambda (shrink_nested), refit_tuned (refit_nestedC), refit_c1,
      lam0p1 (fixed lambda = 0.1), no_adaptation (lam=0).

      rr_heldout_*   q1: held-out arm, 1,345 paired cells, pooled over budgets
      rr_margin_*    q2: harmed-cell counts at AUC-loss margin m
      rr_loao_*      q4: left-out anatomy, schedule refitted without it
      rr_worst_*     q5: the single worst cross-validated-lambda cell
    """
    heldout_policy_names = {"schedule_tau0.05": "schedule", "oracle_schedule": "oracle_schedule",
                "threshold_only_abstain50": "abstain50", "shrink_nested": "cv_lambda",
                "refit_nestedC": "refit_tuned", "refit_C1": "refit_c1",
                "shrink_lam0.1": "lam0p1"}
    loo_policy_names = {"loo_schedule": "schedule", "no_adaptation": "no_adaptation",
                "abstain50_lam1": "abstain50", "cv_lambda": "cv_lambda",
                "refit_tuned": "refit_tuned", "refit_C1": "refit_c1"}
    unit = "cell (target x backbone x seed x budget x resample)"

    if hm is not None and len(hm):
        p = hm[hm.budget.astype(str) == "pooled"]
        n = p.n_cells.unique()
        if len(n) != 1:
            raise SystemExit(f"!! q1 policies are not paired: n_cells {sorted(n)}")
        N.set("rr_heldout_paired_cells", int(n[0]),
              "held-out cells present for every policy (GRAZ_initial, MFXR, "
              "FRAC_XR_HAND)", unit)
        for _, r in p[p.policy.isin(heldout_policy_names)].iterrows():
            k = f"rr_heldout_{heldout_policy_names[r.policy]}"
            N.set(f"{k}_n_harmed", int(r.n_harmed),
                  "held-out cells with AUC below the transferred model", unit)
            if r.n_harmed > 0:
                N.set(f"{k}_median_loss", float(r.median_loss),
                      "median AUC loss among harmed held-out cells", unit)
                N.set(f"{k}_worst_loss", float(r.worst_loss),
                      "largest AUC loss among harmed held-out cells", unit)
        print(f"  [rr] held-out harm magnitude: {len(p)} policies, "
              f"{int(n[0])} paired cells")

    if mg is not None and len(mg):
        for _, r in mg[mg.policy.isin(heldout_policy_names)].iterrows():
            nm = heldout_policy_names[r.policy]
            if nm in ("oracle_schedule", "lam0p1"):
                continue
            m = f"{float(r.margin):g}".replace(".", "p")
            N.set(f"rr_margin_{nm}_m{m}", int(r.n_harmed),
                  f"held-out cells whose AUC falls by more than {float(r.margin):g}",
                  unit)

    if lo is not None and len(lo):
        lo = lo[lo.policy.isin(loo_policy_names)]
        for t, g in lo.groupby("left_out_anatomy"):
            if g.n_cells.nunique() != 1:
                raise SystemExit(f"!! q4 {t}: policies are not paired")
            a = str(t).lower().replace("xr_", "")
            N.set(f"rr_loao_{a}_n_cells", int(g.n_cells.iloc[0]),
                  f"left-out {t} cells, identical for every policy", unit)
            for _, r in g.iterrows():
                k = f"rr_loao_{a}_{loo_policy_names[r.policy]}"
                N.set(f"{k}_n_harmed", int(r.n_harmed),
                      f"harmed cells on {t} when held out of the schedule fit", unit)
                N.set(f"{k}_harm_rate", float(r.harm_rate),
                      f"harm rate on left-out {t}", unit)
                N.set(f"{k}_mean_gain", float(r.mean_gain),
                      f"mean AUC gain over all left-out {t} cells", unit)
        w = lo.assign(g_=lo.mean_gain * lo.n_cells).groupby("policy")[
            ["n_cells", "n_harmed", "g_"]].sum()
        for pol, r in w.iterrows():
            k = f"rr_loao_pooled_{loo_policy_names[pol]}"
            N.set(f"{k}_n_cells", int(r.n_cells),
                  "left-out-anatomy cells pooled over the six anatomies", unit)
            N.set(f"{k}_n_harmed", int(r.n_harmed), "", unit)
            N.set(f"{k}_harm_rate", float(r.n_harmed / r.n_cells),
                  "pooled harm rate, left-out anatomies", unit)
            N.set(f"{k}_mean_gain", float(r.g_ / r.n_cells),
                  "cell-weighted mean AUC gain, left-out anatomies", unit)
        print(f"  [rr] leave-one-anatomy-out: "
              f"{lo.left_out_anatomy.nunique()} anatomies x {len(w)} policies")

    if wc is not None and len(wc):
        c = wc[(wc.setting == "shrink_nested") & (wc["rank"] == 1)]
        if len(c) != 1:
            raise SystemExit("!! q5: expected one rank-1 shrink_nested row")
        c = c.iloc[0]
        N.set("rr_worst_cv_lambda_zeroshot_auc", float(c.zeroshot_auc),
              "transferred-model AUC of the worst cross-validated-lambda cell "
              "(identity in worst_loss_cells.csv)", unit)
        N.set("rr_worst_cv_lambda_adapted_auc", float(c.auc_study),
              "post-adaptation AUC of that cell", unit)


def numbers_review6(ls, sh, fb, hm, mg, wc, out: Path, N: Numbers) -> None:
    """b21. Schedule, worst-cell, margin and monotonicity values from the stored cells.

    Every input is already registered: lam_sched (b16), shrink (b13),
    frac_budget (b14) and the harm_analyses CSVs b20 reads. Nothing is
    fitted here.

      heldout_schedule_*   b16 per_target rows, tau = 0.05, one per dataset
      rr_worst_*           the worst cross-validated-lambda held-out cell, read
                           back from its own b13/b14 per-cell rows
      rr_top10_*           composition of the ten largest such losses
      rr_margin_lam0p1_*   fixed lambda = 0.1 at the q2 margins
      sched_tau0_*         the tau = 0 schedule on the held-out arm
      grid_harm_*          harm monotone in lambda / non-increasing in budget,
                           checked on BOTH b16 grids (anatomy and held-out).
                           Gain is deliberately not tested: it is not monotone
                           in budget and nothing claims it is.
    """
    tag = {"GRAZ_initial": "graz", "MFXR": "mfxr", "FRAC_XR_HAND": "frac"}
    unit = "cell (target x backbone x seed x budget x resample)"

    if ls is not None and len(ls):
        pt = ls[(ls.rule == "per_target") & (ls.arm == "test")
                & np.isclose(ls.tau, 0.05)]
        for _, r in pt.iterrows():
            k = tag.get(str(r.target))
            if k is None:
                continue
            N.set(f"heldout_schedule_n_cells_{k}", int(r.n_cells),
                  f"held-out {r.target} cells under the tau=0.05 schedule", unit)
            N.set(f"heldout_schedule_n_harmed_{k}",
                  int(round(r.harm_rate * r.n_cells)),
                  f"harmed {r.target} cells under the tau=0.05 schedule", unit)
            N.set(f"heldout_schedule_mean_gain_{k}", float(r.mean_gain),
                  f"mean AUC gain on {r.target} under the tau=0.05 schedule", unit)

        t0 = ls[(ls.rule == "schedule") & (ls.arm == "test")
                & np.isclose(ls.tau, 0.0)].sort_values("budget")
        for _, r in t0.iterrows():
            N.set(f"sched_tau0_lam_at_{int(r.budget)}", float(r.lam),
                  "lambda the tau = 0 schedule prescribes at this budget")
        if len(t0):
            r5 = t0[t0.budget == t0.budget.max()].iloc[0]
            N.set(f"sched_tau0_heldout_mean_gain_at_{int(r5.budget)}",
                  float(r5.mean_gain),
                  "held-out mean AUC gain of the tau = 0 schedule at the "
                  "largest budget only", unit)
            N.set("sched_tau0_heldout_mean_gain_pooled",
                  float((t0.mean_gain * t0.n_cells).sum() / t0.n_cells.sum()),
                  "held-out mean AUC gain of the tau = 0 schedule over all "
                  "budgets (cell-weighted)", unit)

        gr = ls[ls.rule == "grid"]
        mono_lam, mono_b, n = True, True, 0
        for arm, a in gr.groupby("arm"):
            hp = a.pivot(index="budget", columns="lam", values="harm_rate")
            mono_lam &= bool((hp.diff(axis=1).iloc[:, 1:] >= -1e-12).all().all())
            mono_b &= bool((hp.diff(axis=0).iloc[1:] <= 1e-12).all().all())
            n += int(a[np.isclose(a.lam, 0.0)].n_cells.sum())
        N.set("grid_harm_monotone_in_lambda", float(mono_lam),
              "1 if the harm rate is non-decreasing in lambda at every budget, "
              "in both the anatomy and the held-out grid")
        N.set("grid_harm_nonincreasing_in_budget", float(mono_b),
              "1 if the harm rate is non-increasing in budget at every lambda, "
              "in both grids")
        N.set("grid_harm_monotone_n_cells", n,
              "cells per lambda over both grids", unit)
        s25 = gr[(gr.arm == "test") & np.isclose(gr.lam, 0.25) & (gr.budget == 50)]
        if len(s25):
            N.set("heldout_lam0p25_harm_at_50", float(s25.harm_rate.iloc[0]),
                  "held-out harm RATE of fixed lambda = 0.25 at 50 studies", unit)

    if hm is not None and len(hm):
        a = hm[hm.policy == "threshold_only_abstain50"]
        for _, r in a.iterrows():
            if str(r.budget) == "pooled":
                N.set("rr_heldout_abstain50_mean_gain", float(r.mean_gain),
                      "held-out mean AUC gain over all cells, abstain<=50 then "
                      "lambda = 1", unit)
            else:
                N.set(f"rr_heldout_abstain50_n_harmed_at_{int(r.budget)}",
                      int(r.n_harmed), "held-out harmed cells at this budget",
                      unit)

    if mg is not None and len(mg):
        for _, r in mg[mg.policy == "shrink_lam0.1"].iterrows():
            m = f"{float(r.margin):g}".replace(".", "p")
            N.set(f"rr_margin_lam0p1_m{m}", int(r.n_harmed),
                  f"held-out cells where fixed lambda = 0.1 loses more than "
                  f"{float(r.margin):g} AUC", unit)

    if wc is not None and len(wc):
        cv = wc[wc.setting == "shrink_nested"].sort_values("rank")
        top = cv.head(10)
        N.set("rr_top10_cv_lambda_n_lam1", int(np.isclose(top.lam, 1.0).sum()),
              "of the ten largest held-out cv-lambda losses, cells deployed at "
              "lambda = 1")
        N.set("rr_top10_cv_lambda_n_budget10", int((top.budget == 10).sum()),
              "of the ten largest held-out cv-lambda losses, cells at 10 studies")
        c = cv.iloc[0]
        N.set("rr_worst_cv_lambda_target", str(c.target),
              "target of the worst cv-lambda cell (GRAZ_initial = "
              "GRAZPEDWRI-DX, initial examinations)")
        N.set("rr_worst_cv_lambda_budget", int(c.budget), "labelled studies")
        N.set("rr_worst_cv_lambda_lambda", float(c.lam), "lambda deployed")
        N.set("rr_worst_cv_lambda_C", float(c.chosen_C), "C chosen by the CV")
        src = fb if str(c.target).startswith("FRAC") else sh
        if src is not None and len(src):
            grid_c = sorted(src.chosen_C.dropna().unique())
            N.set("rr_worst_cv_lambda_C_is_grid_min",
                  float(np.isclose(c.chosen_C, grid_c[0])),
                  "1 if that C is the strongest penalty on the C grid "
                  f"({', '.join(f'{v:g}' for v in grid_c)})")
            cell = src[(src.target == c.target) & (src.backbone == c.backbone)
                       & (src.seed == c.seed) & (src["budget"] == c.budget)
                       & (src["resample"] == c["resample"])]
            fx = cell[cell.setting.isin(["shrink_lam0.1", "shrink_lam0.25",
                                         "shrink_lam0.5", "shrink_lam0.75",
                                         "shrink_lam0.9"])]
            if len(fx) != 5:
                raise SystemExit(f"!! worst cell: expected 5 fixed-lambda rows, "
                                 f"found {len(fx)}")
            N.set("rr_worst_fixed_lam_min_auc", float(fx.auc_study.min()),
                  "lowest study AUC over fixed lambda in {0.1..0.9} in the "
                  "worst cv-lambda cell", unit)

    if fb is not None and len(fb):
        d = fb[fb.backbone.isin(PAPER_BB) & (fb["budget"] > 0)]
        g = d[(d.setting == "refit_nestedC") & (d["budget"] == d["budget"].max())]
        if len(g):
            N.set(f"frac_refit_nested_gain_at_{int(d['budget'].max())}",
                  float(g.delta_vs_zeroshot.mean()),
                  "FracAtlas mean AUC gain of the nested-C refit at the largest "
                  "budget (table7 refit_nestedC)", unit)


def numbers_review7(ls, sh, hm, out: Path, N: Numbers) -> None:
    """b22. Numbers the seventh review found printed but unkeyed.

      rr_heldout_abstain50_mean_gain_at_*   q1 per-budget rows (held-out arm)
      shrink_c1_*                           refit_C1 vs refit_nestedC harm on
                                            the Table 5/6 pool, from the b13
                                            rows with the table6 filter
      heldout_best_zero_harm_mean_gain      per budget, the best held-out grid
                                            gain among lambdas with harm
                                            EXACTLY 0, cell-weighted. Chosen on
                                            the test arm, so an oracle bound.
      sched_share_of_zero_harm_gain         tau=0.05 schedule gain / that bound
    """
    unit = "cell (target x backbone x seed x budget x resample)"

    if hm is not None and len(hm):
        a = hm[(hm.policy == "threshold_only_abstain50")
               & (hm.budget.astype(str) != "pooled")]
        for _, r in a.iterrows():
            if int(r.budget) > 50:
                N.set(f"rr_heldout_abstain50_mean_gain_at_{int(r.budget)}",
                      float(r.mean_gain),
                      "held-out mean AUC gain at this budget, abstain<=50 then "
                      "lambda = 1 (equal to the unshrunk nested-C refit here)",
                      unit)

    if sh is not None and len(sh):
        d = sh[sh.backbone.isin(PAPER_BB) & (sh["budget"] > 0)]
        h = (d[d.setting.isin(["refit_C1", "refit_nestedC"])]
             .groupby(["setting", "budget"]).harm.mean().unstack(0))
        ex = h["refit_C1"] - h["refit_nestedC"]
        N.set("shrink_c1_harm_exceeds_tuned_all_budgets", float((ex > 0).all()),
              "1 if the C = 1 refit harms a larger fraction of cells than the "
              "nested-C refit at every budget (Table 5/6 pool)")
        N.set("shrink_c1_min_harm_excess", float(ex.min()),
              "smallest harm-rate excess of C = 1 over nested C across budgets",
              unit)
        N.set("shrink_c1_min_harm_excess_budget", int(ex.idxmin()),
              "budget at which that smallest excess occurs")

    if ls is not None and len(ls):
        g = ls[(ls.rule == "grid") & (ls.arm == "test")]
        tot = n = 0.0
        for b, a in g.groupby("budget"):
            z = a[a.harm_rate == 0.0]
            tot += float(z.mean_gain.max()) * int(z.n_cells.iloc[0])
            n += int(z.n_cells.iloc[0])
        best = tot / n
        N.set("heldout_best_zero_harm_mean_gain", best,
              "held-out gain of the best zero-harm grid lambda per budget, "
              "picked on the test arm (an oracle bound), cell-weighted", unit)
        s = ls[(ls.rule == "schedule(tau=0.05)") & (ls.arm == "test")]
        if len(s) and best > 0:
            N.set("sched_share_of_zero_harm_gain",
                  float(s.mean_gain.iloc[0]) / best,
                  "held-out gain of the tau = 0.05 schedule as a fraction of "
                  "heldout_best_zero_harm_mean_gain")


def numbers_bspc_hybrids(sh, fb, ls, hm, mg, out: Path, N: Numbers) -> None:
    """b25. Abstaining hybrids and zero-harm bounds.

    Nothing is fitted. Hybrids ("abstain at <= 50 studies, then policy X")
    are assembled cell by cell from the stored b13/b14 rows, on the same
    paired cells as q1 (held-out) and q4 (left-out anatomy). The abstention
    threshold is 50 in every leave-one-out refit
    (lambda_schedule_abstain_upto_loo_min), so no schedule is refitted.

      rr_loao_*            left-out anatomy, new hybrids and fixed lambda 0.5
      rr_heldout_*         held-out arm, new hybrids, fixed lambda 0.5, and
                           gain above the abstention threshold (q1 rows)
      rr_margin_lam0p5_*   q2 rows for fixed lambda 0.5
      calib_grid_lam0p1_*  b16 calibration-arm harm of fixed lambda 0.1
      heldout_schedule_*   zero-harm counts and one-sided 95% bounds
                           1 - 0.05**(1/n) at four levels of clustering
    """
    K = ["target", "backbone", "seed", "budget", "resample"]
    unit = "cell (target x backbone x seed x budget x resample)"
    UPTO = 50

    def paired(d):
        common = None
        for s in ("shrink_lam0", "shrink_lam1", "refit_C1", "refit_nestedC",
                  "shrink_nested"):
            i = d[d.setting == s].set_index(K).index
            common = i if common is None else common.intersection(i)
        return lambda s: d[d.setting == s].set_index(K).loc[common]

    def hybrid(get, above):
        lo, hi = get("shrink_lam0"), get(above)
        low = lo.index.get_level_values("budget") <= UPTO
        return pd.DataFrame({
            "harm": np.where(low, lo.harm, hi.harm),
            "gain": np.where(low, lo.delta_vs_zeroshot, hi.delta_vs_zeroshot)},
            index=lo.index)

    def as_hg(df):
        return pd.DataFrame({"harm": df.harm.to_numpy(),
                             "gain": df.delta_vs_zeroshot.to_numpy()},
                            index=df.index)

    pols = {"abstain50_cv_lambda": lambda g: hybrid(g, "shrink_nested"),
            "abstain50_lam0p5": lambda g: hybrid(g, "shrink_lam0.5"),
            "fixed_lam0p5": lambda g: as_hg(g("shrink_lam0.5"))}

    if sh is not None and len(sh):
        d = sh[sh.backbone.isin(PAPER_BB) & (sh["budget"] > 0)]
        anat = d[d.target.astype(str).str.match(r"^XR_[A-Z]+$")]
        tot = {p: [0, 0, 0.0] for p in pols}
        for t, dt in anat.groupby("target"):
            a = str(t).lower().replace("xr_", "")
            get = paired(dt)
            for p, f in pols.items():
                r = f(get)
                N.set(f"rr_loao_{a}_{p}_n_harmed", int(r.harm.sum()),
                      f"harmed cells on {t} when held out of the schedule fit", unit)
                N.set(f"rr_loao_{a}_{p}_mean_gain", float(r.gain.mean()),
                      f"mean AUC gain over all left-out {t} cells", unit)
                tot[p][0] += len(r); tot[p][1] += int(r.harm.sum())
                tot[p][2] += float(r.gain.sum())
        for p, (n, h, g) in tot.items():
            k = f"rr_loao_pooled_{p}"
            N.set(f"{k}_n_cells", n, "left-out-anatomy cells, six anatomies", unit)
            N.set(f"{k}_n_harmed", h, "", unit)
            N.set(f"{k}_harm_rate", h / n, "pooled harm rate, left-out anatomies", unit)
            N.set(f"{k}_mean_gain", g / n, "cell-weighted mean AUC gain", unit)

        if fb is not None and len(fb):
            cols = K + ["setting", "harm", "delta_vs_zeroshot"]
            f2 = fb[fb.backbone.isin(PAPER_BB) & (fb["budget"] > 0)]
            ho = pd.concat([d[~d.target.astype(str).str.match(r"^XR_[A-Z]+$")][cols],
                            f2[cols]], ignore_index=True)
            get = paired(ho)
            for p in ("abstain50_cv_lambda", "abstain50_lam0p5"):
                r = pols[p](get)
                N.set(f"rr_heldout_{p}_n_harmed", int(r.harm.sum()),
                      "held-out cells below the transferred model", unit)
                N.set(f"rr_heldout_{p}_mean_gain", float(r.gain.mean()),
                      "held-out mean AUC gain over all cells", unit)

            if ls is not None and len(ls):
                sl = ls[ls.rule == "schedule_lam"]
                lam_at = {int(r.budget): float(r.lam) for r in sl.itertuples()}
                lam0 = get("shrink_lam0")
                b = lam0.index.get_level_values("budget")
                harm = np.zeros(len(lam0), dtype=int)
                for bud, lam in lam_at.items():
                    if lam == 0.0:
                        continue
                    s = get(f"shrink_lam{lam:g}")
                    harm = np.where(b == bud, s.harm.to_numpy(), harm)
                cell = pd.DataFrame({"harm": harm}, index=lam0.index)
                adapted = np.isin(b, [k for k, v in lam_at.items() if v > 0])
                units = {
                    "cells_adapted": (int(adapted.sum()),
                                      int(cell.harm[adapted].sum()),
                                      "held-out cells at budgets where the "
                                      "schedule adapts (lambda > 0)"),
                    "pairs": (cell.groupby(level=["backbone", "seed", "target"])
                              .harm.max().size,
                              int(cell.groupby(level=["backbone", "seed", "target"])
                                  .harm.max().sum()),
                              "model x dataset pairs (backbone x seed x target)"),
                    "models": (cell.groupby(level=["backbone", "seed"]).harm.max().size,
                               int(cell.groupby(level=["backbone", "seed"])
                                   .harm.max().sum()),
                               "source models (backbone x seed)"),
                    "target_backbone": (
                        cell.groupby(level=["target", "backbone"]).harm.max().size,
                        int(cell.groupby(level=["target", "backbone"])
                            .harm.max().sum()),
                        "target x backbone configurations (seeds pooled)")}
                for u, (n, h, note) in units.items():
                    N.set(f"heldout_schedule_n_{u}", n, note)
                    N.set(f"heldout_schedule_harmed_{u}", h,
                          f"{note} with at least one harmed cell")
                    if h == 0:
                        N.set(f"heldout_schedule_harm_ub95_{u}", 1 - 0.05 ** (1 / n),
                              "one-sided 95% upper bound on the harm probability "
                              f"with 0 of {n} harmed, 1 - 0.05**(1/n)")

    if hm is not None and len(hm):
        q = hm.assign(b=hm.budget.astype(str))
        r = q[(q.policy == "shrink_lam0.5") & (q.b == "pooled")]
        if len(r):
            r = r.iloc[0]
            N.set("rr_heldout_lam0p5_n_harmed", int(r.n_harmed), "", unit)
            N.set("rr_heldout_lam0p5_mean_gain", float(r.mean_gain), "", unit)
            N.set("rr_heldout_lam0p5_worst_loss", float(r.worst_loss),
                  "largest AUC loss among harmed held-out cells", unit)
        for _, r in q[(q.policy == "shrink_lam0.5") & (q.b != "pooled")].iterrows():
            N.set(f"rr_heldout_lam0p5_n_harmed_at_{int(r.b)}", int(r.n_harmed),
                  "held-out harmed cells at this budget", unit)
        nl = q[(q.policy == "shrink_nested") & (q.b != "pooled")]
        nl = nl[nl.b.astype(int) <= UPTO]
        N.set("rr_heldout_cv_lambda_n_harmed_le50", int(nl.n_harmed.sum()),
              "held-out cv-lambda harmed cells at budgets <= 50", unit)
        for pol, tag in (("schedule_tau0.05", "schedule"), ("shrink_lam0.5", "lam0p5"),
                         ("shrink_nested", "cv_lambda")):
            s = q[(q.policy == pol) & (q.b != "pooled")]
            s = s[s.b.astype(int) > UPTO]
            N.set(f"rr_heldout_{tag}_mean_gain_above50",
                  float((s.mean_gain * s.n_cells).sum() / s.n_cells.sum()),
                  "held-out mean AUC gain at budgets > 50, cell-weighted", unit)

    if mg is not None and len(mg):
        for _, r in mg[mg.policy == "shrink_lam0.5"].iterrows():
            m = f"{float(r.margin):g}".replace(".", "p")
            N.set(f"rr_margin_lam0p5_m{m}", int(r.n_harmed),
                  f"held-out cells where fixed lambda = 0.5 loses more than "
                  f"{float(r.margin):g} AUC", unit)

    if ls is not None and len(ls):
        c = ls[(ls.arm == "calibrate") & (ls.target == "ALL") & (ls.rule == "grid")
               & np.isclose(ls.lam, 0.1)]
        for r in c.itertuples():
            N.set(f"calib_grid_lam0p1_harm_at_{int(r.budget)}", float(r.harm_rate),
                  "calibration-arm (anatomy) harm rate of fixed lambda = 0.1", unit)


def numbers_l2sp(l2, sh, fb, out: Path, N: Numbers) -> None:
    """b26. L2-SP readout comparator (l2sp_comparator).

    l2sp_cv is the readout refitted toward the source head with C chosen by the
    same grouped nested CV as b13 over {1e-8..1e-5} + 1e-4..10; l2sp_c1 fixes
    C = 1; abstain50_l2sp_cv uses the transferred model at <= 50 studies and
    l2sp_cv above (nothing fitted). Scored on the same paired cells as q1
    (held-out) and q4 (left-out anatomy), read from the stored per-cell rows.
    """
    if l2 is None or not len(l2) or sh is None or fb is None:
        return
    K = ["target", "backbone", "seed", "budget", "resample"]
    unit = "cell (target x backbone x seed x budget x resample)"
    cols = K + ["setting", "harm", "delta_vs_zeroshot"]
    base = pd.concat([sh[cols], fb[cols]], ignore_index=True)
    base = base[base.backbone.isin(PAPER_BB) & (base["budget"] > 0)]
    l2 = l2[l2.backbone.isin(PAPER_BB)]
    N.set("l2sp_n_rows", int(len(l2)), "l2sp_comparator rows (both fitted arms)")
    N.set("l2sp_n_not_converged", int((l2.converged == 0).sum()),
          "L-BFGS fits that did not report convergence")

    for arm in ("heldout", "loao"):
        anat = base.target.astype(str).str.match(r"^XR_[A-Z]+$")
        b = base[anat] if arm == "loao" else base[~anat]
        common = None
        for s in ("shrink_lam0", "shrink_lam1", "refit_C1", "refit_nestedC",
                  "shrink_nested"):
            i = b[b.setting == s].set_index(K).index
            common = i if common is None else common.intersection(i)
        lam0 = b[b.setting == "shrink_lam0"].set_index(K).loc[common]
        cv = l2[l2.setting == "l2sp_cv"].set_index(K).reindex(common)
        c1 = l2[l2.setting == "l2sp_C1"].set_index(K).reindex(common)
        if cv.harm.isna().any() or c1.harm.isna().any():
            raise SystemExit(f"!! l2sp rows missing for {arm} cells; "
                             "l2sp_comparator must cover every paired cell")
        low = lam0.index.get_level_values("budget") <= 50
        ab = pd.DataFrame({"harm": np.where(low, lam0.harm, cv.harm),
                           "delta_vs_zeroshot": np.where(
                               low, lam0.delta_vs_zeroshot, cv.delta_vs_zeroshot)},
                          index=lam0.index)
        pols = {"l2sp_cv": cv, "l2sp_c1": c1, "abstain50_l2sp_cv": ab}
        for p, d in pols.items():
            h = d.harm.astype(int)
            loss = -d.delta_vs_zeroshot
            if arm == "heldout":
                k = f"rr_heldout_{p}"
                N.set(f"{k}_n_cells", int(len(d)), "held-out paired cells", unit)
                N.set(f"{k}_n_harmed", int(h.sum()),
                      "held-out cells below the transferred model", unit)
                N.set(f"{k}_mean_gain", float(d.delta_vs_zeroshot.mean()),
                      "held-out mean AUC gain over all cells", unit)
                if h.sum():
                    N.set(f"{k}_median_loss", float(loss[h == 1].median()),
                          "median AUC loss among harmed held-out cells", unit)
                    N.set(f"{k}_worst_loss", float(loss[h == 1].max()),
                          "largest AUC loss among harmed held-out cells", unit)
                for m in (0.0, 0.01, 0.02):
                    N.set(f"rr_margin_{p}_m{f'{m:g}'.replace('.', 'p')}",
                          int((loss > m).sum()),
                          f"held-out cells losing more than {m:g} AUC", unit)
                for bud, g in d.groupby(level="budget"):
                    N.set(f"{k}_n_harmed_at_{int(bud)}", int(g.harm.sum()),
                          "held-out harmed cells at this budget", unit)
            else:
                for t, g in d.groupby(level="target"):
                    a = str(t).lower().replace("xr_", "")
                    N.set(f"rr_loao_{a}_{p}_n_harmed", int(g.harm.sum()),
                          f"harmed cells on {t} when held out", unit)
                    N.set(f"rr_loao_{a}_{p}_harm_rate", float(g.harm.mean()),
                          f"harm rate on left-out {t}", unit)
                    N.set(f"rr_loao_{a}_{p}_mean_gain",
                          float(g.delta_vs_zeroshot.mean()),
                          f"mean AUC gain over all left-out {t} cells", unit)
                k = f"rr_loao_pooled_{p}"
                N.set(f"{k}_n_cells", int(len(d)), "left-out-anatomy cells", unit)
                N.set(f"{k}_n_harmed", int(h.sum()), "", unit)
                N.set(f"{k}_harm_rate", float(h.mean()), "pooled harm rate", unit)
                N.set(f"{k}_mean_gain", float(d.delta_vs_zeroshot.mean()),
                      "cell-weighted mean AUC gain", unit)
        if arm == "heldout":
            for bud, g in cv.groupby(level="budget"):
                N.set(f"rr_heldout_l2sp_cv_n_c_le_1e6_at_{int(bud)}",
                      int((g.chosen_C <= 1e-6).sum()),
                      "held-out cells where the CV chose C <= 1e-6 "
                      "(effectively the transferred model)", unit)
                N.set(f"rr_heldout_l2sp_cv_median_c_at_{int(bud)}",
                      float(g.chosen_C.median()), "median CV-chosen C", unit)


def fig_comparators(sh, fb, ls, l2, out: Path, N: Numbers) -> None:
    """b27. Comparator figure: BIRS against the other adaptation policies.

    Two stacked panels against labelled budget on the held-out external arm
    (the q1 pairing, 1,345 cells): harm rate on top with the tau line, mean AUC
    gain below. Abstention region shaded, prescribed lambda in the tick labels,
    as fig_lambda_schedule. Every plotted value is computed here from the
    registered per-cell inputs; values that already have keys are cross-checked
    against them and not re-keyed:
      BIRS            lamsched_test_harm_at_* (rate), lamsched_test_gain_at_*
      fixed 0.5       rr_heldout_lam0p5_n_harmed_at_*   (b25)
      L2-SP cv        rr_heldout_l2sp_cv_n_harmed_at_*  (b26)
    New keys: rr_heldout_n_cells_at_*, and rr_heldout_{p}_{n_harmed,mean_gain}_at_*
    for whatever is not keyed yet.
    """
    import matplotlib.pyplot as plt
    if any(x is None or not len(x) for x in (sh, fb, ls, l2)):
        print("  [fig] comparators: an input is missing; skipped")
        return
    K = ["target", "backbone", "seed", "budget", "resample"]
    unit = "cell (target x backbone x seed x budget x resample)"
    cols = K + ["setting", "harm", "delta_vs_zeroshot"]
    base = pd.concat([sh[cols], fb[cols]], ignore_index=True)
    base = base[base.backbone.isin(PAPER_BB) & (base["budget"] > 0)
                & ~base.target.astype(str).str.match(r"^XR_[A-Z]+$")]
    common = None
    for s in ("shrink_lam0", "shrink_lam1", "refit_C1", "refit_nestedC",
              "shrink_nested"):
        i = base[base.setting == s].set_index(K).index
        common = i if common is None else common.intersection(i)
    get = lambda d, s: d[d.setting == s].set_index(K).reindex(common)

    sl = ls[ls.rule == "schedule_lam"].sort_values("budget")
    lam_at = {int(r.budget): float(r.lam) for r in sl.itertuples()}
    tau = float(sl.tau.iloc[0])
    budgets = sorted(lam_at)
    bud = common.get_level_values("budget")
    birs = get(base, "shrink_lam0")[["harm", "delta_vs_zeroshot"]].copy()
    for b_, lam in lam_at.items():
        s = get(base, f"shrink_lam{lam:g}")
        m = np.asarray(bud == b_)
        birs.loc[m, ["harm", "delta_vs_zeroshot"]] = s.loc[m, ["harm", "delta_vs_zeroshot"]].to_numpy()

    series = [  # key stem, label, frame, colour, marker, linestyle, width
        ("birs", "BIRS", birs, "#2a78d6", "o", "-", 2.2),
        ("lam0p5", r"Fixed $\lambda$ = 0.5 (WiSE-FT default)",
         get(base, "shrink_lam0.5"), "#eb6834", "s", "--", 1.1),
        ("cv_lambda", r"Cross-validated $\lambda$",
         get(base, "shrink_nested"), "#1baf7a", "^", ":", 1.1),
        ("l2sp_cv", "L2-SP, cross-validated strength",
         get(l2, "l2sp_cv"), "#eda100", "D", "-.", 1.1),
        ("refit_tuned", "Refit, tuned penalty",
         get(base, "refit_nestedC"), "#e87ba4", "v", (0, (5, 1.5)), 1.1),
    ]
    reuse = {("birs", "harm_rate"): "lamsched_test_harm_at_{b}",
             ("birs", "mean_gain"): "lamsched_test_gain_at_{b}",
             ("lam0p5", "n_harmed"): "rr_heldout_lam0p5_n_harmed_at_{b}",
             ("l2sp_cv", "n_harmed"): "rr_heldout_l2sp_cv_n_harmed_at_{b}"}
    curves = {}
    for stem, _lab, d, *_ in series:
        if d.harm.isna().any():
            raise SystemExit(f"!! comparators: {stem} has missing paired cells")
        g = d.groupby(level="budget").agg(n=("harm", "size"),
                                          h=("harm", "sum"),
                                          gain=("delta_vs_zeroshot", "mean"))
        curves[stem] = g
        for b_ in budgets:
            vals = {"n_harmed": int(g.loc[b_, "h"]),
                    "harm_rate": float(g.loc[b_, "h"] / g.loc[b_, "n"]),
                    "mean_gain": float(g.loc[b_, "gain"])}
            for what, v in vals.items():
                k = reuse.get((stem, what))
                if k is not None:
                    k = k.format(b=b_)
                    if k not in N.d or abs(N.d[k]["value"] - v) > 1e-9:
                        raise SystemExit(f"!! comparators: {stem} {what} at {b_} "
                                         f"= {v!r} disagrees with {k}")
                    continue
                if what == "harm_rate" or (stem == "birs"):
                    continue          # rates follow from counts / n; BIRS fully reused
                N.set(f"rr_heldout_{stem}_{what}_at_{b_}", v,
                      f"held-out {what.replace('_', ' ')} at {b_} studies "
                      f"(Fig. 3 comparators)", unit)
    n0 = curves["birs"]
    for b_ in budgets:
        N.set(f"rr_heldout_n_cells_at_{b_}", int(n0.loc[b_, "n"]),
              "held-out paired cells at this budget", unit)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True)
    x = np.arange(len(budgets))
    abstain = [i for i, b_ in enumerate(budgets) if lam_at[b_] == 0.0]
    for ax in (ax1, ax2):
        if abstain:
            ax.axvspan(-0.5, max(abstain) + 0.5, color="0.92", zorder=0)
        ax.grid(axis="y", color="0.88", linewidth=0.4, zorder=0)
    for stem, lab, _d, col, mk, lsty, lw in reversed(series):
        g = curves[stem]
        z = 5 if stem == "birs" else 3
        kw = dict(color=col, marker=mk, linestyle=lsty, linewidth=lw,
                  markersize=4.5 if stem == "birs" else 3.5, label=lab, zorder=z,
                  markeredgecolor="white", markeredgewidth=0.5)
        ax1.plot(x, [100 * g.loc[b_, "h"] / g.loc[b_, "n"] for b_ in budgets], **kw)
        ax2.plot(x, [g.loc[b_, "gain"] for b_ in budgets], **kw)
    ax1.axhline(100 * tau, color="0.35", linewidth=0.7, linestyle="-.", zorder=2)
    ax1.text(len(budgets) - 0.55, 100 * tau, f"$\\tau$ = {100 * tau:g}%",
             va="bottom", ha="right", fontsize=6, color="0.3")
    ax1.set_ylabel("Harm rate (%)")
    ax1.set_ylim(bottom=-2)
    ax2.axhline(0.0, color="0.35", linewidth=0.6, zorder=2)
    ax2.set_ylabel("Mean AUC gain")
    ax2.set_xlabel("Labelled target studies")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b_}\n$\\lambda^*$={lam_at[b_]:g}" for b_ in budgets])
    if abstain:
        ax1.text(max(abstain) / 2.0, 0.97, "abstain", transform=ax1.get_xaxis_transform(),
                 ha="center", va="top", fontsize=6, color="0.3")
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=2, fontsize=5.6,
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
               columnspacing=1.0)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_comparators.pdf")
    fig.savefig(out / "fig_comparators.png", dpi=600)
    plt.close(fig)
    print("  [fig] fig_comparators.pdf / .png (600 dpi)")


def numbers_foundation_arm(prov: dict, N: Numbers) -> None:
    """b28. Foundation-encoder arm (analysis plan in Supplementary File 2, with
    amendments A1 and A2): SKELEX, RAD-DINO, MedSigLIP-448 and DINOv3-B,
    fine-tuned on MURA wrist (arm A) and frozen with a linear readout (arm B),
    each on exactly the DenseNet-121 cell keys of the paper.

    The values are computed by foundation_arm/analysis/foundation_summary.py, which first
    reproduces the paper's own schedule, comparator, abstain-50, LOO and
    own-penalty numbers from the transfer cells (its gate 1), and are checked by
    foundation_summary_verify.py, an independent second pass. This block registers them
    after three checks of its own:
      * every FA cell CSV under $FA_ROOT/cells has the sha256 the json declares;
      * foundation_summary.py and foundation_summary_verify.py sit beside the json and are hashed into
        the provenance with it;
      * the headline values per (arm, encoder) -- BIRS schedule, its external
        harm and gain, LOO harm, the published schedule's external and anatomy
        harm and gain, the joint-penalty refit's harm -- are re-derived here
        from the cells and must equal the json's.
    A confidence interval stored as a pair becomes two keys, _lo and _hi, so
    the drift check covers both ends. FA_ROOT defaults to ../foundation_arm, the
    repository layout when run from code/.
    """
    import os

    root = Path(os.environ.get("FA_ROOT", "../foundation_arm"))
    adir = root / "analysis"
    jp = adir / "foundation_summary.json"
    need = [jp, adir / "foundation_summary.py", adir / "foundation_summary_verify.py"]
    absent = [str(p) for p in need if not p.exists()]
    if absent:
        raise SystemExit(f"!! b28: missing {absent}. Set FA_ROOT to the folder "
                         "holding analysis/ and cells/.")
    J = json.loads(jp.read_text(encoding="utf-8"))
    for p in need:
        prov[p.stem] = {"path": str(p), "relative": None,
                        "nominal": f"$FA_ROOT/analysis/{p.name}",
                        "resolved_by_search": False, "sha256": sha256(p),
                        "rows": len(J["numbers"]) if p == jp else 0}

    frames = []
    for name, want in J["inputs"]["budget_cells"].items():
        p = root / "cells" / name
        got = sha256(p)
        if got != want:
            raise SystemExit(f"!! b28: {name} has sha256 {got[:12]}, but the "
                             f"foundation_summary json was computed from {want[:12]}")
        df = pd.read_csv(p, comment="#", low_memory=False)
        prov[f"fa_{p.stem}"] = {"path": str(p), "relative": None,
                                "nominal": f"$FA_ROOT/cells/{name}",
                                "resolved_by_search": False, "sha256": got,
                                "rows": int(len(df))}
        frames.append(df.assign(arm="a" if "_arm_a" in name else "b"))
    c = pd.concat(frames, ignore_index=True)
    c = c[c.setting != "zeroshot"]

    # re-derive the headline values from the cells
    jv = {k: v["value"] for k, v in J["numbers"].items()}
    cal = {"XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER"}
    ext = {"GRAZ_initial", "MFXR", "FRAC_XR_HAND"}
    bud = (10, 25, 50, 100, 200, 500)
    pub = dict(zip(bud, (0.0, 0.0, 0.0, 0.1, 0.75, 1.0)))
    col = lambda lam: f"shrink_lam{lam:g}"
    unit = ["target", "seed", "budget", "resample"]
    checked = 0

    def same(key, v):
        nonlocal checked
        ref = jv[key]
        ok = ref == v if isinstance(v, str) else abs(ref - v) <= 1e-12
        if not ok:
            raise SystemExit(f"!! b28: {key} is {ref!r} in the json, {v!r} from the cells")
        checked += 1

    for (arm, bb), g in c.groupby(["arm", "backbone"]):
        assert not g.duplicated(unit + ["setting"]).any(), (arm, bb)
        w = g.set_index(unit + ["setting"]).delta_vs_zeroshot.unstack("setting").reset_index()
        t = f"b28_{arm}_{bb.replace('-', '')}"

        def schedule(targets):
            s, prev = {}, 0.0
            for n in bud:
                q = w[w.target.isin(targets) & (w.budget == n)]
                best, bm = 0.0, None
                for lam in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):   # ascending: ties keep the smaller weight
                    v = q[col(lam)]
                    if (v < 0).mean() <= 0.05 and (bm is None or v.mean() > bm + 1e-12):
                        best, bm = lam, v.mean()
                prev = max(prev, best if bm is not None and bm > 0 else 0.0)
                s[n] = prev
            return s

        def apply(s, targets):
            q = w[w.target.isin(targets)]
            d = pd.concat([q.loc[q.budget == n, col(s[n])] for n in bud])
            return int((d < 0).sum()), float(d.mean())

        s = schedule(cal)
        same(f"{t}_o2_schedule", ", ".join(f"{n}:{s[n]:g}" for n in bud))
        k, m = apply(s, ext)
        same(f"{t}_o2_ext_harmed_cells", k)
        same(f"{t}_o2_ext_mean_gain", m)
        same(f"{t}_o2_loo_n_harmed", sum(apply(schedule(cal - {a}), {a})[0] for a in cal))
        for lab, tg in (("ext", ext), ("cal", cal)):
            k, m = apply(pub, tg)
            same(f"{t}_o3_published_{lab}_n_harmed", k)
            same(f"{t}_o3_published_{lab}_mean_gain", m)
            r = w[w.target.isin(tg)]
            same(f"{t}_o1_refit_{lab}_all_n_harmed", int((r.refit_nestedC < 0).sum()))
            same(f"{t}_o1_refit_{lab}_le50_n_harmed", int((r.refit_nestedC[r.budget <= 50] < 0).sum()))
    if checked != 96:
        raise SystemExit(f"!! b28: expected 96 re-derived values (8 arm-encoder pairs x 12), got {checked}")
    print(f"  [b28] cells match the json's sha256; {checked} headline values re-derived from the cells")

    n0 = len(N.d)
    for k, v in J["numbers"].items():
        if isinstance(v["value"], list):
            lo, hi = v["value"]
            N.set(f"{k}_lo", lo, v["note"] + " (lower bound)")
            N.set(f"{k}_hi", hi, v["note"] + " (upper bound)")
        else:
            N.set(k, v["value"], v["note"])
    print(f"  [b28] {len(N.d) - n0} foundation-arm values registered from {jp.name}")


def check_drift(jpath: Path, N: Numbers) -> None:
    """Refuse to write anything if a previously recorded value has moved.

    Called before the tables are emitted, not after. A drifting run must leave
    the working tree exactly as it found it, otherwise the guard protects the
    registry and corrupts the artefacts the registry is supposed to certify.
    """
    if not jpath.exists():
        return
    old = json.loads(jpath.read_text()).get("numbers", {})
    drift = [k for k, v in N.d.items()
             if k in old and isinstance(v["value"], float)
             and isinstance(old[k]["value"], float)
             and abs(v["value"] - old[k]["value"]) > TOL]
    if drift:
        detail = "\n".join(
            f"    {k}: recorded {old[k]['value']!r} -> now {N.d[k]['value']!r}"
            for k in drift[:12])
        raise SystemExit(
            f"NON-DETERMINISTIC REGENERATION: {len(drift)} value(s) moved\n"
            f"{detail}\n"
            "Nothing has been written. Regeneration must be bit-stable, so "
            "either an input changed or a definition did. If an input changed, "
            "check it was not appended to twice. If a definition changed, the "
            "key should change name.")
    gone = sorted(k for k in old if k not in N.d)
    if gone:
        raise SystemExit(
            f"KEYS WOULD DISAPPEAR: {len(gone)} key(s) recorded in the previous "
            f"paper_numbers.json are not produced by this run, e.g. {gone[:8]}\n"
            "Nothing in --out has changed (outputs are staged). "
            "A key vanishes when an input goes "
            "missing; this is how 91 boot_* keys were lost silently before "
            "2026-09-24. Restore the input, or, if the removal is intended, "
            "move the old paper_numbers.json aside first.")
    compared = sum(1 for k in old if k in N.d)
    print(f"\n  [check] {compared} previous values reproduced within {TOL}; "
          f"{len(N.d) - compared} new")


def numbers_bootstrap(path, N: Numbers) -> None:
    """Register the patient-clustered intervals from bootstrap_intervals.py.

    Optional: skipped until that script has been run. Once present, every
    interval traces to a registered value like every other number.
    """
    if path is None:
        print("  [boot] WARNING: no bootstrap_ci.csv under the results root; "
              "NO boot_* intervals registered (the 2026-08-13 run had 91). "
              "Run bootstrap_intervals.py --out <results-root>/transfer/"
              "bootstrap_ci.csv, or pass --bootstrap.")
        return
    b = path if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    need = {"scope", "term", "estimate", "lo", "hi"}
    if not need.issubset(b.columns):
        print(f"  [!] {path} lacks {sorted(need - set(b.columns))}; skipped")
        return
    for r in b.itertuples():
        s = str(r.scope).lower().replace("-", "_")
        for suffix, v in (("", r.estimate), ("_lo", r.lo), ("_hi", r.hi)):
            N.set(f"boot_{s}_{r.term}{suffix}", float(v),
                  "patient-clustered bootstrap, percentile interval; the "
                  "resample is drawn once and applied to every cell and "
                  "condition, so the pairing is preserved",
                  "(target, backbone, seed) cells, patients resampled")
    if "n_boot" in b.columns and len(b):
        N.set("boot_n_resamples", float(b.n_boot.iloc[0]), "resamples")
    print(f"  [boot] {len(b)} interval(s) registered from bootstrap_ci.csv")


def numbers_budgets_allbb(creg: pd.DataFrame, N: Numbers) -> None:
    """The same harm rates with ResNet50 included.

    ResNet50 is a complete arm of the budget sweep and is excluded from the
    reported set because it shares the CNN/batch-norm cell with DenseNet121.
    That exclusion is not neutral, so the four-backbone figures are registered
    as well.
    """
    d = creg[creg.setting.astype(str).isin(["nested", "nestedCV"])]
    zs = (creg[creg.budget == 0][["backbone", "seed", "target", "auc_study"]]
          .rename(columns={"auc_study": "zs"}).drop_duplicates())
    m = d[d.budget > 0].merge(zs, on=["backbone", "seed", "target"])
    cell = (m.groupby(["backbone", "seed", "target", "budget"], as_index=False)
              .agg(auc=("auc_study", "mean"), zs=("zs", "first")))
    cell["harm"] = cell.auc < cell.zs
    for b in sorted(cell.budget.unique()):
        s = cell[cell.budget == b]
        N.set(f"allbb_refit_harmful_frac_at_{b}", s.harm.mean(),
              f"fraction of cells where refitting on {b} labelled studies is "
              f"worse than the source head, ALL FOUR backbones including "
              f"ResNet50, CV-chosen penalty",
              "cell (target x backbone x seed), resamples averaged")
    N.set("allbb_refit_n_cells_per_budget",
          float(len(cell[cell.budget == cell.budget.min()])))
    N.set("allbb_n_backbones", float(creg.backbone.nunique()))


def numbers_matched_comparison(shortcut: pd.DataFrame, zeroshot: pd.DataFrame,
                               N: Numbers, caliper: float = 0.03) -> None:
    """The primary Stage B comparison of the analysis plan: compared cells are
    matched within a 0.03 study-AUC caliper.

    The result is a null, and the Swin cells fall outside the support range
    the plan declared in advance; both are reported.

    Unit: one cell per (backbone, source, target) with seeds averaged.
    Matching is within backbone.
    """
    c = shortcut[shortcut.source != shortcut.target]
    pool = (c.groupby(["backbone", "source", "target"], as_index=False)
             .agg(auc=("zs_auc_study", "mean"), delta=("delta_study", "mean")))
    lo, hi = float(pool.auc.min()), float(pool.auc.max())
    N.set("matched_support_lo", lo, "lowest cross-anatomy cell AUC; Amendment 2 "
          "pre-declared that a GRAZ cell outside this range cannot be compared")
    N.set("matched_support_hi", hi, "highest cross-anatomy cell AUC")
    N.set("matched_caliper", caliper, "study AUC caliper half-width")
    N.set("matched_n_pool_cells", float(len(pool)),
          "cross-anatomy cells available for matching",
          "(backbone, source, target), seeds averaged")

    gz = (zeroshot.groupby(["prep", "score_label", "backbone"], as_index=False)
                  .agg(auc=("zs_auc_study", "mean"), delta=("delta_study", "mean")))
    advs, n_out = {}, 0
    for r in gz.itertuples():
        p = pool[(pool.backbone == r.backbone) &
                 ((pool.auc - r.auc).abs() <= caliper)]
        if not len(p):
            continue
        adv = float(r.delta - p.delta.mean())
        insup = lo <= r.auc <= hi
        n_out += (not insup)
        advs[(r.prep, r.score_label, r.backbone)] = adv
        if r.prep == "standard" and r.score_label == "label_abnormal":
            bb = r.backbone
            N.set(f"matched_advantage_{bb}", adv,
                  f"{BACKBONE_LABEL.get(bb, bb)}: GRAZ projection delta minus "
                  f"the mean delta of cross-anatomy cells within the caliper. "
                  f"Standard prep, abnormal label.",
                  "(backbone, source, target) cell, seeds averaged")
            N.set(f"matched_n_{bb}", float(len(p)),
                  "cross-anatomy cells inside the caliper for this GRAZ cell")
            N.set(f"matched_graz_auc_{bb}", float(r.auc),
                  "GRAZ zero-shot study AUC used for matching")
    if advs:
        N.set("matched_advantage_min", min(advs.values()),
              "most negative advantage across all preprocessing and label arms")
        N.set("matched_advantage_max", max(advs.values()),
              "most positive advantage across all preprocessing and label arms")
        N.set("matched_n_arms", float(len(advs)), "GRAZ cells compared")
        N.set("matched_n_out_of_support", float(n_out),
              "GRAZ cells whose AUC falls outside the pre-declared support "
              "range, for which Amendment 2 says the comparison cannot be made")
        print(f"  [prereg] matched advantage spans "
              f"{min(advs.values()):+.4f} to {max(advs.values()):+.4f}; "
              f"{n_out} of {len(advs)} cells out of support")


def numbers_budgets_nested(creg: pd.DataFrame, out: Path, N: Numbers) -> None:
    """PRIMARY budget curves at the cross-validated penalty. Wrist source."""
    grid, nest = _split_creg(creg)
    if not len(nest):
        return
    budgets = sorted(b for b in nest.budget.unique() if b > 0)
    zs = (nest[nest.budget == 0][["backbone", "seed", "target", "auc_study"]]
          .rename(columns={"auc_study": "zs"}).drop_duplicates())
    if not len(zs):                       # zero-shot rows are tagged separately
        zs = (creg[(creg.budget == 0) & creg.backbone.isin(PAPER_BB)]
              [["backbone", "seed", "target", "auc_study"]]
              .rename(columns={"auc_study": "zs"}).drop_duplicates())
    m = nest[nest.budget > 0].merge(zs, on=["backbone", "seed", "target"])
    cell = (m.groupby(["backbone", "seed", "target", "budget"], as_index=False)
              .agg(auc=("auc_study", "mean"), zs=("zs", "first")))
    cell["harm"] = cell.auc < cell.zs
    cell["gain"] = cell.auc - cell.zs

    for b in budgets:
        s = cell[cell.budget == b]
        N.set(f"nest_refit_harmful_frac_at_{b}", s.harm.mean(),
              f"fraction of cells where refitting on {b} labelled studies is "
              f"worse than the source head, at the CV-chosen penalty",
              "cell (target x backbone x seed), resamples averaged")
    N.set("nest_refit_n_cells_per_budget", float(len(cell[cell.budget == budgets[0]])))

    rows = []
    for t, g in cell.groupby("target"):
        k = tkey(t)
        mu = {b: g[g.budget == b].gain.mean() for b in budgets}
        be = next((b for i, b in enumerate(budgets)
                   if all(mu[c] > 0 for c in budgets[i:])), None)
        rows.append({"target": t, "break_even": be,
                     **{f"gain_{b}": mu[b] for b in budgets}})
        N.set(f"nest_breakeven_wrist_{k}", float(be) if be else -1.0,
              "smallest labelled-study budget at which refitting beats zero "
              "shot and stays ahead; -1 means it never does", "(target)")
        N.set(f"nest_budget500_gain_wrist_{k}", mu[budgets[-1]],
              f"mean AUC change at {budgets[-1]} labelled studies", "(target)")
    pd.DataFrame(rows).to_csv(out / "table_break_even_nested.csv", index=False)
    N.set("nest_n_never_breakeven",
          float(sum(1 for r in rows if r["break_even"] is None)),
          "targets that never break even at the CV-chosen penalty")


# ------------------------------------------------------------------ main ----

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=str(Path.home() / "diagnosis" / "results"))
    ap.add_argument("--out", default="./paper")
    ap.add_argument("--bootstrap", default=None,
                    help="override for the registered input "
                         "transfer/bootstrap_ci.csv under --results-root")
    ap.add_argument("--check", action="store_true",
                    help="compare against an existing paper_numbers.json and exit")
    ap.add_argument("--allow-missing", action="store_true",
                    help="emit partial output instead of failing (NOT for submission)")
    cfg = ap.parse_args()

    style()
    root, final_out = Path(cfg.results_root).expanduser(), Path(cfg.out)
    final_out.mkdir(parents=True, exist_ok=True)
    # b24: stage every output; move into --out only after check_drift passes
    import atexit, os, shutil, tempfile
    out = Path(tempfile.mkdtemp(prefix=f".{final_out.resolve().name}.staging_",
                                dir=final_out.resolve().parent))
    atexit.register(shutil.rmtree, out, True)
    print(f"staging outputs in {out}")
    print(f"reading from {root}")
    data, prov, missing = load(root)

    if missing:
        print("\n!! MISSING REQUIRED INPUTS "
              f"(not at the nominal path and not found by name anywhere under {root}):")
        for k, rel in missing:
            print(f"   {k:<16} {rel}")
            print(f"   {'':<16} supplies: {SUPPLIES.get(k, 'see INPUTS')}")
        if not cfg.allow_missing:
            raise SystemExit(
                "\nRefusing to emit partial numbers. Every figure and every value in "
                "the manuscript must come from a present, hashed file (plan §12.1).\n"
                "Copy the file anywhere under the results root; it will be found.\n"
                "Use --allow-missing ONLY for a dry run; never for submission.")
        print("   --allow-missing set: continuing with a PARTIAL run.\n")

    N = Numbers()
    ceil = ceilings(data["source_models"]) if "source_models" in data else pd.Series(dtype=float)
    graz_ceil, graz_ceil_src = graz_ceiling_map(data)

    print("\nfigures")
    if "creg_probe" in data:
        fig_budget(data["creg_probe"], out, N)
    if {"probe_wrist", "probe_hand", "probe_humerus"} & set(data) and len(ceil):
        cells = probe_cells(data, ceil, graz_ceil)
        cells.to_csv(out / "probe_cells.csv", index=False)
        q = cells.dropna(subset=["recovery"])
        for arm in ("anatomy", "dataset"):
            s = q[q.arm == arm]
            if len(s):
                N.set(f"probe_recovery_{arm}", s.recovery.mean(),
                      f"mean readout recovery at the largest budget, {arm} shift",
                      "cell (source x target x backbone x seed), gap >= 0.10")
                N.set(f"probe_recovery_{arm}_n", int(len(s)))
        for bb, s in q[q.arm == "anatomy"].groupby("backbone"):
            N.set(f"probe_recovery_anatomy_{bb}_n", int(len(s)),
                  "gated cells contributing for this backbone; NOT the pooled count")
            N.set(f"probe_recovery_anatomy_{bb}", s.recovery.mean(),
                  f"{BACKBONE_LABEL.get(bb, bb)}", "cell")
        by = q[q.arm == "anatomy"].groupby("backbone").recovery.mean()
        if len(by) > 1:
            N.set("probe_recovery_anatomy_backbone_spread", by.max() - by.min(),
                  "max-min across backbones; small means the mechanism is "
                  "independent of architecture AND normalisation")
    if "angle" in data:
        fig_decomposition(data["angle"], out, N, data.get("creg_oracle"))
    if "diagonal" in data:
        fig_diagonal(data["diagonal"], out, N)
    fig_graz_composition(data.get("strata"), data.get("easy"), out, N)
    if "lam_sched" in data:
        fig_lambda_schedule(data["lam_sched"], out, N)

    if "frac_stats" in data:
        f = data["frac_stats"].iloc[0]
        for k in ("matched_mean", "mismatched_mean", "difference",
                  "p_matched_less", "p_difference_permutation"):
            if k in f:
                N.set(f"fracatlas_{k}", f[k], "FracAtlas hand target, within-target "
                      "contrast across source anatomies", "cell (backbone x seed)")

    print("\nnumbers")
    if "angle" in data:
        numbers_oracle(data["angle"], out, N)
    if {"probe_wrist", "probe_hand", "probe_humerus"} & set(data):
        numbers_budgets(data, out, N)
    if "step1" in data:
        numbers_gating_null(data["step1"], N)
    if "zeroshot" in data:
        numbers_prep_arms(data["zeroshot"], N)
    numbers_graz_audit(data, N)
    if "confounds" in data:
        numbers_confounds(data["confounds"], N)
    if "fracatlas" in data:
        numbers_fracatlas(data["fracatlas"], out, N)
    if "shortcut" in data:
        numbers_shortcut(data["shortcut"], N)
    if "source_models" in data:
        numbers_gaps(data["source_models"], N)
    if "diagonal" in data:
        numbers_diagonal_per_anatomy(data["diagonal"], out, N)
    numbers_datasets(data, out, N)
    if "creg_oracle" in data:
        numbers_oracle_nested(data["creg_oracle"], out, N)
    if "creg_probe" in data:
        numbers_budgets_nested(data["creg_probe"], out, N)
        # The four-backbone figures, so the ResNet50 exclusion is quantified in
        # the paper rather than left to a reader who opens the sweep file.
        numbers_budgets_allbb(data["creg_probe"], N)
    if "shortcut" in data and "zeroshot" in data:
        numbers_matched_comparison(data["shortcut"], data["zeroshot"], N)
    if "creg_oracle" in data:
        numbers_cell_separation(data["creg_oracle"], N)
    numbers_backbone_coverage(data, N)
    if "creg_probe" in data and "creg_oracle" in data:
        # Registers the cross-validated-penalty values used by the backbone
        # table; without them it would fall back to the fixed-penalty values.
        numbers_backbones_nested(data["creg_probe"], data["creg_oracle"], N)
    if "src_prep" in data:
        numbers_source_pool(data["src_prep"], N)
    if "layerwise" in data:
        numbers_not_a_bound(data["layerwise"], ceil, N)
    if "cohorts" in data:
        numbers_cohorts(data["cohorts"], out, N)
    if "bitdepth" in data:
        numbers_bitdepth(data["bitdepth"], N)
    if "shrink" in data:
        numbers_shrinkage(data["shrink"], out, N)
    if "frac_budget" in data:
        numbers_frac_budget(data["frac_budget"], out, N)
    if "mfxr" in data:
        numbers_mfxr(data["mfxr"], data.get("creg_oracle"), out, N)
    if "lam_sched" in data:
        numbers_lambda_schedule(data["lam_sched"], out, N)
    numbers_label_audit(data.get("graz_labels"), data.get("frac_budget"),
                        out, N)
    numbers_harm_analyses(data.get("rr_harm"), data.get("rr_margin"),
                              data.get("rr_loao"), data.get("rr_worst"),
                              out, N)
    numbers_review6(data.get("lam_sched"), data.get("shrink"),
                    data.get("frac_budget"), data.get("rr_harm"),
                    data.get("rr_margin"), data.get("rr_worst"), out, N)
    numbers_review7(data.get("lam_sched"), data.get("shrink"),
                    data.get("rr_harm"), out, N)
    numbers_bspc_hybrids(data.get("shrink"), data.get("frac_budget"),
                         data.get("lam_sched"), data.get("rr_harm"),
                         data.get("rr_margin"), out, N)
    numbers_l2sp(data.get("l2sp"), data.get("shrink"),
                 data.get("frac_budget"), out, N)
    fig_comparators(data.get("shrink"), data.get("frac_budget"),
                    data.get("lam_sched"), data.get("l2sp"), out, N)
    if cfg.bootstrap:  # explicit override of the registered input
        bp = Path(cfg.bootstrap).expanduser()
        data["bootstrap"] = pd.read_csv(bp, comment="#")
        prov["bootstrap"] = {"path": str(bp), "relative": None,
                             "nominal": "--bootstrap",
                             "resolved_by_search": False,
                             "sha256": sha256(bp),
                             "rows": int(len(data["bootstrap"]))}
    numbers_bootstrap(data.get("bootstrap"), N)
    crosscheck_source_pool(N)
    numbers_foundation_arm(prov, N)   # b28

    # The drift check has to happen BEFORE anything is written. It used to sit
    # just above the json write, after the tables had already been emitted, so
    # a run that drifted left corrupted table*.tex on disk next to a json the
    # guard had refused to update. That is how duplicated rows in the oracle
    # csv reached Table 2: n_g doubled, the guard fired, the json was protected
    # and the tables were not.
    check_drift(final_out / "paper_numbers.json", N)
    emit_latex_tables(N, out)
    emit_tables_6_to_9(N, out)

    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": _sklearn_version(),
        "results_root": str(root),
        "inputs": prov,
        "graz_ceiling_source_per_backbone": graz_ceil_src,
        "missing_required_inputs": [rel for _, rel in missing],
        "partial_run": bool(missing),
        "forbidden_statistics": {
            "significance_test_on_shift_type_axis":
                "NOT COMPUTED. The GRAZ arm is ONE dataset comparison; its "
                "backbone x seed replicates are pseudoreplication. The "
                "previously circulated p = 1.3e-09 must not be quoted. Report "
                "effect size, ratio to the WORST anatomy, and per-backbone / "
                "per-seed consistency instead.",
            "recovery_fraction_below_gap_0.10":
                "NOT COMPUTED. Undefined or explosive on small gaps.",
            "rank_permutation_with_few_groups":
                "NOT COMPUTED. Floor of 1/k; with 7 groups it cannot reject at 0.05.",
        },
        "numbers": N.d,
    }

    jpath = final_out / "paper_numbers.json"
    if jpath.exists() and cfg.check:
        print("  --check passed; nothing written"); return
    (out / "paper_numbers.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True))
    moved = 0
    for f in sorted(p for p in out.rglob("*") if p.is_file()):
        dest = final_out / f.relative_to(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(f, dest)
        moved += 1
    print(f"\nwrote {jpath}  ({len(N.d)} values; {moved} files moved "
          f"into {final_out})")
    if missing:
        print("PARTIAL RUN — not suitable for submission.")


if __name__ == "__main__":
    main()
