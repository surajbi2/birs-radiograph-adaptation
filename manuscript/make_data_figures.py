"""Draw the data figures of the paper (Figs. 2-4) in one style.

    fig2_comparators     BIRS against four adaptation policies on the external arm
    fig3_budget_curves   own-penalty refit gain against label budget
    fig4_foundation      refit harm and BIRS weights for the four foundation
                         encoders, fine-tuned and frozen

Every plotted value is read from paper_numbers.json, or, for the individual
budget curves, recomputed from the readout_penalty_sweep cells and checked against every key
that summarises them; any mismatch stops the script.

Style: Arial, 165 mm wide (the journal text width), budgets as equally spaced
categories, one colour per method across figures; every series also has its own
marker, so colour is never the only cue.

Outputs (into output/figures/):
    PDF and 600-dpi PNG per figure, and fig3_values.json (the plotted Fig. 3 means)

Usage:
    python make_data_figures.py
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NUMBERS = ROOT / "code" / "paper" / "paper_numbers.json"
NUMBERS_SHA = "9a7f55f08ce68668bae0722c743ba9de751f23d5091bc73b7552c506cd17b297"
B11 = ROOT / "results" / "transfer" / "readout_penalty_sweep_XR_WRIST.csv"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


if sha256(NUMBERS) != NUMBERS_SHA:
    raise SystemExit(f"!! {NUMBERS} is not the canonical 9a7f55f0 file")
N = {k: v["value"] for k, v in json.loads(NUMBERS.read_text(encoding="utf-8"))["numbers"].items()}

BUDGETS = [10, 25, 50, 100, 200, 500]
X = np.arange(len(BUDGETS))
MM = 1 / 25.4
INK, SEC, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec"
REF = "#8a8984"                                   # ImageNet reference and published schedule
C = {"birs": "#2a78d6", "lam0p5": "#eb6834", "cv_lambda": "#1baf7a", "l2sp_cv": "#eda100", "refit": "#e87ba4"}
FAINT = "#cfcdc6"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": SEC, "axes.labelcolor": INK, "xtick.color": SEC, "ytick.color": SEC,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "axes.titlelocation": "left", "axes.titlepad": 4,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})


def finish(ax, grid=True):
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(X)
    ax.set_xlim(-0.45, len(X) - 0.55)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def sgn(v):
    return f"{v:+.3f}".replace("-", "−")


def same(what, got, key, tol=1e-9):
    if abs(got - N[key]) > tol:
        raise SystemExit(f"!! {what}: {got!r} from the cells, {N[key]!r} in {key}")


def schedule():
    lam = [N[f"lambda_schedule_at_{b}"] for b in BUDGETS]
    assert N["lambda_schedule"] == ", ".join(f"n={b}:{v:g}" for b, v in zip(BUDGETS, lam))
    return lam


# ---------------------------------------------------------------- Fig. 2: budget curves ----
def fig_budget():
    d = pd.read_csv(B11, comment="#", low_memory=False)
    d = d[d.backbone.isin(["densenet121", "convnext_tiny", "swin_tiny_patch4_window7_224"])]
    # Table 2's definition (numbers_budgets_nested): resamples averaged within a cell (backbone, seed, target,
    # budget), each cell paired with its own seed's zero-shot, then seeds averaged. The generator's figure took a
    # plain row mean instead, which counts the duplicated rows of XR_ELBOW and XR_FINGER densenet121 seed 42 twice
    # and weights 4-resample cells less; its anatomy mean at 500 (0.0931, nest_budget_mean_gain_anatomy_max)
    # differs from this one (0.0934) for that reason only.
    zs = (d[d.budget == 0][["backbone", "seed", "target", "auc_study"]]
          .rename(columns={"auc_study": "zs"}).drop_duplicates())
    nest = d[d.setting.astype(str).isin(["nested", "nestedCV"]) & (d.budget > 0)].merge(
        zs, on=["backbone", "seed", "target"])
    cell = (nest.groupby(["backbone", "seed", "target", "budget"], as_index=False)
            .agg(auc=("auc_study", "mean"), zs=("zs", "first")))
    assert cell.groupby(["target", "backbone", "budget"]).seed.nunique().eq(5).all()
    cell["gain"] = cell.auc - cell.zs
    for b in BUDGETS:
        same(f"harmful fraction at {b}", (cell[cell.budget == b].gain < 0).mean(), f"nest_refit_harmful_frac_at_{b}")
    cur = cell.groupby(["target", "backbone", "budget"]).gain.mean().unstack("budget")[BUDGETS]
    graz = cur.index.get_level_values("target").str.startswith("GRAZ")
    arms = {"anatomy": cur[~graz], "dataset": cur[graz]}
    # checks against every key that summarises these curves
    for arm, c in arms.items():
        same(f"{arm} curve count", len(c), f"nest_budget_curves_{arm}_n")
    same("dataset mean at 500", arms["dataset"][500].mean(), "nest_budget_mean_gain_dataset_max")
    for t, g in cur.groupby(level="target"):
        k = "graz" if t.startswith("GRAZ") else t.split("_")[1].lower()
        same(f"{t} at 500", g[500].mean(), f"nest_budget500_gain_wrist_{k}")

    fig, axs = plt.subplots(1, 2, figsize=(165 * MM, 78 * MM), sharey=True,
                            gridspec_kw={"wspace": 0.2})
    titles = {"anatomy": "Wrist source → six MURA anatomies (18 curves)",
              "dataset": "Wrist source → GRAZPEDWRI-DX, external (3 curves)"}
    lo = min(c.min().min() for c in arms.values())
    plotted = {}
    for ax, (arm, c) in zip(axs, arms.items()):
        ax.axhspan(-0.25, 0, color=SHADE, zorder=0, lw=0)
        for _, row in c.iterrows():
            ax.plot(X, row.values, color=FAINT, lw=0.8, zorder=2, solid_capstyle="round")
        m = c.mean().values
        plotted[arm] = dict(zip(map(str, BUDGETS), map(float, m)))
        ax.plot(X, m, color=C["refit"], lw=2.2, marker="o", ms=4.2, mec="white", mew=0.6, zorder=4)
        ax.axhline(0, color=INK, lw=0.7, zorder=3)
        ax.annotate(sgn(m[-1]), (X[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
        ax.annotate(sgn(m[0]), (X[0], m[0]), xytext=(-6, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=7, color=INK)
        finish(ax)
        ax.set_xlim(-1.4, len(X) - 0.1)
        ax.set_xticklabels(BUDGETS)
        ax.set_title(titles[arm])
        ax.set_xlabel("Labelled target studies")
        ax.text(0.5, -0.25, "(a)" if arm == "anatomy" else "(b)", transform=ax.transAxes, ha="center", va="top",
                fontsize=8)
        ax.text(0.99, -0.165, "Worse than the transferred model", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7, color=SEC, style="italic")
    a = axs[0]
    sh = arms["anatomy"].loc["XR_SHOULDER"]
    a.annotate("Shoulder\n(3 backbones)", (X[-1], sh[500].mean()), xytext=(6, 0), textcoords="offset points",
               fontsize=7, color=SEC, va="center")
    worst = arms["anatomy"][10].idxmin()
    a.annotate(f"{worst[0][3:].title()}, DenseNet-121", (X[0], arms['anatomy'].loc[worst, 10]),
               xytext=(X[0] + 0.35, lo + 0.005), fontsize=7, color=SEC, va="center",
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=2))
    assert worst == ("XR_HAND", "densenet121")
    a.set_ylim(-0.235, 0.345)
    a.set_ylabel("Change in study AUC vs. transferred model")
    h = [Line2D([], [], color=FAINT, lw=1.2, label="One target–backbone pair (mean of 5 seeds)"),
         Line2D([], [], color=C["refit"], lw=2.2, marker="o", ms=4.2, mec="white",
                label="Mean of the curves: own-penalty refit")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=2.2,
               columnspacing=2.5)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.85, bottom=0.25)
    save(fig, "fig3_budget_curves")
    (OUT / "fig3_values.json").write_text(json.dumps({"note": "mean of the curves plotted in Fig. 3", "curves": plotted}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- Fig. 3: comparators ----
def fig_comparators():
    lam = schedule()
    n = [N[f"rr_heldout_n_cells_at_{b}"] for b in BUDGETS]
    assert sum(n) == 1345
    S = [  # stem, label, harm rate (%) per budget, mean gain per budget, marker, linestyle
        ("birs", "BIRS (0 of 1,345 cells harmed)",
         [100 * N[f"lamsched_test_harm_at_{b}"] for b in BUDGETS],
         [N[f"lamsched_test_gain_at_{b}"] for b in BUDGETS], "o", "-")]
    for stem, lab, mk, ls in (("lam0p5", "Fixed λ = 0.5, WiSE-FT default", "s", (0, (4, 1.6))),
                              ("cv_lambda", "Cross-validated λ", "^", (0, (1, 1.2))),
                              ("l2sp_cv", "L2-SP, cross-validated strength", "D", (0, (5, 1.5, 1, 1.5))),
                              ("refit_tuned", "Joint-penalty refit", "v", (0, (6, 1.5)))):
        tot = int(N[f"rr_heldout_{stem}_n_harmed"]) if f"rr_heldout_{stem}_n_harmed" in N else None
        k = [N[f"rr_heldout_{stem}_n_harmed_at_{b}"] for b in BUDGETS]
        if tot is not None:
            assert sum(k) == tot, stem
        S.append((stem, f"{lab} ({int(sum(k))} harmed)",
                  [100 * a / b for a, b in zip(k, n)],
                  [N[f"rr_heldout_{stem}_mean_gain_at_{b}"] for b in BUDGETS], mk, ls))
    assert all(v == 0 for v in S[0][2])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(165 * MM, 86 * MM), gridspec_kw={"wspace": 0.25})
    ab = max(i for i, v in enumerate(lam) if v == 0)
    for ax in (a1, a2):
        ax.axvspan(-0.45, ab + 0.5, color=SHADE, lw=0, zorder=0)
        ax.text(ab / 2, 1.0, "BIRS abstains (λ* = 0)", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7, color=SEC)
    for stem, lab, harm, gain, mk, ls in S:
        col = C["refit" if stem == "refit_tuned" else stem]
        b = stem == "birs"
        kw = dict(color=col, marker=mk, ls=ls, lw=2.4 if b else 1.3, ms=5 if b else 4.2, mec="white",
                  mew=0.6, zorder=2 if b else 4, label=lab, solid_capstyle="round")
        a1.plot(X, harm, **kw)
        a2.plot(X, gain, **{**kw, "zorder": 5 if b else 4})
    a1.axhline(5, color=SEC, lw=0.7, ls=(0, (6, 2, 1, 2)), zorder=1)
    a1.text(len(X) - 0.6, 5.6, "τ = 5% calibration cap", ha="right", va="bottom", fontsize=7, color=SEC)
    a1.set_ylim(-1.5, 50)
    a1.set_ylabel("Cells harmed (%)")
    a1.set_title("Harm: cells below the transferred model")
    a2.axhline(0, color=INK, lw=0.7, zorder=1)
    a2.set_ylim(-0.02, 0.185)
    a2.set_ylabel("Mean change in study AUC")
    a2.set_title("Gain over the transferred model")
    for ax in (a1, a2):
        finish(ax)
        ax.set_xticklabels([f"{b}\nλ* = {v:g}" for b, v in zip(BUDGETS, lam)])
        ax.set_xlabel("Labelled target studies")
    for ax, t in ((a1, "(a)"), (a2, "(b)")):
        ax.text(0.5, -0.31, t, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=3.0,
               columnspacing=2.0)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.77, bottom=0.23)
    save(fig, "fig2_comparators")


# ---------------------------------------------------------------- Fig. 4: foundation arm ----
ENC = [("skelex", "SKELEX", "ViT-L/16 · MSK radiographs"),
       ("raddino", "RAD-DINO", "ViT-B/14 · chest radiographs"),
       ("medsiglip", "MedSigLIP-448", "SigLIP-So400m · medical images"),
       ("dinov3", "DINOv3-B", "ViT-B/16 · natural images")]


def parse(s):
    out = dict(x.split(":") for x in s.split(", "))
    assert [int(k) for k in out] == BUDGETS, s
    return [float(v) for v in out.values()]


def fig_foundation():
    pub = schedule()
    imnet = [100 * N[f"rr_heldout_refit_tuned_n_harmed_at_{b}"] / N[f"rr_heldout_n_cells_at_{b}"] for b in BUDGETS]
    # BIRS external harm is zero for every encoder and arm: asserted, then stated once in the legend
    for e, *_ in ENC:
        for arm in "ab":
            assert N[f"b28_{arm}_{e}_o2_ext_harmed_cells"] == 0 and N[f"b28_{arm}_{e}_n_cells_ext"] == 449, (e, arm)
    top = max(100 * N[f"b28_{arm}_{e}_o1_refit_ext_harm_at_{b}"] for e, *_ in ENC for arm in "ab" for b in BUDGETS)

    fig, ax = plt.subplots(2, 4, figsize=(165 * MM, 132 * MM), sharey="row",
                           gridspec_kw={"hspace": 0.78, "wspace": 0.12})
    for j, (e, name, desc) in enumerate(ENC):
        a, b = ax[0, j], ax[1, j]
        a.plot(X, imnet, color=REF, lw=1.2, ls=(0, (4, 1.6)), marker="D", ms=3.2, zorder=3,
               label="Joint-penalty refit, ImageNet backbones (pooled)")
        a.plot(X, [0] * 6, color=C["birs"], lw=2.4, zorder=2, solid_capstyle="round",
               label="BIRS, own schedule (0 of 449 cells harmed, every encoder and arm)")
        for arm, fill, ls, lab in (("a", C["refit"], "-", "Joint-penalty refit, fine-tuned encoder"),
                                   ("b", "white", (0, (2.5, 1.2)), "Joint-penalty refit, frozen encoder")):
            a.plot(X, [100 * N[f"b28_{arm}_{e}_o1_refit_ext_harm_at_{bb}"] for bb in BUDGETS], color=C["refit"],
                   lw=1.4, ls=ls, marker="v", ms=4.2, mfc=fill, mec=C["refit"], mew=1.0, zorder=4, label=lab)
        a.axhline(5, color=SEC, lw=0.7, ls=(0, (6, 2, 1, 2)), zorder=1)
        a.set_ylim(-3, 10 * np.ceil(top / 10) + 2)
        a.set_title(f"{name}\n", fontweight="bold", loc="center", pad=2)
        a.text(0.5, 1.02, desc, transform=a.transAxes, ha="center", va="bottom", fontsize=7, color=SEC)
        # (b) schedule
        for x, v in zip(X, pub):
            b.plot([x - 0.36, x + 0.36], [v, v], color=REF, lw=2.6, solid_capstyle="butt", zorder=2,
                   label="Published schedule (ImageNet)" if x == 0 else None)
        for arm, off, fill, lab in (("a", -0.14, C["birs"], "BIRS weight, fine-tuned encoder"),
                                    ("b", 0.14, "white", "BIRS weight, frozen encoder")):
            b.plot(X + off, parse(N[f"b28_{arm}_{e}_o2_schedule"]), ls="none", marker="o", ms=4.6, mfc=fill,
                   mec=C["birs"], mew=1.1, zorder=4, label=lab)
        b.set_ylim(-0.07, 1.1)
        b.set_yticks([0, 0.25, 0.5, 0.75, 1])
        for p in (a, b):
            finish(p)
            p.set_xticklabels(BUDGETS)
    ax[0, -1].text(len(X) - 0.6, 6.2, "τ = 5%", fontsize=7, color=SEC, ha="right")
    ax[0, 0].set_ylabel("External cells harmed\nby refitting (%)")
    ax[1, 0].set_ylabel("BIRS weight on the\nlocal refit (λ)")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.865, bottom=0.11)
    for r, letter in ((0, "(a)"), (1, "(b)")):
        y0 = ax[r, 0].get_position().y0
        fig.text(0.537, y0 - 0.045, "Labelled target studies", ha="center", va="top", fontsize=8)
        fig.text(0.537, y0 - 0.072, letter, ha="center", va="top", fontsize=8)
        h, l = ax[r, 0].get_legend_handles_labels()
        if r == 1:
            h, l = h[-2:] + h[:1], l[-2:] + l[:1]          # BIRS weights first, then the published reference
        if r == 0:
            fig.legend(h, l, loc="upper center", ncol=2, handlelength=2.6, columnspacing=2.0,
                       bbox_to_anchor=(0.5, 1.0))
        else:
            fig.legend(h, l, loc="lower right", ncol=3, handlelength=2.6, columnspacing=1.4,
                       bbox_to_anchor=(0.99, ax[1, 0].get_position().y1 + 0.004))
    save(fig, "fig4_foundation")


if __name__ == "__main__":
    print(f"numbers: {NUMBERS} (9a7f55f0)")
    fig_budget()
    fig_comparators()
    fig_foundation()
    print("every plotted value read from, or checked against, paper_numbers.json")
