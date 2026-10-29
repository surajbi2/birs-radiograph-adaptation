"""Write the foundation-encoder tables of the paper from paper_numbers.json.

Tables (LaTeX, into output/tables/):
    tab_models.tex        source models: architecture, pretraining data, input
                          size, weights and wrist validation AUC (Table 3)
    tab_foundation.tex    refit harm, BIRS per encoder and the published schedule
                          (Table 7)
    tabS_zeroshot.tex     transferred-model study AUC per target (Table S8)
    tabS_comparators.tex  external harm and gain of every policy (Table S9)
    tabS_caps.tex         schedules under other caps and confidence-bounded caps
                          (Table S10)
    tabS_ci.tex           cluster-bootstrap intervals on mean external gain
                          (Table S11)

Values that are not stored keys (pooled counts and their exact one-sided
Clopper-Pearson bounds) are written, with how they were computed, to
output/derived_numbers.json.

Usage:
    python source_model_aucs.py     # first: writes output/extra_numbers.json
    python foundation_tables.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
NUMBERS = ROOT / "code" / "paper" / "paper_numbers.json"
assert hashlib.sha256(NUMBERS.read_bytes()).hexdigest().startswith("9a7f55f0"), "not the canonical numbers file"
N = {k: v["value"] for k, v in json.loads(NUMBERS.read_text(encoding="utf-8"))["numbers"].items()}
X = {k: v["value"] for k, v in json.loads((HERE / "output" / "extra_numbers.json").read_text(encoding="utf-8"))
     ["numbers"].items()}

BUD = [10, 25, 50, 100, 200, 500]
ENC = [("skelex", "SKELEX"), ("raddino", "RAD-DINO"), ("medsiglip", "MedSigLIP-448"), ("dinov3", "DINOv3-B")]
ARMS = [("a", "fine-tuned"), ("b", "frozen")]
IN = [("densenet121", "DenseNet-121"), ("convnext_tiny", "ConvNeXt-Tiny"), ("swin_tiny", "Swin-Tiny")]
CAL = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER"]
EXT = ["GRAZ_initial", "MFXR", "FRAC_XR_HAND"]
derived = {}


def d3(v):
    return f"{v:.3f}".replace("-", "$-$")


def pct(v, nd=1):
    return f"{100 * v:.{nd}f}"


def cp_upper0(n):
    """Exact one-sided 95% Clopper-Pearson upper bound for 0 events in n trials."""
    return 1 - 0.05 ** (1 / n)


def sched(s):
    v = dict(x.split(":") for x in s.split(", "))
    assert [int(k) for k in v] == BUD, s
    return [float(x) for x in v.values()]


def lam_cells(s):
    """Weights at 100, 200 and 500 studies: those below 100 are zero in every schedule here (checked)."""
    v = sched(s)
    assert v[:3] == [0, 0, 0], s
    return "/".join(f"{x:g}" for x in v[3:])


def setd(key, value, note):
    assert key not in derived
    derived[key] = {"value": value, "note": note}
    return value


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"  wrote tables/{name}")


# ------------------------------------------------------------------------------------- Table 3: models ----
rows = [
    ("DenseNet-121", "CNN, batch norm.", "ImageNet-1k", r"\texttt{densenet121.ra\_in1k}", 224, "densenet121"),
    ("ConvNeXt-Tiny", "CNN, layer norm.", "ImageNet-12k, then 1k", r"\texttt{convnext\_tiny.in12k\_ft\_in1k}", 224,
     "convnext_tiny"),
    ("Swin-Tiny", "Transformer, layer norm.", "ImageNet-1k", r"\texttt{swin\_tiny\_patch4\_window7\_224.ms\_in1k}",
     224, "swin_tiny"),
    ("SKELEX", "ViT-L/16, MAE", "Musculoskeletal radiographs", r"\texttt{skhoha/SKELEX}", 224, "skelex"),
    ("RAD-DINO", "ViT-B/14, DINOv2", "Chest radiographs", r"\texttt{microsoft/rad-dino}", 518, "raddino"),
    ("MedSigLIP-448", "SigLIP-So400m", "Medical images", r"\texttt{google/medsiglip-448}",
     448, "medsiglip"),
    ("DINOv3-B", "ViT-B/16, DINOv3", "Natural images (LVD-1689M)",
     r"\texttt{facebook/dinov3-vitb16-pretrain-lvd1689m}", 224, "dinov3"),
]
t = [r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llllc}", r"\toprule",
     r"Model & Architecture & Pretraining data & Input (pixels) & Wrist AUC \\", r"\midrule",
     r"\multicolumn{5}{l}{\emph{ImageNet backbones (timm)}} \\"]
for i, (m, a, p, w, s, k) in enumerate(rows):
    if i == 3:
        t += [r"\midrule", r"\multicolumn{5}{l}{\emph{Foundation encoders (Hugging Face)}} \\"]
    t.append(f"{m} & {a} & {p} & {s} & {X[f'cbm_src_auc_{k}_mean']:.3f} ({X[f'cbm_src_auc_{k}_sd']:.3f}) \\\\")
t += [r"\bottomrule", r"\end{tabular*}", "", r"\smallskip", r"\raggedright\footnotesize",
      "Weights: " + "; ".join(f"{m}, {w}" for m, a, p, w, s, k in rows) + "."]
write("tab_models.tex", "\n".join(t) + "\n")

# ----------------------------------------------------------------------------- Table 7: foundation arm ----
le50 = sum(N[f"rr_heldout_refit_tuned_n_harmed_at_{b}"] for b in BUD[:3])
n50 = sum(N[f"rr_heldout_n_cells_at_{b}"] for b in BUD[:3])
assert (le50, n50) == (214, 670)
setd("cbm_in_refit_le50_rate", le50 / n50, "ImageNet pooled joint-penalty refit external harm at <=50 studies, "
     "sum of rr_heldout_refit_tuned_n_harmed_at_{10,25,50} / sum of rr_heldout_n_cells_at_{10,25,50}")
assert N["lambda_schedule"] == "n=10:0, n=25:0, n=50:0, n=100:0.1, n=200:0.75, n=500:1"
# Two-line headers, compact weights and a short first column keep all eight columns inside the text width.
t = [r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lccccccc}", r"\toprule",
     r"& Refit & \multicolumn{4}{c}{\method{} calibrated per encoder} & \multicolumn{2}{c}{Published schedule} \\",
     r"\cmidrule{3-6}\cmidrule{7-8}",
     r"Source model & \makecell{harm\\$\le 50$ (\%)} & \makecell{$\lambda^{\star}$ at\\100/200/500} "
     r"& \makecell{Harmed\\(adapted)} & Gain & \makecell{LOO\\harm (\%)} "
     r"& \makecell{External\\harmed} & \makecell{Anatomy\\harm (\%)} \\", r"\midrule",
     f"ImageNet, pooled & {pct(le50 / n50)} & 0.1/0.75/1 & 0 (675) & "
     f"{d3(N['lamsched_test_gain_schedule'])} & {pct(N['rr_loao_pooled_schedule_harm_rate'])} & -- & -- \\\\"]
assert N["heldout_schedule_harm_ub95_cells_adapted"] < 0.0045
for arm, lab in ARMS:
    t += [r"\midrule", f"\\multicolumn{{8}}{{l}}{{\\emph{{Foundation encoders, {lab}}}}} \\\\"]
    for e, name in ENC:
        g = lambda k: N[f"b28_{arm}_{e}_{k}"]
        assert g("o2_ext_harmed_cells") == 0 and g("o2_ext_harmed_cells_adapted") == 0
        t.append(f"{name} & {pct(g('o1_refit_ext_le50_harm_rate'))} & {lam_cells(g('o2_schedule'))} & "
                 f"0 ({int(g('o2_ext_n_cells_adapted'))}) & {d3(g('o2_ext_mean_gain'))} & {pct(g('o2_loo_harm_rate'))} "
                 f"& {int(g('o3_published_ext_harmed_cells'))} & {pct(g('o3_published_cal_harm_rate'))} \\\\")
t += [r"\bottomrule", r"\end{tabular*}"]
write("tab_foundation.tex", "\n".join(t) + "\n")

# pooled counts and bounds for the text
for arm, lab in ARMS:
    nad = sum(N[f"b28_{arm}_{e}_o2_ext_n_cells_adapted"] for e, _ in ENC)
    nce = sum(N[f"b28_{arm}_{e}_o2_ext_n_cells"] for e, _ in ENC)
    npa = sum(N[f"b28_{arm}_{e}_o2_ext_n_pairs"] for e, _ in ENC)
    nmo = sum(N[f"b28_{arm}_{e}_o2_ext_n_models"] for e, _ in ENC)
    assert sum(N[f"b28_{arm}_{e}_o2_ext_harmed_cells"] for e, _ in ENC) == 0
    setd(f"cbm_{arm}_pooled_ext_cells", nce, f"arm {arm}: external cells over the four encoders")
    setd(f"cbm_{arm}_pooled_adapted_cells", nad, f"arm {arm}: adapted external cells over the four encoders")
    setd(f"cbm_{arm}_pooled_pairs", npa, f"arm {arm}: model-dataset pairs over the four encoders")
    setd(f"cbm_{arm}_pooled_models", nmo, f"arm {arm}: source models over the four encoders")
    setd(f"cbm_{arm}_pooled_ub95_adapted", cp_upper0(nad), f"arm {arm}: exact one-sided 95% bound, 0 of {nad} adapted cells")
    setd(f"cbm_{arm}_pooled_ub95_pairs", cp_upper0(npa), f"arm {arm}: exact one-sided 95% bound, 0 of {npa} pairs")
    setd(f"cbm_{arm}_pooled_ub95_models", cp_upper0(nmo), f"arm {arm}: exact one-sided 95% bound, 0 of {nmo} models")
    # check the bound formula against a stored per-encoder key
    assert abs(cp_upper0(N[f"b28_{arm}_skelex_o2_ext_n_pairs"]) - N[f"b28_{arm}_skelex_o2_ext_ub95_pairs"]) < 1e-9

# ------------------------------------------------------------------------------ Table S8: zero-shot AUC ----
hdr = ["Elbow", "Finger", "Forearm", "Hand", "Humerus", "Shoulder", "GRAZ", "MFXR", "FracAtlas"]
t = [r"\begin{tabular}{l" + "c" * 11 + "}", r"\toprule",
     r"& \multicolumn{6}{c}{MURA anatomies} & \multicolumn{3}{c}{External} & \multicolumn{2}{c}{Mean} \\",
     r"\cmidrule(lr){2-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}",
     "Source model & " + " & ".join(hdr) + r" & Anat. & Ext. \\", r"\midrule"]


def zrow(name, pre):
    a = [N[f"{pre}_zs_auc_{x}"] for x in CAL + EXT]
    setd(f"cbm_zs_mean_anat_{pre}", sum(a[:6]) / 6, f"{name}: mean transferred study AUC over the six anatomies")
    setd(f"cbm_zs_mean_ext_{pre}", sum(a[6:]) / 3, f"{name}: mean transferred study AUC over the three external sets")
    return (f"{name} & " + " & ".join(f"{v:.3f}" for v in a) + f" & {sum(a[:6]) / 6:.3f} & {sum(a[6:]) / 3:.3f} \\\\")


for k, name in IN:
    t.append(zrow(name, f"b28_in_{k}"))
for arm, lab in ARMS:
    t += [r"\midrule", f"\\multicolumn{{12}}{{l}}{{\\emph{{Foundation encoders, {lab}}}}} \\\\"]
    for e, name in ENC:
        t.append(zrow(name, f"b28_{arm}_{e}"))
t += [r"\bottomrule", r"\end{tabular}"]
write("tabS_zeroshot.tex", "\n".join(t) + "\n")

# --------------------------------------------------------------------------- Table S9: comparators ----
POL = [("o2_ext", "\\method{} (own schedule)"), ("o3_published_ext", "Published schedule"),
       ("abstain50_cv_lambda_ext", "Abstain, then cross-validated $\\lambda$"),
       ("lam0.5_ext", "Fixed $\\lambda = 0.5$"), ("cv_lambda_ext", "Cross-validated $\\lambda$"),
       ("l2sp_cv_ext", "L2-SP, cross-validated strength"), ("refit_tuned_ext", "Joint-penalty refit"),
       ("refit_own_ext", "Own-penalty refit")]


def harmed(arm, e, p):
    k = f"b28_{arm}_{e}_{p}_harmed_cells" if p in ("o2_ext", "o3_published_ext") else f"b28_{arm}_{e}_{p}_n_harmed"
    return int(N[k])


t = [r"\begin{tabular}{l" + "c" * 4 + "}", r"\toprule", "Policy & " + " & ".join(n for _, n in ENC) + r" \\"]
for arm, lab in ARMS:
    t += [r"\midrule", f"\\multicolumn{{5}}{{l}}{{\\emph{{{lab.capitalize()} encoders}}}} \\\\"]
    for p, name in POL:
        t.append(name + " & " + " & ".join(
            f"{harmed(arm, e, p)} / {d3(N[f'b28_{arm}_{e}_{p}_mean_gain'])}" for e, _ in ENC) + r" \\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tabS_comparators.tex", "\n".join(t) + "\n")

# ----------------------------------------------------------------------------------- Table S10: caps ----
CAPS = [("tau0", "$\\tau = 0$"), ("o2", "$\\tau = 0.05$ (primary)"), ("tau0p1", "$\\tau = 0.10$"),
        ("tau0p2", "$\\tau = 0.20$"), ("cap_cp", "Exact upper bound $\\le 0.05$"),
        ("cap_cluster", "Cluster-bootstrap bound $\\le 0.05$")]


def cap_cells(arm, e, c):
    s = sched(N[f"b28_{arm}_{e}_{c}_schedule"])
    ext = N[f"b28_{arm}_{e}_o2_ext_harmed_cells"] if c == "o2" else N[f"b28_{arm}_{e}_{c}_ext_n_harmed"]
    loo = N[f"b28_{arm}_{e}_{c}_loo_n_harmed"]
    return "/".join(f"{x:g}" for x in s), int(ext), int(loo)


t = [r"\begin{tabular}{llccc}", r"\toprule",
     r"Source model & Cap & $\lambda^{\star}$ at 10/25/50/100/200/500 & External harmed & LOO harmed \\"]
for arm, lab in ARMS:
    for e, name in ENC:
        t.append(r"\midrule")
        for i, (c, cn) in enumerate(CAPS):
            s, ex, lo = cap_cells(arm, e, c)
            t.append((f"{name}, {lab}" if i == 0 else "") + f" & {cn} & {s} & {ex} & {lo} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tabS_caps.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------------------------- Table S11: CIs ----
CI = [("birs", "\\method{} (own schedule)"), ("published", "Published schedule"),
      ("abstain50_cv_lambda", "Abstain, then cross-validated $\\lambda$"), ("cv_lambda", "Cross-validated $\\lambda$"),
      ("refit_tuned", "Joint-penalty refit"), ("birs_minus_cv", "\\method{} minus cross-validated $\\lambda$")]
ci = lambda lo, hi: f"{d3(lo)} to {d3(hi)}"
t = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"& \multicolumn{2}{c}{Fine-tuned (20 models, 60 pairs)} & \multicolumn{2}{c}{Frozen (20 models, 60 pairs)} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Policy & Models & Pairs & Models & Pairs \\", r"\midrule"]
for p, name in CI:
    cells = []
    for arm, _ in ARMS:
        for u in ("models", "pairs"):
            cells.append(ci(N[f"b28_{arm}_pooled_{u}_ci_{p}_lo"], N[f"b28_{arm}_pooled_{u}_ci_{p}_hi"]))
    t.append(f"{name} & " + " & ".join(cells) + r" \\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tabS_ci.tex", "\n".join(t) + "\n")

setd("cbm_ab_pooled_ext_cells", derived["cbm_a_pooled_ext_cells"]["value"] + derived["cbm_b_pooled_ext_cells"]["value"],
     "external foundation-encoder cells, both arms")
(HERE / "output" / "derived_numbers.json").write_text(json.dumps({"script": Path(__file__).name, "numbers": derived},
                                                              indent=1), encoding="utf-8")
for k, v in derived.items():
    print(f"  {k:32s} {v['value']}")
