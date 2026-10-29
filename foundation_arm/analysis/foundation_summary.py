"""Summarise the foundation-encoder arm: outcomes O1-O4 of the analysis plan.

The plan is Supplementary File 2 of the paper (outcomes in its sections 4-6).
Reads the four cell files written by budget_cells.py from ../cells/ (their SHA-256
are checked against FA_SHA) and applies the schedule, Clopper-Pearson bound and
leave-one-anatomy-out procedure used for the ImageNet backbones. Two gates run
first: the ImageNet-arm numbers are reproduced from the ImageNet cells, then the
structure and pairing of the foundation-arm cells are checked.

Inputs:
    ../cells/cells_arm_{a,b}[_dinov3].csv
    RES: results/transfer/ and paper_2026-10-04b/paper_numbers.json of the
         ImageNet arm (set at the top; see README, Paths)

Output:
    foundation_summary.json   every value with a key and a note; read by
                             code/make_paper_outputs.py

Usage:
    python foundation_summary.py
"""
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
FA = HERE.parent / "cells"
FA3 = HERE.parent / "cells"   # DINOv3 run, same budget_cells.py (d4ba7cbd)
RES = Path("/path/to/imagenet_arm_results")
NJ = json.load(open(RES / "paper_2026-10-04b/paper_numbers.json"))["numbers"]
key = lambda k: float(NJ[k]["value"])
FA_SHA = {"cells_arm_a.csv": "7129ef856d6da2fbff1384191b6817b133a8c30003418707eaff18f7f906a0ed",
          "cells_arm_b.csv": "4515f8825e6748a8af4b33b9d660c16b331198321d5ac0511ad89ba82571bec4",
          "cells_arm_a_dinov3.csv": "bc706ab346f3d96e860b677c852ee2f53543caf5669799d40de9845eddf2da7b",
          "cells_arm_b_dinov3.csv": "98f4b0361815f3a47524afc927c4925a45ad57110ef5db27349954a8af29588d"}
BUDGETS = [10, 25, 50, 100, 200, 500]
GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
EXT = ["GRAZ_initial", "MFXR", "FRAC_XR_HAND"]
CAL = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER"]
TAU = 0.05
PUBLISHED = dict(zip(BUDGETS, [0, 0, 0, 0.1, 0.75, 1]))
ENCODERS = ["skelex", "rad-dino", "medsiglip", "dinov3"]
IMAGENET = ["densenet121", "convnext_tiny", "swin_tiny_patch4_window7_224"]


def read(p):
    return list(csv.DictReader(io.StringIO("".join(l for l in open(p, encoding="utf-8") if not l.startswith("#")))))


SETTING = {"refit_C1": "refit_c1", "refit_nestedC": "refit_tuned", "shrink_nested": "cv_lambda",
           "refit_ownC": "refit_own", "l2sp_cv": "l2sp_cv", "l2sp_C1": "l2sp_c1"}
SETTING.update({f"shrink_lam{g:g}": f"lam{g:g}" for g in GRID})
NPOL = len(SETTING)
cells = defaultdict(dict)   # (target, model, seed, budget, resample) -> policy -> delta; model = "<arm>:<encoder>"
auc = defaultdict(dict)     # same key -> policy -> study AUC
zs = {}                     # (target, model, seed) -> transferred-model study AUC


def add(model, r, pol):
    c = (r["target"], model, r["seed"], int(r["budget"]), r["resample"])
    d = float(r["delta_vs_zeroshot"])
    assert "harm" not in r or (r["harm"] == "1") == (d < 0), r   # b11 rows carry no harm column
    cells[c][pol] = d
    auc[c][pol] = float(r["auc_study"])
    z = float(r["zeroshot_auc"])   # the transferred model's AUC on this unit, carried on every row
    assert abs(zs.setdefault((c[0], model, c[2]), z) - z) < 1e-12, c


# ---------- paper cells (ImageNet arm) ----------
for f in ("readout_shrinkage_XR_WRIST.csv", "fracatlas_budget_cells_XR_WRIST_XR_HAND.csv"):
    for r in read(RES / "results/transfer" / f):
        if r["setting"] in SETTING:
            add("paper:" + r["backbone"], r, SETTING[r["setting"]])
for r in read(RES / "results/transfer/l2sp_comparator_XR_WRIST.csv"):
    c = (r["target"], "paper:" + r["backbone"], r["seed"], int(r["budget"]), r["resample"])
    if c in cells:
        add("paper:" + r["backbone"], r, "l2sp_cv" if r["setting"] == "l2sp_cv" else "l2sp_c1")
own_seen = {}
for r in read(RES / "results/transfer/readout_penalty_sweep_XR_WRIST.csv"):   # own-penalty refit; exact duplicates collapse
    if r["setting"] == "nestedCV":
        c = (r["target"], "paper:" + r["backbone"], r["seed"], int(r["budget"]), r["resample"])
        if c not in cells:   # b11 also swept resnet50, which is not in the paper
            continue
        assert own_seen.setdefault(c, r["delta_vs_zeroshot"]) == r["delta_vs_zeroshot"], c
        add("paper:" + r["backbone"], r, "refit_own")
paper = [c for c in cells if c[1].startswith("paper:")]
assert all(len(cells[c]) - ("refit_own" in cells[c]) == NPOL - 1 for c in paper)
# ---------- FA cells ----------
for d, f, arm in ((FA, "cells_arm_a.csv", "A"), (FA, "cells_arm_b.csv", "B"),
                  (FA3, "cells_arm_a_dinov3.csv", "A"), (FA3, "cells_arm_b_dinov3.csv", "B")):
    assert hashlib.sha256((d / f).read_bytes()).hexdigest() == FA_SHA[f], f
    for r in read(d / f):
        assert r["source"] == "XR_WRIST" and r["rng_bb"] == "densenet121", r
        m = f"{arm}:{r['backbone']}"
        if r["setting"] == "zeroshot":
            assert abs(zs.setdefault((r["target"], m, r["seed"]), float(r["auc_study"])) - float(r["auc_study"])) < 1e-12
        else:
            add(m, r, SETTING[r["setting"]])
MODELS = ["paper:" + b for b in IMAGENET] + [f"{a}:{e}" for a in "AB" for e in ENCODERS]
of = lambda m: [c for c in cells if c[1] == m]

# ---------- gate 1: the paper's numbers reproduce from the paper's cells (same procedure as the ImageNet-arm analyses) ----------
pcal = [c for c in paper if c[0] in CAL]
pext = [c for c in paper if c[0] in EXT]
assert (len(pext), len(pcal)) == (1345, 2693)


def fit_schedule(cs, tau=TAU, mode="point", monotone=True):
    """lambda*(n): max mean gain s.t. harm bound <= tau; ties -> smaller lambda; lambda=0 if no nonzero fits."""
    out, raw, ties = {}, {}, 0
    for n in BUDGETS:
        sub = [c for c in cs if c[3] == n]
        best, cand = 0.0, []
        for g in GRID[1:]:
            d = [cells[c][f"lam{g:g}"] for c in sub]
            k, N = sum(x < 0 for x in d), len(d)
            h = k / N if mode == "point" else cp_upper(k, N) if mode == "cp" else cluster_bound(sub, g)
            if h <= tau:
                cand.append((sum(d) / N, g))
        if cand:
            top = max(m for m, _ in cand)
            tied = [g for m, g in cand if abs(m - top) < 1e-12]
            ties += len(tied) > 1
            if top > 0:
                best = min(tied)
        raw[n] = best
    run = 0.0
    for n in BUDGETS:
        run = max(run, raw[n]) if monotone else raw[n]
        out[n] = run
    return out, raw, ties


def cp_upper(k, n, alpha=0.05):
    """One-sided exact Clopper-Pearson upper bound by bisection on the binomial CDF."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > alpha else (lo, mid)
    return hi


_cache = {}


def cluster_bound(sub, g, reps=2000):
    """95th pct harm rate of weight g over a bootstrap of (target, model, seed) clusters within sub (one budget)."""
    ck = (frozenset(sub), g)
    if ck in _cache:
        return _cache[ck]
    groups = defaultdict(list)
    for c in sub:
        groups[(c[0], c[1], c[2])].append(1 if cells[c][f"lam{g:g}"] < 0 else 0)
    stats = [(sum(v), len(v)) for v in groups.values()]
    r = random.Random(7)
    hs = []
    for _ in range(reps):
        k = n = 0
        for _ in stats:
            a, b = stats[r.randrange(len(stats))]
            k, n = k + a, n + b
        hs.append(k / n)
    _cache[ck] = sorted(hs)[int(0.95 * reps) - 1]
    return _cache[ck]


def apply(schedule, cs):
    d = [cells[c][f"lam{schedule[c[3]]:g}"] for c in cs]
    return sum(x < 0 for x in d), sum(d) / len(d)


def pol_d(pol, c):
    if pol == "abstain50_cv_lambda":
        return 0.0 if c[3] <= 50 else cells[c]["cv_lambda"]
    return cells[c][pol]


def policy_stats(pol, cs):
    d = [pol_d(pol, c) for c in cs]
    return sum(x < 0 for x in d), sum(d) / len(d)


def loo(cal, fit_kwargs):
    k = s = n = 0
    thr = []
    for a in CAL:
        sch, _, _ = fit_schedule([c for c in cal if c[0] != a], **fit_kwargs)
        thr.append(abstain_upto(sch))
        cs = [c for c in cal if c[0] == a]
        kk, mg = apply(sch, cs)
        k, s, n = k + kk, s + mg * len(cs), n + len(cs)
    return k, s / n, min(thr), max(thr)


def abstain_upto(s):
    """Largest budget with lambda = 0 before the first nonzero weight (0 if adaptation starts at 10)."""
    z = [n for n in BUDGETS if s[n] == 0]
    return max(z) if z and all(s[n] == 0 for n in BUDGETS if n <= max(z)) else 0


sched, raw, ties = fit_schedule(pcal)
assert sched == PUBLISHED, sched
kh, gain = apply(sched, pext)
assert kh == 0 and round(gain, 4) == round(key("lamsched_test_gain_schedule"), 4)
for pol, kk in (("lam0.5", "rr_heldout_lam0p5_n_harmed"), ("cv_lambda", "rr_heldout_cv_lambda_n_harmed"),
                ("refit_tuned", "rr_heldout_refit_tuned_n_harmed"), ("l2sp_cv", "rr_heldout_l2sp_cv_n_harmed")):
    assert policy_stats(pol, pext)[0] == key(kk), pol
assert policy_stats("refit_c1", pext)[0] == key("lamsched_test_harm_k_refit_c1")
assert round(policy_stats("abstain50_cv_lambda", pext)[1], 4) == round(key("rr_heldout_abstain50_cv_lambda_mean_gain"), 4)
assert policy_stats("abstain50_cv_lambda", pext)[0] == key("rr_heldout_abstain50_cv_lambda_n_harmed")
kl, gl, _, _ = loo(pcal, {})
assert kl == key("rr_loao_pooled_schedule_n_harmed") and round(gl, 4) == round(key("rr_loao_pooled_schedule_mean_gain"), 4)
own_cal = [c for c in pcal if "refit_own" in cells[c]]
own_graz = [c for c in pext if "refit_own" in cells[c]]
assert (len(own_cal), len(own_graz)) == (2693, 450) and all(c[0] == "GRAZ_initial" for c in own_graz)
assert sum(cells[c]["refit_own"] < 0 for c in own_cal) == 585 and sum(cells[c]["refit_own"] < 0 for c in own_graz) == 87
print("gate 1 passed: paper schedule, external harm/gain, comparators, abstain-50, LOO and own-penalty counts reproduce")

# ---------- gate 2: FA cell structure and pairing with the DenseNet-121 cells ----------
dn_keys = {(c[0], c[2], c[3], c[4]) for c in paper if c[1] == "paper:densenet121"}
for m in MODELS[3:]:
    cs = of(m)
    assert all(len(cells[c]) == NPOL for c in cs), m
    assert {(c[0], c[2], c[3], c[4]) for c in cs} == dn_keys, m
    assert all(cells[c]["lam0"] == 0 for c in cs), m
    assert all(cells[c]["refit_tuned"] == cells[c]["lam1"] for c in cs), m   # joint-penalty refit = lambda 1 at joint C
    for c in cs:
        assert abs(auc[c]["lam0"] - zs[(c[0], m, c[2])]) < 1e-12
assert len(dn_keys) == 1346
print("gate 2 passed: every FA (arm, encoder) has the DenseNet-121 key set (1,346 cells), lambda 0 = transferred model, "
      "joint refit = lambda 1")

# ---------- outcomes ----------
OUT = {}
fmt = lambda s: ", ".join(f"{n}:{s[n]:g}" for n in BUDGETS)
tag = lambda m: m.replace("swin_tiny_patch4_window7_224", "swin_tiny").replace("paper:", "in_").replace("A:", "a_").replace("B:", "b_").replace("-", "")


def put(k, v, note):
    OUT["b28_" + k] = {"value": v, "note": note}


def levels(cs, d_of):
    """Harmed units at the cell, model-dataset pair (target x seed) and source-model (seed) levels, with exact 95% upper bounds."""
    out = {}
    for lvl, f in (("cells", lambda c: c), ("pairs", lambda c: (c[0], c[2])), ("models", lambda c: c[2])):
        units = defaultdict(bool)
        for c in cs:
            units[f(c)] |= d_of(c) < 0
        k, n = sum(units.values()), len(units)
        out[lvl] = (k, n, cp_upper(k, n))
    return out


POLS = ["lam0.5", "cv_lambda", "refit_tuned", "refit_own", "l2sp_cv", "l2sp_c1", "refit_c1", "abstain50_cv_lambda"]
for m in MODELS:
    t = tag(m)
    cs = of(m)
    cal = [c for c in cs if c[0] in CAL]
    ext = [c for c in cs if c[0] in EXT]
    put(f"{t}_n_cells_cal", len(cal), f"{m}: calibration cells")
    put(f"{t}_n_cells_ext", len(ext), f"{m}: external cells")
    # O4: transferred-model AUC per target (mean over seeds) and refit AUC at 500 (mean over seeds and resamples)
    for tg in CAL + EXT:
        v = [zs[(tg, m, s)] for s in sorted({c[2] for c in cs})]
        put(f"{t}_zs_auc_{tg}", sum(v) / len(v), f"{m}: transferred-model study AUC on {tg}, mean of {len(v)} seeds")
        r5 = [auc[c]["refit_tuned"] for c in cs if c[0] == tg and c[3] == 500]
        if r5:
            put(f"{t}_refit500_auc_{tg}", sum(r5) / len(r5), f"{m}: joint-penalty refit study AUC at 500 studies on {tg}")
    # O1: harm of the joint-penalty refit
    for lab, sub in (("ext", ext), ("cal", cal)):
        for rng_lab, f in (("le50", lambda c: c[3] <= 50), ("all", lambda c: True)):
            ss = [c for c in sub if f(c)]
            k = sum(cells[c]["refit_tuned"] < 0 for c in ss)
            put(f"{t}_o1_refit_{lab}_{rng_lab}_n_harmed", k, f"{m}: joint-penalty refit harmed {lab} cells, budgets {rng_lab}")
            put(f"{t}_o1_refit_{lab}_{rng_lab}_harm_rate", k / len(ss), f"{m}: joint-penalty refit harm rate, {lab}, {rng_lab}")
        for n in BUDGETS:
            ss = [c for c in sub if c[3] == n]
            put(f"{t}_o1_refit_{lab}_harm_at_{n}", sum(cells[c]["refit_tuned"] < 0 for c in ss) / len(ss),
                f"{m}: joint-penalty refit harm rate on {lab} cells at {n} studies")
    if m.startswith("paper:"):
        continue
    # O2: per-encoder BIRS
    s, rw, ti = fit_schedule(cal)
    put(f"{t}_o2_schedule", fmt(s), f"{m}: BIRS schedule calibrated on this encoder's six-anatomy cells (tau 0.05)")
    put(f"{t}_o2_schedule_raw", fmt(rw), f"{m}: schedule before the running maximum")
    put(f"{t}_o2_ties", ti, f"{m}: budgets with tied feasible weights")
    put(f"{t}_o2_abstain_upto", abstain_upto(s), f"{m}: largest budget at which BIRS prescribes no adaptation")
    lv = levels(ext, lambda c: cells[c][f"lam{s[c[3]]:g}"])
    for lvl, (k, n, ub) in lv.items():
        put(f"{t}_o2_ext_harmed_{lvl}", k, f"{m}: BIRS external harmed {lvl} (of {n})")
        put(f"{t}_o2_ext_n_{lvl}", n, f"{m}: external {lvl}")
        put(f"{t}_o2_ext_ub95_{lvl}", ub, f"{m}: one-sided exact 95% upper bound on the harmed share of {lvl}")
    k, g = apply(s, ext)
    put(f"{t}_o2_ext_mean_gain", g, f"{m}: BIRS external mean AUC gain")
    for n in BUDGETS:
        put(f"{t}_o2_ext_gain_at_{n}", apply(s, [c for c in ext if c[3] == n])[1], f"{m}: BIRS external mean gain at {n}")
    for ds in EXT:
        k, g = apply(s, [c for c in ext if c[0] == ds])
        put(f"{t}_o2_ext_{ds}_n_harmed", k, f"{m}: BIRS harmed cells on {ds}")
        put(f"{t}_o2_ext_{ds}_mean_gain", g, f"{m}: BIRS mean gain on {ds}")
    kl, gl, tmin, tmax = loo(cal, {})
    put(f"{t}_o2_loo_n_harmed", kl, f"{m}: leave-one-anatomy-out harmed cells (of {len(cal)})")
    put(f"{t}_o2_loo_harm_rate", kl / len(cal), f"{m}: leave-one-anatomy-out harm rate")
    put(f"{t}_o2_loo_mean_gain", gl, f"{m}: leave-one-anatomy-out mean gain")
    put(f"{t}_o2_loo_abstain_upto_min", tmin, f"{m}: smallest abstain threshold over the six LOO refits")
    put(f"{t}_o2_loo_abstain_upto_max", tmax, f"{m}: largest abstain threshold over the six LOO refits")
    # O3: the published ImageNet schedule applied unchanged
    for lab, sub in (("ext", ext), ("cal", cal)):
        k, g = apply(PUBLISHED, sub)
        put(f"{t}_o3_published_{lab}_n_harmed", k, f"{m}: published schedule 0/0/0/0.1/0.75/1, harmed {lab} cells")
        put(f"{t}_o3_published_{lab}_harm_rate", k / len(sub), f"{m}: published schedule, {lab} harm rate")
        put(f"{t}_o3_published_{lab}_mean_gain", g, f"{m}: published schedule, {lab} mean gain")
    lv = levels(ext, lambda c: cells[c][f"lam{PUBLISHED[c[3]]:g}"])
    for lvl, (k, n, ub) in lv.items():
        put(f"{t}_o3_published_ext_harmed_{lvl}", k, f"{m}: published schedule, external harmed {lvl} (of {n})")
    # comparators
    for p in POLS:
        for lab, sub in (("ext", ext), ("cal", cal)):
            k, g = policy_stats(p, sub)
            put(f"{t}_{p}_{lab}_n_harmed", k, f"{m}: {p}, harmed {lab} cells (of {len(sub)})")
            put(f"{t}_{p}_{lab}_mean_gain", g, f"{m}: {p}, {lab} mean gain")
    # confidence-bounded caps
    for mode in ("cp", "cluster"):
        sm, _, _ = fit_schedule(cal, mode=mode)
        k, g = apply(sm, ext)
        kl, gl, _, _ = loo(cal, {"mode": mode})
        put(f"{t}_cap_{mode}_schedule", fmt(sm), f"{m}: schedule with the {mode} 95% upper bound on calibration harm <= 0.05")
        put(f"{t}_cap_{mode}_ext_n_harmed", k, f"{m}: {mode} cap, external harmed cells")
        put(f"{t}_cap_{mode}_ext_mean_gain", g, f"{m}: {mode} cap, external mean gain")
        put(f"{t}_cap_{mode}_loo_n_harmed", kl, f"{m}: {mode} cap, LOO harmed cells")
        put(f"{t}_cap_{mode}_loo_mean_gain", gl, f"{m}: {mode} cap, LOO mean gain")
    # tau sensitivity (secondary)
    for tau in (0.0, 0.10, 0.20):
        st, _, _ = fit_schedule(cal, tau=tau)
        k, g = apply(st, ext)
        kl, gl, _, _ = loo(cal, {"tau": tau})
        tt = f"{tau:g}".replace(".", "p")
        put(f"{t}_tau{tt}_schedule", fmt(st), f"{m}: schedule at tau={tau:g}")
        put(f"{t}_tau{tt}_ext_n_harmed", k, f"{m}: tau={tau:g}, external harmed cells")
        put(f"{t}_tau{tt}_ext_mean_gain", g, f"{m}: tau={tau:g}, external mean gain")
        put(f"{t}_tau{tt}_loo_n_harmed", kl, f"{m}: tau={tau:g}, LOO harmed cells")
        put(f"{t}_tau{tt}_loo_mean_gain", gl, f"{m}: tau={tau:g}, LOO mean gain")

# ---------- adapted cells and cluster-bootstrap CIs of the external gain (paper's A1, same seed and percentiles) ----------
own_sched = {m: fit_schedule([c for c in of(m) if c[0] in CAL])[0] for m in MODELS[3:]}
own_sched.update({m: PUBLISHED for m in MODELS[:3]})   # the paper's ImageNet cells use the published (pooled) schedule
for m in MODELS:
    t = tag(m)
    s = own_sched[m]
    ad = [c for c in of(m) if c[0] in EXT and s[c[3]] > 0]
    k = sum(cells[c][f"lam{s[c[3]]:g}"] < 0 for c in ad)
    put(f"{t}_o2_ext_n_cells_adapted", len(ad), f"{m}: external cells where the schedule adapts (lambda > 0)")
    put(f"{t}_o2_ext_harmed_cells_adapted", k, f"{m}: harmed among adapted external cells")
    put(f"{t}_o2_ext_ub95_cells_adapted", cp_upper(k, len(ad)), f"{m}: exact one-sided 95% upper bound, adapted cells")
assert OUT["b28_in_densenet121_o2_ext_n_cells_adapted"]["value"] + OUT["b28_in_convnext_tiny_o2_ext_n_cells_adapted"]["value"] \
    + OUT["b28_in_swin_tiny_o2_ext_n_cells_adapted"]["value"] == key("heldout_schedule_n_cells_adapted")

CI_POLS = {"birs": lambda c: cells[c][f"lam{own_sched[c[1]][c[3]]:g}"],
           "published": lambda c: cells[c][f"lam{PUBLISHED[c[3]]:g}"],
           "cv_lambda": lambda c: cells[c]["cv_lambda"], "refit_tuned": lambda c: cells[c]["refit_tuned"],
           "abstain50_cv_lambda": lambda c: pol_d("abstain50_cv_lambda", c)}


def boot_ci(ext, unit_of, label, note):
    groups = defaultdict(list)
    for c in ext:
        groups[unit_of(c)].append(c)
    gk = list(groups)
    rng = random.Random(20261006)
    reps = {p: [] for p in CI_POLS}
    diff = []
    for _ in range(5000):
        cs = [c for g in (groups[gk[rng.randrange(len(gk))]] for _ in gk) for c in g]
        mm = {p: sum(f(c) for c in cs) / len(cs) for p, f in CI_POLS.items()}
        for p in CI_POLS:
            reps[p].append(mm[p])
        diff.append(mm["birs"] - mm["cv_lambda"])
    for p in CI_POLS:
        v = sorted(reps[p])
        put(f"{label}_ci_{p}", [v[124], v[4874]], f"{note}: 95% cluster-bootstrap CI of mean external gain, {len(gk)} clusters, 5000 reps")
    v = sorted(diff)
    put(f"{label}_ci_birs_minus_cv", [v[124], v[4874]], f"{note}: 95% CI of (BIRS - CV lambda) mean external gain")


for arm in "AB":
    fam = [f"{arm}:{e}" for e in ENCODERS]
    ext = [c for c in cells if c[1] in fam and c[0] in EXT]
    boot_ci(ext, lambda c: (c[1], c[2]), f"{arm.lower()}_pooled_models", f"arm {arm}, {len(ENCODERS)} encoders pooled, source models")
    boot_ci(ext, lambda c: (c[0], c[1], c[2]), f"{arm.lower()}_pooled_pairs", f"arm {arm}, {len(ENCODERS)} encoders pooled, model-dataset pairs")
    for m in fam:
        boot_ci([c for c in ext if c[1] == m], lambda c: (c[0], c[2]), f"{tag(m)}_pairs", f"{m}, model-dataset pairs")

src = {f: FA_SHA[f] for f in FA_SHA}
json.dump({"source": "foundation_arm/analysis/foundation_summary.py",
           "inputs": {"budget_cells": src, "paper": "results/transfer b11/b13/b14/b17 (gate 1 reproduces paper_numbers)"},
           "numbers": OUT}, open(HERE / "foundation_summary.json", "w"), indent=1)
print(f"{len(OUT)} keys written")
