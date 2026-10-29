"""Independent second pass over the headline numbers of foundation_summary.py.

Shares no code with foundation_summary.py: reads the cell files from ../cells/ directly,
rebuilds the schedule with its own loop and compares every headline value with
foundation_summary.json. Exits non-zero on any difference.

Usage:
    python foundation_summary_verify.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "cells"
CSV3 = HERE.parent / "cells"
J = {k[4:]: v["value"] for k, v in json.load(open(HERE / "foundation_summary.json"))["numbers"].items()}
NS = (10, 25, 50, 100, 200, 500)
LAMS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
ANAT = {"XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER"}
OUTSIDE = {"GRAZ_initial", "MFXR", "FRAC_XR_HAND"}
PUB = {10: 0, 25: 0, 50: 0, 100: 0.1, 200: 0.75, 500: 1.0}

checked = 0


def same(k, v, tol=1e-12):
    global checked
    ref = J[k]
    ok = ref == v if isinstance(v, str) else abs(ref - v) <= tol
    assert ok, (k, ref, v)
    checked += 1


for arm in ("a", "b"):
    rows = []
    for p in (CSV / f"cells_arm_{arm}.csv", CSV3 / f"cells_arm_{arm}_dinov3.csv"):
        with open(p, encoding="utf-8") as fh:
            rows += [r for r in csv.DictReader(l for l in fh if l[0] != "#")]
    for enc in ("skelex", "rad-dino", "medsiglip", "dinov3"):
        t = f"{arm}_{enc.replace('-', '')}"
        d = {}    # (setting, target, seed, budget, resample) -> delta
        for r in rows:
            if r["backbone"] == enc and r["setting"] != "zeroshot":
                d[(r["setting"], r["target"], r["seed"], int(r["budget"]), r["resample"])] = float(r["delta_vs_zeroshot"])
        keys = sorted({k[1:] for k in d})
        lam = lambda x: "shrink_lam1" if x == 1.0 else f"shrink_lam{x}"

        def schedule(targets):
            s, prev = {}, 0.0
            for n in NS:
                best, bestm = 0.0, None
                for x in LAMS:   # ascending, so a strict ">" keeps the smaller weight on ties
                    v = [d[(lam(x),) + k] for k in keys if k[0] in targets and k[2] == n]
                    if sum(e < 0 for e in v) / len(v) <= 0.05:
                        m = sum(v) / len(v)
                        if bestm is None or m > bestm + 1e-12:
                            best, bestm = x, m
                pick = best if bestm is not None and bestm > 0 else 0.0
                prev = max(prev, pick)
                s[n] = prev
            return s

        def harm_gain(s, targets):
            v = [d[(lam(s[k[2]]) if s[k[2]] else "shrink_lam0",) + k] for k in keys if k[0] in targets]
            return sum(e < 0 for e in v), sum(v) / len(v)

        s = schedule(ANAT)
        same(f"{t}_o2_schedule", ", ".join(f"{n}:{s[n]:g}" for n in NS))
        k, g = harm_gain(s, OUTSIDE)
        same(f"{t}_o2_ext_harmed_cells", k)
        same(f"{t}_o2_ext_mean_gain", g, 1e-12)
        loo_k = sum(harm_gain(schedule(ANAT - {a}), {a})[0] for a in ANAT)
        same(f"{t}_o2_loo_n_harmed", loo_k)
        for lab, tg in (("ext", OUTSIDE), ("cal", ANAT)):
            k, g = harm_gain(PUB, tg)
            same(f"{t}_o3_published_{lab}_n_harmed", k)
            same(f"{t}_o3_published_{lab}_mean_gain", g, 1e-12)
            for lo, f in (("le50", lambda n: n <= 50), ("all", lambda n: True)):
                v = [d[("refit_nestedC",) + kk] for kk in keys if kk[0] in tg and f(kk[2])]
                same(f"{t}_o1_refit_{lab}_{lo}_n_harmed", sum(e < 0 for e in v))
            for p, col in (("cv_lambda", "shrink_nested"), ("refit_own", "refit_ownC"), ("l2sp_cv", "l2sp_cv")):
                v = [d[(col,) + kk] for kk in keys if kk[0] in tg]
                same(f"{t}_{p}_{lab}_n_harmed", sum(e < 0 for e in v))
print(f"second pass: {checked} b28 values agree")
