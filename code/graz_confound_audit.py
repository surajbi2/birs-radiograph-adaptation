#!/usr/bin/env python3
"""Confound audit for the MURA -> GRAZPEDWRI-DX dataset shift.

The dataset contrast changes many factors at once (site, scanner, age group,
label definition, bit depth, case mix). This script tests three of them: the age effect
(oldest against youngest age band), the label definition (whether specificity
collapses while sensitivity holds) and a shared appearance shortcut (the
cross-anatomy advantage after projecting out the appearance component).

Outputs:
    graz_confound_audit.csv (path set by --out)

Usage:
    python graz_confound_audit.py

CPU only; reads the stored score dumps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mskmodel as M

AGE_BANDS = [(-1, 6, "0-6"), (6, 10, "6-10"), (10, 13, "10-13"),
             (13, 16, "13-16"), (16, 100, "16-19")]
CROSS_ANATOMY_RANGE = (0.7665, 0.8448)     # hand .. forearm, wrist source


def load_dumps(dump_dir: Path, pattern: str) -> list:
    d = sorted(Path(dump_dir).glob(pattern))
    if not d:
        raise SystemExit(f"no dumps matching {pattern} in {dump_dir}")
    return d


def main() -> None:
    from shortcut_reliance import score
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--dump-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                              "dumps"))
    ap.add_argument("--matrix", default=str(Path(M.DEFAULT_RESULTS) / "layerwise" /
                                            "shortcut_reliance_matrix.csv"))
    ap.add_argument("--prep", default="standard")
    ap.add_argument("--label", default="label_abnormal")
    ap.add_argument("--out", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "graz_confound_audit.csv"))
    cfg = ap.parse_args()

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    init = index.set_index("uid").get("meta_initial_exam")
    age = index.set_index("uid").get("meta_age")
    if init is None or age is None:
        raise SystemExit("index lacks meta_initial_exam / meta_age")

    dumps = load_dumps(Path(cfg.dump_dir),
                       f"zeroshot__{cfg.label}__{cfg.prep}__*.npz")
    print(f"using {len(dumps)} zero-shot dumps ({cfg.prep}, {cfg.label})\n")
    rows = []

    # ---------------------------------------------------------------- C1 ----
    print("=" * 78)
    print("C1. GROWTH PLATES -- is MURA->GRAZ an ANATOMY comparison in disguise?")
    print("=" * 78)
    print("  Restricted to INITIAL EXAMS. Bands are age in years.\n")
    per_band = {b[2]: [] for b in AGE_BANDS}
    per_band_n = {b[2]: 0 for b in AGE_BANDS}
    for d in dumps:
        z = np.load(d, allow_pickle=True)
        uid = z["uid"]
        a = pd.to_numeric(age.reindex(uid), errors="coerce").to_numpy()
        ie = pd.to_numeric(init.reindex(uid), errors="coerce").to_numpy() == 1
        for lo, hi, nm in AGE_BANDS:
            m = ie & (a > lo) & (a <= hi)
            if m.sum() < 80 or len(np.unique(z["label"][m])) < 2:
                continue
            s = score(z["logit"][m], z["label"][m], z["study"][m])
            per_band[nm].append(s["auc_study"])
            per_band_n[nm] = int(m.sum())
    print(f"  {'band':<8}{'n':>7}{'zero-shot AUC':>16}{'seeds':>8}")
    means = {}
    for _, _, nm in AGE_BANDS:
        v = per_band[nm]
        if not v:
            print(f"  {nm:<8}{per_band_n[nm]:>7}{'-':>16}{0:>8}")
            continue
        means[nm] = float(np.mean(v))
        print(f"  {nm:<8}{per_band_n[nm]:>7}{np.mean(v):>16.4f}{len(v):>8}")
        rows.append({"analysis": "C1_age_band", "band": nm,
                     "n": per_band_n[nm], "auc": float(np.mean(v))})
    if len(means) >= 2:
        ks = list(means)
        eff = means[ks[-1]] - means[ks[0]]
        print(f"\n  oldest - youngest = {eff:+.4f}")
        print(f"  cross-anatomy range for reference: "
              f"{CROSS_ANATOMY_RANGE[0]:.4f} - {CROSS_ANATOMY_RANGE[1]:.4f}")
        if eff >= 0.08 and means[ks[-1]] >= CROSS_ANATOMY_RANGE[0]:
            print("\n  >>> C1a: GROWTH PLATES DOMINATE. MURA->GRAZ is an ANATOMY")
            print("      comparison in disguise. The Stage B conclusion is")
            print("      WITHDRAWN and the anatomy framing is SUPPORTED.")
        elif eff < 0.04:
            print("\n  >>> C1b: age is not the driver. The bundle is.")
        else:
            print("\n  >>> intermediate: age contributes but does not dominate.")
            print("      Report the age-stratified table, not a single number.")

    # ---------------------------------------------------------------- C2 ----
    print("\n" + "=" * 78)
    print("C2. LABEL MISMATCH -- does SPECIFICITY collapse while SENSITIVITY holds?")
    print("=" * 78)
    print("  *** THE FIRST VERSION OF THIS TEST WAS INVALID. *** It thresholded")
    print("  at the 59.1st PERCENTILE OF THE TARGET'S OWN LOGITS, which fixes the")
    print("  positive-call rate at 40.9% by construction. Over-calling is exactly")
    print("  a change in the positive-call rate, so the test normalised away the")
    print("  thing it was supposed to detect and returned C2b mechanically.")
    print("  It reported sensitivity 0.5783 / specificity 0.7165; that pair is a")
    print("  restatement of AUC 0.715 at a fixed rate, not evidence about labels.\n")
    print("  CORRECT TEST: apply the model's OWN native decision boundary")
    print("  (logit = 0, i.e. sigmoid = 0.5, the boundary BCE training produces)")
    print("  and compare the positive-call RATE against the true prevalence.")
    print("  An abnormality detector judged against a fracture label should call")
    print("  MORE positive than there are fractures.\n")
    src_prev = 0.409
    rate, prev_t, sens0, spec0 = [], [], [], []
    for d in dumps:
        z = np.load(d, allow_pickle=True)
        uid = z["uid"]
        ie = pd.to_numeric(init.reindex(uid), errors="coerce").to_numpy() == 1
        lg, y = z["logit"][ie], z["label"][ie]
        if len(np.unique(y)) < 2:
            continue
        pred = lg >= 0.0
        rate.append(float(pred.mean()))
        prev_t.append(float(y.mean()))
        sens0.append(float(pred[y == 1].mean()))
        spec0.append(float((~pred[y == 0]).mean()))
    if rate:
        r_, p_ = float(np.mean(rate)), float(np.mean(prev_t))
        print(f"  positive-call rate at logit>=0 : {r_:.4f}")
        print(f"  true prevalence in the subset  : {p_:.4f}")
        print(f"  source training prevalence     : {src_prev:.4f}")
        print(f"  over-call                      : {r_ - p_:+.4f}")
        print(f"  sensitivity {np.mean(sens0):.4f}   "
              f"specificity {np.mean(spec0):.4f}  (at the NATIVE boundary)")
        rows.append({"analysis": "C2_native_threshold", "call_rate": r_,
                     "true_prevalence": p_,
                     "sensitivity": float(np.mean(sens0)),
                     "specificity": float(np.mean(spec0))})
        if (r_ - p_) > 0.15:
            print("\n  >>> C2a: the model OVER-CALLS by a wide margin. Consistent")
            print("      with an abnormality detector meeting a fracture label.")
            print("      The gap OVERSTATES representation failure and the")
            print("      readout-refit experiment is mandatory.")
        elif (r_ - p_) < -0.15:
            print("\n  >>> UNDER-calling. Not a label-breadth artifact; the source")
            print("      logits have shifted DOWN on this domain, which is a")
            print("      calibration/covariate-shift signature instead.")
        else:
            print("\n  >>> C2b: call rate tracks prevalence. Label breadth is not")
            print("      producing a systematic bias. Note this still does NOT")
            print("      rule out label mismatch costing AUC -- only the readout")
            print("      refit can separate that from representation failure.")

    # ---------------------------------------------------------------- C3 ----
    print("\n" + "=" * 78)
    print("C3. SHORTCUT BONUS -- does within-MURA transfer get help GRAZ cannot?")
    print("=" * 78)
    c = pd.read_csv(cfg.matrix, comment="#", low_memory=False)
    BB = ["densenet121", "swin_tiny_patch4_window7_224"]
    w = c[(c.source == "XR_WRIST") & (c.backbone.isin(BB))]
    ca_raw = float(w.zs_auc_study.mean())
    ca_res = float(w.resid_auc_study.mean())
    gz_raw, gz_res = [], []
    for d in dumps:
        z = np.load(d, allow_pickle=True)
        uid = z["uid"]
        ie = pd.to_numeric(init.reindex(uid), errors="coerce").to_numpy() == 1
        gz_raw.append(score(z["logit"][ie], z["label"][ie], z["study"][ie])["auc_study"])
        gz_res.append(score(z["resid"][ie], z["label"][ie], z["study"][ie])["auc_study"])
    gr, gs = float(np.mean(gz_raw)), float(np.mean(gz_res))
    print(f"  {'':<26}{'raw':>10}{'residual':>12}{'lost to projection':>21}")
    print(f"  {'cross-anatomy (wrist src)':<26}{ca_raw:>10.4f}{ca_res:>12.4f}"
          f"{ca_res-ca_raw:>+21.4f}")
    print(f"  {'GRAZ initial-only':<26}{gr:>10.4f}{gs:>12.4f}{gs-gr:>+21.4f}")
    adv_raw, adv_res = ca_raw - gr, ca_res - gs
    print(f"\n  cross-anatomy advantage over GRAZ: raw {adv_raw:+.4f} -> "
          f"residual {adv_res:+.4f}   shrink {adv_raw-adv_res:+.4f}")
    rows.append({"analysis": "C3_projection", "cross_raw": ca_raw,
                 "cross_resid": ca_res, "graz_raw": gr, "graz_resid": gs})
    if (adv_raw - adv_res) >= 0.03:
        print("\n  >>> C3a: part of the within-MURA advantage was SHARED")
        print("      ACQUISITION SHORTCUT. The raw comparison was unfair to")
        print("      GRAZ; report the residual comparison alongside it.")
    else:
        print("\n  >>> C3b: the shortcut bonus does not explain the difference.")

    print("\n" + "=" * 78)
    print("C4. CASE-MIX SELECTION -- NOT TESTABLE, DECLARE IT")
    print("=" * 78)
    print("  MURA: study-level radiologist abnormality, one US institution.")
    print("  GRAZ: consecutive pediatric wrist radiographs, one Austrian centre.")
    print("  FracAtlas: curated fracture collection, no patient identifiers.")
    print("  The difficulty of the positive cases is not matched and cannot be")
    print("  matched with the data available. State it; do not model around it.")

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg.out, index=False)
    print(f"\nwritten to {cfg.out}")


if __name__ == "__main__":
    main()
