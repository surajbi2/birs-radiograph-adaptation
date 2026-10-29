#!/usr/bin/env python3
"""Patient-clustered bootstrap intervals for paired AUC differences.

Resamples patients (not images or studies) within each evaluation split and
recomputes paired differences between models scored on the same images, giving
percentile intervals that respect the clustering of images within patients.

Output:
    bootstrap_ci.csv (path set by --out)

Usage:
    python bootstrap_intervals.py --inventory          # list available dumps
    python bootstrap_intervals.py --n-boot 10000 --out results/transfer/bootstrap_ci.csv

CPU only.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (42, 123, 456, 789, 2024)
BACKBONES = ("densenet121", "convnext_tiny", "swin_tiny_patch4_window7_224")
ANATOMY = ("XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS",
           "XR_SHOULDER")
DATASET = ("GRAZ_initial",)
ALL_TARGETS = ANATOMY + DATASET + ("XR_WRIST",)   # wrist is a target on the diagonal
CONDITIONS = ("zeroshot", "oracle", "reference")

# `initial_only` is the GRAZ reference model scored on INITIAL EXAMINATIONS,
# which is the 0.960 the paper reports and the only GRAZ reference it uses.
#
# `ceiling` is deliberately NOT aliased. It is the same model scored on the
# FULL pool, which runs about 0.968 and belongs to a cohort the paper does not
# report. Folding the two together would silently mix the two cohorts into one
# "reference" condition and put an interval on a quantity that appears nowhere.
ALIASES = {"initial_only": "reference"}


# ------------------------------------------------------------------ scoring --

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC. Ties averaged, which matters once resampling duplicates."""
    n1 = float(y.sum())
    n0 = float(y.size - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def study_scores(logit, label, study):
    """Sigmoid, then mean per study. Returns (study_id, y, score) arrays."""
    p = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
    d = pd.DataFrame({"study": study, "y": label, "p": p})
    g = d.groupby("study", sort=True).agg(y=("y", "max"), p=("p", "mean"))
    return g.index.to_numpy(), g["y"].to_numpy(), g["p"].to_numpy()


# ------------------------------------------------------------------- dumps --

def parse_name(path: Path):
    r"""Two naming schemes are in use, and the difference is not cosmetic.

        diag__<target>__<backbone>__seed<k>.npz          per-anatomy arms
        frac__<source>__<backbone>__seed<k>.npz
        <cond>__<label>__<prep>__<backbone>__seed<k>.npz GRAZ arm

    The GRAZ arm carries NO target token, because that whole arm is MURA wrist
    to GRAZPEDWRI-DX and the target is implicit. Requiring a token silently
    discarded every GRAZ dump, which is most of them.
    """
    stem = path.stem
    parts = stem.split("__")
    if len(parts) < 2:
        return None
    cond = ALIASES.get(parts[0], parts[0])
    m = re.search(r"seed(\d+)", stem)
    seed = int(m.group(1)) if m else None
    bb = next((b for b in BACKBONES if b in stem), None)
    tgt = next((t for t in ALL_TARGETS if t in stem), None)
    if tgt is None:
        # no explicit target: GRAZ arm if it carries a label/prep token pair
        if any(p.startswith("label_") for p in parts):
            tgt = "GRAZ_initial"
        elif "GRAZ" in stem:
            tgt = "GRAZ_initial"
    label = next((p for p in parts if p.startswith("label_")), None)
    prep = next((p for p in parts if p in ("standard", "p0p100", "maskpct",
                                           "histmatch")), None)
    return {"path": path, "condition": cond, "target": tgt, "backbone": bb,
            "seed": seed, "label": label, "prep": prep, "stem": stem}


def inventory(dump_dir: Path) -> pd.DataFrame:
    rows, unparsed = [], []
    vocab = defaultdict(int)                 # every prefix, parsed or not
    for p in sorted(dump_dir.rglob("*.npz")):
        vocab[p.stem.split("__")[0]] += 1
        r = parse_name(p)
        if r is None or r["target"] is None or r["backbone"] is None:
            unparsed.append(p.name)
            continue
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"scanned {dump_dir}")
    print(f"  {len(rows)} parsed, {len(unparsed)} unparsed")
    # The condition vocabulary is the point of the inventory, so report it for
    # every file on disk. Reporting only parsed names hides the thing you came
    # to look for when the parser is the part that is wrong.
    print("\n  condition prefixes on disk (all files):")
    for k in sorted(vocab):
        print(f"    {k:<24} {vocab[k]:4d}")
    if unparsed:
        print("  unparsed examples:")
        for u in unparsed[:8]:
            print(f"    {u}")
    if df.empty:
        print("\n  NO USABLE DUMPS. The bootstrap cannot run.")
        return df
    print("\n  conditions found:", sorted(df.condition.unique()))
    print("  targets found   :", sorted(df.target.dropna().unique()))
    print("  backbones found :", sorted(df.backbone.dropna().unique()))
    print("  seeds found     :", sorted(int(s) for s in df.seed.dropna().unique()))
    if "label" in df:
        print("  labels found    :", sorted(df.label.dropna().unique()))
        print("  preps found     :", sorted(df.prep.dropna().unique()))
    print("\n  cells per condition (target x backbone x seed):")
    for c, g in df.groupby("condition"):
        print(f"    {c:<24} {len(g):4d}")
    have = set(df.condition.unique())
    missing = [c for c in CONDITIONS if c not in have]
    print()
    if missing:
        print(f"  MISSING CONDITIONS: {missing}")
        print("  The encoder residual is reference - oracle and the gap is")
        print("  reference - zero shot. Whatever is missing cannot be")
        print("  bootstrapped, and this script will not substitute a different")
        print("  quantity under the same name. Re-dump the missing condition or")
        print("  restrict the claim.")
    else:
        print("  all three conditions present; the full decomposition can be")
        print("  bootstrapped.")
    # one file inspected so the key names are on the record
    z = np.load(df.path.iloc[0], allow_pickle=True)
    print(f"\n  keys in {df.stem.iloc[0]}: {sorted(z.files)}")
    return df


def patient_of(uid, index: pd.DataFrame | None):
    """Patient id per image. MURA encodes it in the path; GRAZ needs the index."""
    uid = np.asarray(uid).astype(str)
    pat = np.array([None] * uid.size, dtype=object)
    m = np.array([bool(re.search(r"patient(\d+)", u)) for u in uid])
    pat[m] = [re.search(r"patient(\d+)", u).group(1) for u in uid[m]]
    if (~m).any() and index is not None and "patient_id" in index.columns:
        lut = index.set_index("uid")["patient_id"]
        pat[~m] = lut.reindex(uid[~m]).to_numpy()
    return pat


# --------------------------------------------------------------- bootstrap --

def load_cells(df: pd.DataFrame, index):
    """cell -> {condition: study-level frame with patient, y, p}.

    UID IS THE JOIN KEY, NOT STUDY. The conditions were written by different
    scripts, and only oracle_readout_cv's dumps carry `patient` and a study id built
    from the same frame. Joining on study silently produced an empty
    intersection; joining on uid and taking study and patient from whichever
    condition has them keeps every cell on one consistent labelling.
    """
    raw = defaultdict(dict)
    for r in df.itertuples():
        z = np.load(r.path, allow_pickle=True)
        d = pd.DataFrame({
            "uid": np.asarray(z["uid"]).astype(str),
            "study": np.asarray(z["study"]).astype(str),
            "y": np.asarray(z["label"]).astype(int),
            "p": 1.0 / (1.0 + np.exp(-np.asarray(z["logit"]).astype(np.float64))),
        })
        d["patient"] = (np.asarray(z["patient"]).astype(str)
                        if "patient" in z.files else None)
        raw[(r.target, r.backbone, r.seed)][r.condition] = d

    cells = defaultdict(dict)
    for k, conds in raw.items():
        if not all(c in conds for c in CONDITIONS):
            continue
        # donor: a condition that knows the patient for each uid
        donor = next((d for d in conds.values() if d["patient"].notna().any()),
                     None)
        if donor is None:
            if index is None:
                raise SystemExit(
                    f"!! {k}: no dump carries `patient` and no --index given. "
                    "Patients are the resampling unit; without them this would "
                    "silently become a study-level bootstrap with intervals "
                    "that are too narrow. Pass --index.")
            donor = conds[CONDITIONS[0]].copy()
            donor["patient"] = patient_of(donor["uid"].to_numpy(), index)
        lut = donor.dropna(subset=["patient"]).drop_duplicates("uid")
        lut = lut.set_index("uid")[["study", "patient"]]

        common = None
        for c in CONDITIONS:
            u = pd.Index(conds[c]["uid"]).intersection(lut.index)
            common = u if common is None else common.intersection(u)
        if len(common) == 0:
            raise SystemExit(
                f"!! {k}: the three conditions share no uid. They were scored "
                "on different images, so no paired difference exists. Check "
                "that the reference dump covers the same pool.")
        for c in CONDITIONS:
            d = conds[c].drop(columns=["study", "patient"]).set_index("uid")
            d = d.loc[common].join(lut)
            n_drop = len(conds[c]) - len(d)
            if n_drop:
                print(f"  [align] {k} {c}: dropped {n_drop} of "
                      f"{len(conds[c])} images")
            g = (d.groupby("study", sort=True)
                   .agg(patient=("patient", "first"), y=("y", "max"),
                        p=("p", "mean")))
            cells[k][c] = g
    return cells


def run(cells, n_boot: int, rng: np.random.Generator):
    """Resample patients once per replicate; recompute every cell on it."""
    # patients are shared within a target, so index them per target
    per_target_patients = {}
    for (t, bb, sd), conds in cells.items():
        any_g = next(iter(conds.values()))
        per_target_patients.setdefault(t, np.unique(any_g["patient"].to_numpy()))

    keys = sorted(cells)
    out = defaultdict(list)
    for b in range(n_boot + 1):          # replicate 0 is the point estimate
        draw = {}
        for t, pats in per_target_patients.items():
            draw[t] = (pats if b == 0
                       else rng.choice(pats, size=pats.size, replace=True))
        for (t, bb, sd) in keys:
            conds = cells[(t, bb, sd)]
            if not all(c in conds for c in CONDITIONS):
                continue
            sel = None
            vals = {}
            for c in CONDITIONS:
                g = conds[c]
                if sel is None:
                    idx = pd.Index(g["patient"].to_numpy())
                    sel = np.concatenate([np.flatnonzero(idx == p) for p in draw[t]])
                vals[c] = auc(g["y"].to_numpy()[sel], g["p"].to_numpy()[sel])
            out[(t, bb, sd, "gap")].append(vals["reference"] - vals["zeroshot"])
            out[(t, bb, sd, "readout")].append(vals["oracle"] - vals["zeroshot"])
            out[(t, bb, sd, "residual")].append(vals["reference"] - vals["oracle"])
        if b and b % 500 == 0:
            print(f"    {b}/{n_boot}", file=sys.stderr)
    return out


def summarise(out, n_boot):
    """Per-target and pooled intervals for each decomposition term."""
    rows = []
    terms = ("gap", "readout", "residual")
    byt = defaultdict(lambda: defaultdict(list))
    for (t, bb, sd, term), series in out.items():
        byt[(t, term)][0].append(np.asarray(series))
    for (t, term), d in byt.items():
        arr = np.nanmean(np.stack(d[0]), axis=0)      # mean over cells, per replicate
        rows.append({"scope": t, "term": term, "estimate": arr[0],
                     "lo": np.nanpercentile(arr[1:], 2.5),
                     "hi": np.nanpercentile(arr[1:], 97.5),
                     "n_boot": n_boot})
    # pooled anatomy vs dataset, and the contrast the paper turns on
    for term in terms:
        a = [np.asarray(s) for (t, bb, sd, tm) in out
             for s in [out[(t, bb, sd, tm)]] if tm == term and t in ANATOMY]
        g = [np.asarray(s) for (t, bb, sd, tm) in out
             for s in [out[(t, bb, sd, tm)]] if tm == term and t in DATASET]
        if not a or not g:
            continue
        am, gm = np.nanmean(np.stack(a), axis=0), np.nanmean(np.stack(g), axis=0)
        for scope, arr in (("ANATOMY_pooled", am), ("DATASET_pooled", gm),
                           ("DATASET_minus_ANATOMY", gm - am)):
            rows.append({"scope": scope, "term": term, "estimate": arr[0],
                         "lo": np.nanpercentile(arr[1:], 2.5),
                         "hi": np.nanpercentile(arr[1:], 97.5),
                         "n_boot": n_boot})
    return pd.DataFrame(rows).sort_values(["term", "scope"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", type=Path,
                    default=Path.home() / "diagnosis/results/transfer/dumps")
    ap.add_argument("--extra-dumps", type=Path, nargs="*", default=[],
                    help="further dump dirs to draw the reference condition "
                         "from, for targets oracle_readout_cv could not dump it for")
    ap.add_argument("--index", type=Path, default=None,
                    help="dataset index csv, needed for GRAZ patient ids")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    cfg = ap.parse_args()

    df = inventory(cfg.dumps)
    if not df.empty and cfg.extra_dumps:
        # oracle_readout_cv cannot dump the GRAZ reference: that model was trained by
        # graz_initial_only_ceiling, not by it. The dump already exists in the original dumps
        # directory as initial_only__label_abnormal__standard__*, so pull only
        # that condition in, restricted to the label and prep the paper uses,
        # and never let it displace a cell the primary directory already has.
        for extra in cfg.extra_dumps:
            print(f"\nextra dumps: {extra}")
            e = inventory(extra)
            if e.empty:
                continue
            e = e[(e.condition == "reference")
                  & (e.label.isin([None, "label_abnormal"]) | e.label.isna())
                  & (e.prep.isin([None, "standard"]) | e.prep.isna())]
            key = ["condition", "target", "backbone", "seed"]
            have = set(map(tuple, df[key].itertuples(index=False)))
            e = e[[tuple(r) not in have
                   for r in e[key].itertuples(index=False)]]
            print(f"  adopting {len(e)} reference cell(s): "
                  f"{sorted(e.target.unique())}")
            df = pd.concat([df, e], ignore_index=True)
    if cfg.inventory or df.empty:
        return 0 if not df.empty else 1
    if not set(CONDITIONS).issubset(set(df.condition.unique())):
        print("\nrefusing to run: see MISSING CONDITIONS above")
        return 1

    index = None
    if cfg.index and cfg.index.exists():
        index = pd.read_csv(cfg.index, comment="#", low_memory=False)

    print("\nloading cells")
    cells = load_cells(df[df.condition.isin(CONDITIONS)], index)
    print(f"  {len(cells)} cells")
    print(f"\nbootstrapping {cfg.n_boot} patient-clustered resamples")
    out = run(cells, cfg.n_boot, np.random.default_rng(cfg.seed))
    res = summarise(out, cfg.n_boot)

    pd.set_option("display.width", 120)
    print("\n" + res.to_string(index=False,
                               float_format=lambda v: f"{v:+.4f}"))
    if cfg.out:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(cfg.out, index=False)
        print(f"\nwrote {cfg.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
