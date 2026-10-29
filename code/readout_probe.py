#!/usr/bin/env python3
"""Readout or representation: refit the linear readout on frozen features at
increasing label budgets.

For each target, draws budgets of 10 to 500 labelled studies by whole patients
(five resamples each), refits a logistic readout on the source model's frozen
features and scores it against the transferred model on the full evaluation
split. Budget samples come from a keyed random stream, so later stages draw the
same samples.

Output (under --out-dir):
    readout_probe_<source>.csv

Usage:
    python readout_probe.py --source XR_WRIST --gpu 0

Requires a GPU for feature extraction. Also imported by later stages for its
budgets, sampler and ``fit_probe``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskdata
import mskmodel as M
import preprocessing_variants as P
from train_source_models import ckpt_path, CsvAppender
from dataset_shift import graz_frame, load_index, load_source, cell_rng

BUDGETS = [10, 25, 50, 100, 200, 500]
N_RESAMPLE = 5
GAP_GATE = 0.10
ALL_ANATOMIES = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND",
                 "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]
DEFAULT_SOURCE = "XR_WRIST"


def extract(model, frame, device, cfg, loader_fn):
    """Frozen trunk features, row order matching `frame`."""
    import torch
    loader = P.make_loader(frame, cfg.eval_batch_size, False, cfg.num_workers,
                           cfg.img_size, loader_fn)
    model.eval()
    n = len(frame)
    feats = None
    logit = np.full(n, np.nan)
    label = np.full(n, -1, dtype=np.int64)
    try:
        with torch.no_grad():
            for x, y, idx in loader:
                x = x.to(device, non_blocking=True)
                z, f = model(x, return_features=True)
                i = idx.numpy()
                f = f.detach().float().cpu().numpy()
                if feats is None:
                    feats = np.zeros((n, f.shape[1]), dtype=np.float32)
                feats[i] = f
                logit[i] = z.detach().float().cpu().numpy()
                label[i] = y.numpy().astype(np.int64)
    finally:
        M.shutdown_loader(loader)
    if (label < 0).any():
        raise RuntimeError("loader skipped rows")
    return feats, logit, label


def sample_studies(frame, n_studies, rng):
    """
    Draw whole PATIENTS until the study budget is met.

    Sampling studies directly would split a patient across the probe-train set
    and the pool it is meant to generalise from. mskdata already drops source
    patients from targets; this preserves the same discipline inside the probe.
    """
    pats = np.asarray(frame.patient_group.unique(), dtype=object)
    rng.shuffle(pats)
    if len(set(pats)) != len(pats):          # shuffle must be a permutation
        raise RuntimeError("patient list corrupted by shuffle")
    taken, seen = [], 0
    for p in pats:
        rows = frame.index[frame.patient_group == p].tolist()
        taken.extend(rows)
        seen += frame.loc[rows, "study"].nunique()
        if seen >= n_studies:
            break
    return frame.loc[taken]


def fit_probe(ftr, ytr, fva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2:
        return None
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(ftr, ytr)
    return clf.decision_function(fva)


def main() -> None:
    import torch
    from shortcut_reliance import score
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-root", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--backbones", nargs="+",
                    default=["densenet121", "swin_tiny_patch4_window7_224"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(M.SEEDS))
    ap.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    ap.add_argument("--resamples", type=int, default=N_RESAMPLE)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--source", default=DEFAULT_SOURCE, choices=ALL_ANATOMIES,
                    help="MURA source anatomy. Non-wrist sources are needed to "
                         "get a SECOND cell with gap >= 0.10: from XR_WRIST only "
                         "shoulder qualifies, but XR_HAND->XR_HUMERUS (gap 0.345) "
                         "and XR_HUMERUS->XR_FINGER (gap 0.393) do.")
    ap.add_argument("--skip-graz", action="store_true",
                    help="GRAZ is wrist-only; skip it for non-wrist sources")
    ap.add_argument("--gpu", type=int, default=0)
    cfg = ap.parse_args()
    SOURCE = cfg.source
    MURA_TARGETS = [a for a in ALL_ANATOMIES if a != SOURCE]
    if SOURCE != "XR_WRIST" and not cfg.skip_graz:
        print("[note] source is not XR_WRIST; GRAZ is wrist-only, so the GRAZ "
              "cell would be a cross-ANATOMY *and* cross-dataset comparison.\n"
              "       Including it anyway, flagged in the CSV as such. Pass "
              "--skip-graz to omit.", flush=True)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({"script": "readout_probe.py v1", "source": SOURCE,
                         "source": cfg.source, "budgets": list(cfg.budgets),
                         "resamples": cfg.resamples,
                         "note": "frozen encoder, readout refit, study budgets, "
                                 "patient-grouped sampling"})
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(cfg.out_dir) / f"readout_probe_{cfg.source}.csv"
    res = CsvAppender(out, prov)
    index = load_index(cfg)
    ck_dir = Path(cfg.ckpt_root) / "source_models"

    # ---- target pools ----------------------------------------------------
    pools = {}
    for t in MURA_TARGETS:
        tr = M.frame_for(index, "mura", t, "train").reset_index(drop=True)
        tr = mskdata.drop_source_patients(index, tr, SOURCE, "train")
        va = M.frame_for(index, "mura", t, "valid").reset_index(drop=True)
        pools[t] = (tr.reset_index(drop=True), va)
    if cfg.skip_graz:
        gtr = gva = None
    else:
        gtr = graz_frame(index, "train", "label_abnormal")
        gva = graz_frame(index, "valid", "label_abnormal")
        ie = lambda f: pd.to_numeric(f.meta_initial_exam, errors="coerce") == 1
        pools["GRAZ_initial"] = (gtr[ie(gtr).to_numpy()].reset_index(drop=True),
                                 gva[ie(gva).to_numpy()].reset_index(drop=True))
    for t, (tr, va) in pools.items():
        print(f"[pool] {t:<14s} train {len(tr):>6} imgs / "
              f"{tr.study.nunique():>5} studies  prev={tr.label.mean():.3f} | "
              f"valid {len(va):>5} / {va.study.nunique():>4}", flush=True)

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            cp = ckpt_path(ck_dir, SOURCE, bb, seed)
            model, _ = load_source(cp, bb, device)
            for t, (tr, va) in pools.items():
                t0 = time.time()
                ftr, _, ytr = extract(model, tr, device, cfg, P.load_standard)
                fva, zva, yva = extract(model, va, device, cfg, P.load_standard)
                st = va.study.to_numpy()
                zs = score(zva, yva, st)["auc_study"]
                rows = [{"backbone": bb, "seed": seed, "target": t,
                         "source": SOURCE, "budget": 0, "resample": 0,
                         "n_train_images": 0,
                         "n_train_studies": 0, "auc_study": zs,
                         "delta_vs_zeroshot": 0.0, "zeroshot_auc": zs,
                         "minutes": (time.time() - t0) / 60.0}]
                for b in cfg.budgets:
                    if tr.study.nunique() < b:
                        continue
                    for r in range(cfg.resamples):
                        rng = cell_rng("probe", bb, seed, t, b, r)
                        sub = sample_studies(tr, b, rng)
                        s = fit_probe(ftr[sub.index.to_numpy()],
                                      tr.loc[sub.index, "label"].to_numpy(), fva)
                        if s is None:
                            continue
                        a = score(s, yva, st)["auc_study"]
                        rows.append({"backbone": bb, "seed": seed, "target": t,
                                     "source": SOURCE,
                                     "budget": b, "resample": r,
                                     "n_train_images": len(sub),
                                     "n_train_studies": int(sub.study.nunique()),
                                     "auc_study": a,
                                     "delta_vs_zeroshot": a - zs,
                                     "zeroshot_auc": zs, "minutes": np.nan})
                res.write(rows)
                got = [r for r in rows if r["budget"] == max(cfg.budgets)]
                print(f"  [probe] {M.BACKBONE_SHORT[bb]:<12s} s{seed} "
                      f"{t:<14s} zs={zs:.4f} "
                      f"n{max(cfg.budgets)}="
                      f"{np.mean([r['auc_study'] for r in got]):.4f}"
                      if got else
                      f"  [probe] {M.BACKBONE_SHORT[bb]} s{seed} {t} zs={zs:.4f}",
                      flush=True)
            del model
            torch.cuda.empty_cache()

    # ---- summary ---------------------------------------------------------
    d = pd.read_csv(out, comment="#")
    print("\n" + "=" * 78)
    print("PROBE AUC BY BUDGET  (readout refit, encoder frozen)")
    print("=" * 78)
    tab = d.pivot_table(index="target", columns="budget", values="auc_study",
                        aggfunc="mean")
    print(tab.round(4).to_string())
    print("\n  budget 0 = zero-shot (the frozen MURA head)")

    print("\n" + "=" * 78)
    print("CROSSOVER: smallest budget where the probe BEATS zero-shot")
    print("=" * 78)
    for t in tab.index:
        row = tab.loc[t]
        zs = row.get(0, np.nan)
        cross = [b for b in cfg.budgets if b in row and row[b] > zs]
        print(f"  {t:<14s} zero-shot {zs:.4f}  crossover at "
              f"{cross[0] if cross else 'NEVER'} studies")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    big = max(cfg.budgets)
    mura = tab.loc[[t for t in tab.index if t.startswith("XR_")], big].mean()
    mura_zs = tab.loc[[t for t in tab.index if t.startswith("XR_")], 0].mean()
    if "GRAZ_initial" in tab.index:
        g, gz = tab.loc["GRAZ_initial", big], tab.loc["GRAZ_initial", 0]
        print(f"  at n={big} studies:")
        print(f"    cross-anatomy mean  zs {mura_zs:.4f} -> {mura:.4f}  "
              f"delta {mura-mura_zs:+.4f}")
        print(f"    GRAZ initial-only   zs {gz:.4f} -> {g:.4f}  "
              f"delta {g-gz:+.4f}")
        print(f"\n  Recovery fractions are reported separately only where the")
        print(f"  gap is >= {GAP_GATE}; raw delta above is primary.")
        diff = (g - gz) - (mura - mura_zs)
        print(f"  GRAZ readout gain minus cross-anatomy readout gain: {diff:+.4f}")
        if abs(diff) < 0.05:
            print("\n  >>> The readout is worth about the SAME on both arms.")
            print("      Whatever separates them is not a readout phenomenon.")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
