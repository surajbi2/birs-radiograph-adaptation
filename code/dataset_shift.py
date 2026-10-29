#!/usr/bin/env python3
"""Stage B: dataset shift (MURA wrist -> GRAZPEDWRI-DX wrist) and an acquisition
device control inside GRAZPEDWRI-DX.

Stages:
    precheck   label and metadata checks on the index
    diag       diagonal: every MURA source evaluated on every MURA target
    src_prep   wrist source models under each preprocessing variant
    ceiling    GRAZPEDWRI-DX models trained and evaluated in-domain
    zeroshot   MURA wrist models applied unchanged to GRAZPEDWRI-DX
    scanner    train on one device manufacturer, test on the other

Outputs (under --out-dir):
    appearance_diagonal.csv, source_preprocessing.csv, graz_ceiling.csv,
    graz_zeroshot.csv, graz_scanner_control.csv and per-cell .npz dumps

Usage:
    python dataset_shift.py --stage precheck
    python dataset_shift.py --stage zeroshot --gpu 0

Requires a GPU (except precheck).
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskdata
import mskmodel as M
import preprocessing_variants as P
from train_source_models import ckpt_path, CsvAppender
from measure_appearance import measure_one, FEATURES

# project_out and score are imported from shortcut_reliance, NOT reimplemented,
# so the instrument here is provably the same one that produced the 42-pair
# matrix. The import is deferred because shortcut_reliance pulls in
# layerwise_unfreezing -> torch, and `--stage precheck` / `--stage selftest` must run
# on a login node with no GPU stack.
_SR = None


def _sr():
    global _SR
    if _SR is None:
        import shortcut_reliance as _mod
        _SR = _mod
    return _SR


def project_out(z, X):
    return _sr().project_out(z, X)


def score(logit, label, study):
    return _sr().score(logit, label, study)

MURA_ANATOMIES = list(mskdata.MURA_ANATOMIES)
GRAZ_ANATOMY = "XR_WRIST"              # mskdata.index_graz sets this
GRAZ_LABELS = ("label_abnormal", "label_fracture")

# TRAINING label vs SCORING labels.
#
# The analysis plan trained one model per label to test prediction P1
# (appearance reliance higher under the metal-inclusive label). The precheck
# made that test uninformative: the two GRAZ labels agree on 98.1% of images (prevalence 0.6861 vs
# 0.6666, 396 of 20,327 disagree, 115 of those carry metal). Two models trained
# on labels that differ on 1.9% of rows are the same model.
#
# Worse, appearance_r2 is the R^2 of the LOGIT regressed on the descriptors and
# does not involve the label at all, so the quantity P1 compared would have been
# identical across arms up to the difference between two near-identical models.
#
# So: train once on TRAIN_LABEL, score the SAME logits against both labels.
# Scoring is free -- same forward pass, different y vector -- and the second
# label is reported as a sensitivity check, not as a concept-shift arm.
# The real test of section 4 on GRAZ moved to graz_label_audit.py (P1').
TRAIN_LABEL = "label_abnormal"          # concept-matched to MURA's abnormality
DEFAULT_BACKBONES = ["densenet121", "swin_tiny_patch4_window7_224"]
N_PERM = 200                            # matches shortcut_reliance.py v2
SCANNER_MIN_TRAIN_STUDIES = 500         # runnability gate fixed in the analysis plan


# ================================================================== helpers ==

def cell_rng(*parts) -> np.random.RandomState:
    """crc32, not hash(): hash() is salted per process and is not reproducible."""
    key = "|".join(str(p) for p in parts).encode()
    return np.random.RandomState(zlib.crc32(key))


def descriptors_for(frame: pd.DataFrame) -> pd.DataFrame:
    """
    The five appearance descriptors, ALWAYS from the standard image load.

    The instrument is held fixed across preprocessing arms on purpose. Changing
    what the model sees AND what the instrument measures at the same time makes
    the arms uninterpretable.
    """
    rows = []
    for p in frame.path:
        try:
            rows.append(measure_one(p))
        except Exception as e:
            rows.append({k: np.nan for k in FEATURES})
            print(f"  !! descriptor failed on {p}: {e}", flush=True)
    return pd.DataFrame(rows, index=frame.index)[FEATURES]


def prep_descriptor_summary(frame: pd.DataFrame, loader_fn, n: int = 300,
                            seed: int = 0) -> Dict[str, float]:
    """
    Diagnostic only: what the descriptors look like on the pixels the model
    ACTUALLY sees in this arm. Never used for the projection. Sampled, because
    it is a sanity readout and not a result.
    """
    if len(frame) == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frame), size=min(n, len(frame)), replace=False)
    mu, sd = [], []
    for i in idx:
        try:
            g = loader_fn(frame.iloc[int(i)]["path"])[..., 0].astype(np.float32)
        except Exception:
            continue
        mu.append(float(g.mean()))
        sd.append(float(g.std()))
    if not mu:
        return {}
    return {"prep_proc_mean": float(np.mean(mu)),
            "prep_proc_std": float(np.mean(sd))}


def projection_cell(z: np.ndarray, y: np.ndarray, st: np.ndarray,
                    X: np.ndarray, rng_parts) -> Dict:
    """
    One projection measurement, identical in method to shortcut_reliance.py v2.

      delta > 0  removing the appearance component IMPROVES transfer. The
                 encoder imported a wrongly-signed mapping. The one unambiguous
                 signature -- no amount of legitimate appearance/content
                 correlation explains REMOVING a component and gaining AUC.
      delta ~ 0  no reliance.
      delta < 0  appearance carries genuine transferable signal.

    R^2 OVERSTATES reliance: appearance correlates with real anatomical content,
    so a model reading real pathology still produces appearance-correlated
    logits. The permutation control rules out "any five linear directions", not
    "appearance proxies content".
    """
    base = score(z, y, st)
    resid, r2 = project_out(z, X)
    red = score(resid, y, st)

    rng = cell_rng(*rng_parts)
    perm_std, perm_r2 = [], []
    for _ in range(N_PERM):
        rp, r2p = project_out(z, X[rng.permutation(len(X))])
        perm_std.append(score(rp, y, st)["auc_study"])
        perm_r2.append(r2p)

    # fitted coefficients on standardised descriptors -- the analysis plan
    # requires reporting these on BOTH sides, because a map fitted on GRAZ is a
    # DIFFERENT linear map from one fitted on MURA even with identical code.
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    fit = LinearRegression().fit(Xs, z)
    coefs = {f"coef_{f}": float(c) for f, c in zip(FEATURES, fit.coef_)}

    out = {
        "zs_auc_image": base["auc_image"], "zs_auc_study": base["auc_study"],
        "appearance_r2": r2,
        "resid_auc_image": red["auc_image"], "resid_auc_study": red["auc_study"],
        "delta_image": red["auc_image"] - base["auc_image"],
        "delta_study": red["auc_study"] - base["auc_study"],
        "perm_r2_mean": float(np.mean(perm_r2)),
        "perm_delta_study_mean": float(np.mean(perm_std) - base["auc_study"]),
        "perm_delta_study_sd": float(np.std(perm_std, ddof=1)),
    }
    out.update(coefs)
    return out, resid


def save_dump(path: Path, frame: pd.DataFrame, z: np.ndarray, resid: np.ndarray,
              y: np.ndarray, st: np.ndarray, X: np.ndarray, meta: Dict) -> None:
    """
    Per-image dump. Subgroup analysis (cast, age, projection, manufacturer,
    metal) is then post-hoc and costs nothing. Carrying the metadata here is the
    difference between one run and five.
    """
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {}
    for c in ("has_cast", "has_metal", "has_osteopenia", "has_diagnosis_uncertain",
              "meta_age", "meta_device_manufacturer", "meta_projection",
              "meta_laterality", "meta_gender"):
        if c in frame.columns:
            cols[c] = frame[c].to_numpy()
    np.savez_compressed(
        path, uid=frame["uid"].to_numpy(), study=st, label=y,
        logit=z, resid=resid, descriptors=X, descriptor_names=np.array(FEATURES),
        meta=json.dumps(meta, default=str), **cols)


def load_index(cfg) -> pd.DataFrame:
    idx = pd.read_csv(cfg.index, comment="#", low_memory=False)
    for c in ("uid", "dataset", "path", "anatomy", "patient_group", "study", "split"):
        if c not in idx.columns:
            raise RuntimeError(f"index is missing required column {c!r}")
    return idx


def graz_frame(index: pd.DataFrame, split: str, label_col: str) -> pd.DataFrame:
    f = M.frame_for(index, "graz", GRAZ_ANATOMY, split, label_col
                    ).reset_index(drop=True)
    # MURA and GRAZ patient ids are namespaced ("mura/..", "graz/..") so overlap
    # is impossible by construction. Assert anyway -- it is free.
    mura_pg = set(index[index.dataset == "mura"]["patient_group"])
    bad = set(f["patient_group"]) & mura_pg
    if bad:
        raise AssertionError(f"GRAZ/MURA patient_group collision: {sorted(bad)[:3]}")
    return f


def alt_labels(index: pd.DataFrame, frame: pd.DataFrame,
               score_labels) -> Dict[str, np.ndarray]:
    """
    Label vectors for the SAME rows, aligned to `frame` by uid.

    Scoring one set of logits against several label definitions is free -- same
    forward pass, different y. Retraining per label is only justified when the
    labels actually differ, and on GRAZ they differ on 1.9% of images.
    """
    m = index.set_index("uid")
    out = {}
    for lc in score_labels:
        if lc not in m.columns:
            print(f"  [alt_labels] skipping absent column {lc!r}")
            continue
        v = pd.to_numeric(m.loc[frame["uid"].to_numpy(), lc], errors="coerce")
        if v.isna().any():
            raise ValueError(f"{int(v.isna().sum())} NaN in {lc} for these rows")
        out[lc] = v.to_numpy().astype(int)
    if not out:
        raise ValueError("no usable scoring labels")
    return out


def with_descriptors(frame: pd.DataFrame):
    """Drop rows whose descriptors failed, so X and the frame stay aligned."""
    g = descriptors_for(frame)
    keep = g.notna().all(axis=1)
    fr = frame[keep].reset_index(drop=True)
    X = g[keep].reset_index(drop=True).to_numpy(dtype=float)
    if len(fr) != len(frame):
        print(f"  [descriptors] dropped {len(frame)-len(fr)} of {len(frame)} images",
              flush=True)
    return fr, X


def load_source(cp: Path, backbone: str, device):
    import torch
    if not cp.exists():
        raise FileNotFoundError(f"missing checkpoint {cp}")
    ck = torch.load(cp, map_location="cpu", weights_only=False)
    model = M.build_model(backbone)
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck


# ================================================================= precheck ==

def stage_precheck(cfg) -> None:
    index = load_index(cfg)
    print("=" * 78)
    print("STAGE B PRECHECK")
    print("=" * 78)

    print("\n-- index --")
    print(index.groupby(["dataset", "split"]).size().to_string())

    g = index[index.dataset == "graz"]
    if g.empty:
        print("\n!! NO GRAZ ROWS IN THE INDEX. Rebuild with:")
        print("   python mskdata.py build --mura ~/datasets/MURA-v1.1 "
              "--graz ~/datasets/GRAZPEDWRI-DX --out ~/diagnosis/data/diag_data")
        raise SystemExit(1)

    print(f"\n-- GRAZ -- {len(g)} images, {g.study.nunique()} studies, "
          f"{g.patient_group.nunique()} patients")
    print(g.groupby("split").agg(images=("uid", "size"),
                                 studies=("study", "nunique"),
                                 patients=("patient_group", "nunique")).to_string())
    for lc in GRAZ_LABELS:
        if lc not in g.columns:
            print(f"  !! missing label column {lc}")
            continue
        v = pd.to_numeric(g[lc], errors="coerce")
        print(f"  {lc:<16s} prevalence={v.mean():.4f}  "
              f"n_pos={int(v.sum())}  n_nan={int(v.isna().sum())}")
    if set(GRAZ_LABELS) <= set(g.columns):
        a = pd.to_numeric(g.label_abnormal, errors="coerce")
        f = pd.to_numeric(g.label_fracture, errors="coerce")
        dis = int((a != f).sum())
        print(f"  label_abnormal vs label_fracture disagree on {dis} images "
              f"({dis/len(g):.1%}) -- this is the CONCEPT SHIFT, not noise")
        if "has_metal" in g.columns:
            mm = pd.to_numeric(g.has_metal, errors="coerce").fillna(0)
            print(f"  of those, {int(((a != f) & (mm > 0)).sum())} carry metal "
                  f"-- the hardware class that drives prediction P1")

    # patient-disjointness of the fresh split
    for a in ("train", "valid", "test"):
        for b in ("train", "valid", "test"):
            if a < b:
                ov = (set(g[g.split == a].patient_group)
                      & set(g[g.split == b].patient_group))
                if ov:
                    print(f"  !! GRAZ {a}/{b} patient overlap: {len(ov)}")
    print("  patient-disjointness of GRAZ splits: checked")

    print("\n-- GRAZ scanners (Stage 5 runnability) --")
    if "meta_device_manufacturer" not in g.columns:
        print("  !! meta_device_manufacturer absent; STAGE 5 IS NOT RUNNABLE")
    else:
        t = (g.groupby(["meta_device_manufacturer", "split"])
               .agg(studies=("study", "nunique"),
                    patients=("patient_group", "nunique")).unstack(fill_value=0))
        print(t.to_string())
        tr = (g[g.split == "train"].groupby("meta_device_manufacturer")
              .study.nunique().sort_values(ascending=False))
        big = tr[tr >= SCANNER_MIN_TRAIN_STUDIES]
        print(f"\n  manufacturers with >= {SCANNER_MIN_TRAIN_STUDIES} train "
              f"studies: {list(big.index)}")
        if len(big) < 2:
            print("  >>> STAGE 5 NOT RUNNABLE. Per the pre-registration the "
                  "SITE claim is then DROPPED, not weakened.")
        else:
            print(f"  >>> Stage 5 runnable. Suggested pair: "
                  f"--scanner-a '{big.index[0]}' --scanner-b '{big.index[1]}'")

    for c in ("has_cast", "meta_age", "meta_projection"):
        if c in g.columns:
            v = g[c]
            print(f"\n-- {c} --")
            if c == "meta_age":
                n = pd.to_numeric(v, errors="coerce")
                print(f"  n={int(n.notna().sum())} min={n.min()} max={n.max()} "
                      f"median={n.median()}")
                print("  oldest quartile threshold: "
                      f"{n.quantile(0.75)} (growth-plate subgroup)")
            else:
                print(v.value_counts(dropna=False).head(8).to_string())

    print("\n-- checkpoints --")
    d_std = Path(cfg.ckpt_root) / P.PREP_SRC_DIR["standard"]
    miss = []
    for bb in cfg.backbones:
        for s in cfg.seeds:
            for a in MURA_ANATOMIES:
                if not ckpt_path(d_std, a, bb, s).exists():
                    miss.append(f"{a}/{bb}/s{s}")
    print(f"  source_models source checkpoints missing: {len(miss)}")
    for m in miss[:10]:
        print(f"    {m}")
    if miss:
        print("  (diag needs ALL seven anatomies; zeroshot needs XR_WRIST only)")

    for prep in ("p0p100", "maskpct"):
        d = Path(cfg.ckpt_root) / P.PREP_SRC_DIR[prep]
        have = sum(ckpt_path(d, "XR_WRIST", bb, s).exists()
                   for bb in cfg.backbones for s in cfg.seeds)
        print(f"  {prep} wrist sources present: {have} / "
              f"{len(cfg.backbones)*len(cfg.seeds)} (stage src_prep builds these)")
    print("\nprecheck complete")


# ====================================================== stage 1: diagonal ====

def stage_diag(cfg, prov) -> None:
    """
    THE DIAGONAL NULL. Each MURA source model on its OWN validation split.
    There is no domain shift here, so the projection should do nothing.

    If it does something systematically, the 42-pair matrix has a floor and
    every positive delta in the 42-pair analysis must be re-centred on it before being
    quoted. This is the cheapest way the headline result can be wrong, so it
    runs first and is reported whichever way it lands.
    """
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir) / "appearance_diagonal.csv"
    res = CsvAppender(out, prov)
    done = _resume(out, ["backbone", "seed", "anatomy"])
    index = load_index(cfg)
    ck_dir = Path(cfg.ckpt_root) / P.PREP_SRC_DIR["standard"]

    frames, Xs = {}, {}
    for a in MURA_ANATOMIES:
        fr = M.frame_for(index, "mura", a, "valid").reset_index(drop=True)
        frames[a], Xs[a] = with_descriptors(fr)
        print(f"[diag] {a:<12s} {len(frames[a])} images, "
              f"{frames[a].study.nunique()} studies", flush=True)

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            rows = []
            for a in MURA_ANATOMIES:
                if (bb, seed, a) in done:
                    continue
                model, _ = load_source(ckpt_path(ck_dir, a, bb, seed), bb, device)
                fr, X = frames[a], Xs[a]
                pr = P.predict(model, fr, device, cfg.eval_batch_size,
                               cfg.num_workers, cfg.img_size, P.load_standard)
                z, y, st = pr["logit"], pr["label"], pr["study"]
                cell, resid = projection_cell(z, y, st, X, ("diag", bb, seed, a))
                row = {"backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                       "seed": seed, "anatomy": a, "n_images": len(fr),
                       "n_studies": int(fr.study.nunique()),
                       "prevalence": float(y.mean())}
                row.update(cell)
                rows.append(row)
                save_dump(Path(cfg.out_dir) / "dumps" /
                          f"diag__{a}__{bb}__seed{seed}.npz",
                          fr, z, resid, y, st, X, row)
                print(f"  [diag] {M.BACKBONE_SHORT[bb]} s{seed} {a:<12s} "
                      f"auc={cell['zs_auc_study']:.4f} R2={cell['appearance_r2']:.3f} "
                      f"delta={cell['delta_study']:+.4f}", flush=True)
                del model
                torch.cuda.empty_cache()
            res.write(rows)
    print("[diag] complete", flush=True)


# =================================================== stage 2: p0p100 source ==

RETRAINED_PREPS = ("p0p100", "maskpct")


def stage_src_prep(cfg, prov) -> None:
    """
    MURA-wrist sources trained under the arms that change the pixel statistics
    (p0p100 and maskpct).

    Required, not optional. Evaluating a standard-trained encoder on p0p100 or
    maskpct pixels repeats the cropped-source mistake: the input is
    out-of-distribution for that encoder, and any drop is unattributable between
    "the arm removed a shortcut" and "the encoder has never seen this contrast".
    """
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    index = load_index(cfg)
    out = Path(cfg.out_dir) / "source_preprocessing.csv"
    res = CsvAppender(out, prov)

    tr = M.frame_for(index, "mura", "XR_WRIST", "train").reset_index(drop=True)
    va = M.frame_for(index, "mura", "XR_WRIST", "valid").reset_index(drop=True)
    print(f"[src_prep] MURA wrist train {len(tr)} images / "
          f"{tr.study.nunique()} studies, prev={tr.label.mean():.3f}", flush=True)

    preps = [p for p in cfg.preps if p in RETRAINED_PREPS]
    if not preps:
        print("[src_prep] nothing to do: no retrained-prep arm in --preps")
        return

    for prep in preps:
        fn = P.make_loader_fn(prep)
        ck_dir = Path(cfg.ckpt_root) / P.PREP_SRC_DIR[prep]
        ck_dir.mkdir(parents=True, exist_ok=True)
        for bb in cfg.backbones:
            for seed in cfg.seeds:
                cp = ckpt_path(ck_dir, "XR_WRIST", bb, seed)
                if cp.exists():
                    print(f"  [skip] {cp.name}")
                    continue
                print(f"  [train] {M.BACKBONE_SHORT[bb]} seed{seed} {prep}",
                      flush=True)
                t0 = time.time()
                model, hist = P.train(tr, bb, seed, device, fn,
                                      epochs=cfg.epochs,
                                      batch_size=cfg.batch_size, lr=cfg.lr,
                                      num_workers=cfg.num_workers,
                                      img_size=cfg.img_size)
                torch.save({"state_dict": model.state_dict(), "backbone": bb,
                            "source": "XR_WRIST", "seed": seed, "prep": prep,
                            "epochs": cfg.epochs, "label_col": "label_abnormal",
                            "source_prev": float(tr.label.mean()),
                            "history": hist, "provenance": prov}, cp)
                pr = P.predict(model, va, device, cfg.eval_batch_size,
                               cfg.num_workers, cfg.img_size, fn)
                s = score(pr["logit"], pr["label"], pr["study"])
                res.write({"backbone": bb, "seed": seed, "prep": prep,
                           "n_train_images": len(tr),
                           "n_train_studies": int(tr.study.nunique()),
                           "train_prev": float(tr.label.mean()),
                           "indomain_auc_study": s["auc_study"],
                           "indomain_auc_image": s["auc_image"],
                           "final_loss": hist[-1]["loss"],
                           "minutes": (time.time() - t0) / 60.0})
                # Sanity gate: if an arm's own in-domain wrist AUC collapses
                # relative to Stage A (~0.87), that arm has destroyed the image,
                # not removed a shortcut. p0p100 is the expected offender.
                print(f"    in-domain wrist AUC ({prep}) = {s['auc_study']:.4f}"
                      f"   [Stage A standard reference ~0.87]", flush=True)
                del model
                torch.cuda.empty_cache()
    print("[src_prep] complete", flush=True)


# ==================================================== stage 3: GRAZ ceiling ==

def graz_ckpt(ckpt_root: Path, label_col: str, prep: str, backbone: str,
              seed: int) -> Path:
    return (Path(ckpt_root) / "graz_ceiling_models" /
            f"graz_{label_col}__{prep}__{backbone}__seed{seed}.pt")


def stage_ceiling(cfg, prov) -> None:
    """
    GRAZ in-domain ceiling: train on GRAZ train, evaluate on GRAZ valid.
    One TRAINING per (prep, backbone, seed); each is scored against every label
    in --score-labels from the same forward pass.

    The original design trained a separate arm per label to test prediction P1.
    Precheck retired that: the two GRAZ labels agree on 98.1% of images, and
    appearance_r2 does not involve the label at all (it is the R^2 of the logit
    on the descriptors), so the arms would have differed only through two
    near-identical models. P1 was replaced by P1' in graz_label_audit.py,
    which tests whether CAST -- present on 28.4% of GRAZ, against metal's
    handful -- carries the appearance-label relationship.

    NOTE the standing caveat from Stage A: a "ceiling" is not an upper bound --
    fine-tuning from a source encoder has already exceeded one. It is a
    reference point, and raw delta-AUC is the primary metric.
    """
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    index = load_index(cfg)
    out = Path(cfg.out_dir) / "graz_ceiling.csv"
    res = CsvAppender(out, prov)
    done = _resume(out, ["backbone", "seed", "prep", "score_label"])
    ref = P.get_ref_cdf(index, Path(cfg.cache_dir)) if "histmatch" in cfg.preps else None

    label_col = cfg.train_label
    tr = graz_frame(index, "train", label_col)
    va = graz_frame(index, "valid", label_col)
    va, X = with_descriptors(va)
    alt = alt_labels(index, va, cfg.score_labels)
    print(f"\n[ceiling] train on {label_col}: {len(tr)}/{tr.study.nunique()} "
          f"prev={tr.label.mean():.4f} | valid {len(va)}/{va.study.nunique()} "
          f"prev={va.label.mean():.4f} | scoring against {list(alt)}", flush=True)
    # (single training label; the old per-label loop collapsed to one pass)
    if True:
        for prep in cfg.preps:
            fn = P.make_loader_fn(prep, ref)
            for bb in cfg.backbones:
                for seed in cfg.seeds:
                    if all((bb, seed, prep, sl) in done for sl in alt):
                        continue
                    cp = graz_ckpt(cfg.ckpt_root, label_col, prep, bb, seed)
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    t0 = time.time()
                    if cp.exists():
                        print(f"  [load] {cp.name}", flush=True)
                        model, _ = load_source(cp, bb, device)
                        hist = [{"loss": float("nan")}]
                    else:
                        print(f"  [train] {M.BACKBONE_SHORT[bb]} s{seed} {prep} "
                              f"{label_col}", flush=True)
                        model, hist = P.train(tr, bb, seed, device, fn,
                                              epochs=cfg.epochs,
                                              batch_size=cfg.batch_size,
                                              lr=cfg.lr,
                                              num_workers=cfg.num_workers,
                                              img_size=cfg.img_size)
                        torch.save({"state_dict": model.state_dict(),
                                    "backbone": bb, "source": "GRAZ",
                                    "seed": seed, "prep": prep,
                                    "label_col": label_col,
                                    "source_prev": float(tr.label.mean()),
                                    "history": hist, "provenance": prov}, cp)
                    pr = P.predict(model, va, device, cfg.eval_batch_size,
                                   cfg.num_workers, cfg.img_size, fn)
                    z, st = pr["logit"], pr["study"]
                    pdesc = prep_descriptor_summary(va, fn)
                    for sl, y in alt.items():
                        if (bb, seed, prep, sl) in done:
                            continue
                        cell, resid = projection_cell(
                            z, y, st, X, ("ceiling", bb, seed, prep, sl))
                        row = {"backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                               "seed": seed, "prep": prep,
                               "train_label": label_col, "score_label": sl,
                               "n_train_images": len(tr),
                               "n_train_studies": int(tr.study.nunique()),
                               "train_prev": float(tr.label.mean()),
                               "n_images": len(va),
                               "n_studies": int(va.study.nunique()),
                               "target_prev": float(y.mean()),
                               "final_loss": float(hist[-1]["loss"]),
                               "minutes": (time.time() - t0) / 60.0}
                        row.update(cell)
                        row.update(pdesc)
                        res.write(row)
                        save_dump(Path(cfg.out_dir) / "dumps" /
                                  f"ceiling__{sl}__{prep}__{bb}__seed{seed}.npz",
                                  va, z, resid, y, st, X, row)
                        print(f"    ceiling [{sl}] AUC={cell['zs_auc_study']:.4f} "
                              f"R2={cell['appearance_r2']:.3f} "
                              f"delta={cell['delta_study']:+.4f}", flush=True)
                    del model
                    torch.cuda.empty_cache()
    print("[ceiling] complete", flush=True)


# ==================================================== stage 4: the dataset ===

def stage_zeroshot(cfg, prov) -> None:
    """
    MURA-wrist -> GRAZ-wrist, with the appearance projection applied.

    This is a DATASET gap. Site, age, bit depth, label definition and prevalence
    all move together. Do not call it a site gap anywhere -- Stage 5 is the site
    control.

    The source checkpoint depends on the arm: standard and histmatch use the
    existing source_models wrist source (histmatch moves the TARGET onto the source's
    statistics, so retraining the source would defeat it); p0p100 uses its own.
    """
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    index = load_index(cfg)
    out = Path(cfg.out_dir) / "graz_zeroshot.csv"
    res = CsvAppender(out, prov)
    done = _resume(out, ["backbone", "seed", "prep", "score_label"])
    ref = P.get_ref_cdf(index, Path(cfg.cache_dir)) if "histmatch" in cfg.preps else None

    va = graz_frame(index, "valid", cfg.train_label)
    va, X = with_descriptors(va)
    alt = alt_labels(index, va, cfg.score_labels)
    print(f"\n[zeroshot] GRAZ valid {len(va)} images / {va.study.nunique()} "
          f"studies | scoring against {list(alt)}", flush=True)
    if True:
        for prep in cfg.preps:
            fn = P.make_loader_fn(prep, ref)
            ck_dir = Path(cfg.ckpt_root) / P.PREP_SRC_DIR[prep]
            for bb in cfg.backbones:
                for seed in cfg.seeds:
                    if all((bb, seed, prep, sl) in done for sl in alt):
                        continue
                    cp = ckpt_path(ck_dir, "XR_WRIST", bb, seed)
                    model, ck = load_source(cp, bb, device)
                    pr = P.predict(model, va, device, cfg.eval_batch_size,
                                   cfg.num_workers, cfg.img_size, fn)
                    z, st = pr["logit"], pr["study"]
                    pdesc = prep_descriptor_summary(va, fn)
                    for sl, y in alt.items():
                        if (bb, seed, prep, sl) in done:
                            continue
                        cell, resid = projection_cell(
                            z, y, st, X, ("zeroshot", bb, seed, prep, sl))
                        row = {"backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                               "seed": seed, "prep": prep, "score_label": sl,
                               "source": "MURA_XR_WRIST", "source_ckpt": str(cp),
                               "source_prev": float(ck.get("source_prev", np.nan)),
                               "n_images": len(va),
                               "n_studies": int(va.study.nunique()),
                               "target_prev": float(y.mean())}
                        row.update(cell)
                        row.update(pdesc)
                        res.write(row)
                        save_dump(Path(cfg.out_dir) / "dumps" /
                                  f"zeroshot__{sl}__{prep}__{bb}__seed{seed}.npz",
                                  va, z, resid, y, st, X, row)
                        print(f"  [zs] {M.BACKBONE_SHORT[bb]} s{seed} {prep:<9s} "
                              f"{sl:<15s} auc={cell['zs_auc_study']:.4f} "
                              f"R2={cell['appearance_r2']:.3f} "
                              f"delta={cell['delta_study']:+.4f}", flush=True)
                    del model
                    torch.cuda.empty_cache()
    print("[zeroshot] complete", flush=True)


# ====================================================== stage 5: the site ====

def stage_scanner(cfg, prov) -> None:
    """
    THE SITE CONTROL. Train on GRAZ images from one device_manufacturer,
    evaluate on GRAZ valid images from another.

    Anatomy, age distribution, label definition, bit depth and preprocessing are
    all held constant. Only the acquisition device moves. This is the only place
    in the whole project where "site" is the single moving variable, and it is
    therefore the ONLY result that licenses a claim about site.

    Both directions are run. The in-domain arm (train A, eval A-valid) is the
    reference for the same model, so the gap is within-model and not against a
    separately trained ceiling.
    """
    import torch
    if not cfg.scanner_a or not cfg.scanner_b:
        raise SystemExit("stage scanner needs --scanner-a and --scanner-b; "
                         "run --stage precheck to see the manufacturers")
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    index = load_index(cfg)
    out = Path(cfg.out_dir) / "graz_scanner_control.csv"
    res = CsvAppender(out, prov)
    done = _resume(out, ["backbone", "seed", "label_col", "train_mfr", "eval_mfr"])
    col = "meta_device_manufacturer"

    for label_col in [cfg.train_label]:
        tr_all = graz_frame(index, "train", label_col)
        va_all = graz_frame(index, "valid", label_col)
        if col not in tr_all.columns:
            raise SystemExit(f"{col} not in the index; Stage 5 is not runnable, "
                             "and per the pre-registration the site claim is "
                             "then DROPPED rather than weakened")

        pools = {}
        for m in (cfg.scanner_a, cfg.scanner_b):
            t = tr_all[tr_all[col].astype(str) == m].reset_index(drop=True)
            v = va_all[va_all[col].astype(str) == m].reset_index(drop=True)
            n_tr = int(t.study.nunique())
            print(f"[scanner] {label_col} {m!r}: train {len(t)}/{n_tr} studies "
                  f"prev={t.label.mean() if len(t) else float('nan'):.4f} | "
                  f"valid {len(v)}/{v.study.nunique()} studies", flush=True)
            if n_tr < SCANNER_MIN_TRAIN_STUDIES:
                raise SystemExit(
                    f"{m!r} has {n_tr} train studies, below the pre-registered "
                    f"{SCANNER_MIN_TRAIN_STUDIES}. Stage 5 is NOT RUNNABLE and "
                    "the site claim is dropped, not weakened.")
            vv, XX = with_descriptors(v)
            pools[m] = {"train": t, "valid": vv, "X": XX}

        for train_mfr in (cfg.scanner_a, cfg.scanner_b):
            for bb in cfg.backbones:
                for seed in cfg.seeds:
                    todo = [m for m in (cfg.scanner_a, cfg.scanner_b)
                            if (bb, seed, label_col, train_mfr, m) not in done]
                    if not todo:
                        continue
                    cp = (Path(cfg.ckpt_root) / "graz_scanner_models" /
                          f"graz_{label_col}__mfr_{_slug(train_mfr)}__"
                          f"{bb}__seed{seed}.pt")
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    if cp.exists():
                        model, _ = load_source(cp, bb, device)
                        hist = [{"loss": float("nan")}]
                    else:
                        print(f"  [train] {M.BACKBONE_SHORT[bb]} s{seed} "
                              f"on {train_mfr!r}", flush=True)
                        model, hist = P.train(
                            pools[train_mfr]["train"], bb, seed, device,
                            P.load_standard, epochs=cfg.epochs,
                            batch_size=cfg.batch_size, lr=cfg.lr,
                            num_workers=cfg.num_workers, img_size=cfg.img_size)
                        torch.save({"state_dict": model.state_dict(),
                                    "backbone": bb, "source": f"GRAZ:{train_mfr}",
                                    "seed": seed, "prep": "standard",
                                    "label_col": label_col,
                                    "history": hist, "provenance": prov}, cp)
                    rows = []
                    for eval_mfr in todo:
                        fr = pools[eval_mfr]["valid"]
                        X = pools[eval_mfr]["X"]
                        pr = P.predict(model, fr, device, cfg.eval_batch_size,
                                       cfg.num_workers, cfg.img_size,
                                       P.load_standard)
                        z, y, st = pr["logit"], pr["label"], pr["study"]
                        cell, resid = projection_cell(
                            z, y, st, X,
                            ("scanner", bb, seed, label_col, train_mfr, eval_mfr))
                        row = {"backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                               "seed": seed, "label_col": label_col,
                               "train_mfr": train_mfr, "eval_mfr": eval_mfr,
                               "in_domain": int(train_mfr == eval_mfr),
                               "n_train_images": len(pools[train_mfr]["train"]),
                               "n_train_studies":
                                   int(pools[train_mfr]["train"].study.nunique()),
                               "train_prev":
                                   float(pools[train_mfr]["train"].label.mean()),
                               "n_images": len(fr),
                               "n_studies": int(fr.study.nunique()),
                               "target_prev": float(y.mean()),
                               "final_loss": float(hist[-1]["loss"])}
                        row.update(cell)
                        rows.append(row)
                        save_dump(Path(cfg.out_dir) / "dumps" /
                                  f"scanner__{label_col}__{_slug(train_mfr)}__to__"
                                  f"{_slug(eval_mfr)}__{bb}__seed{seed}.npz",
                                  fr, z, resid, y, st, X, row)
                        print(f"    {train_mfr!r} -> {eval_mfr!r} "
                              f"auc={cell['zs_auc_study']:.4f} "
                              f"delta={cell['delta_study']:+.4f}", flush=True)
                    res.write(rows)
                    del model
                    torch.cuda.empty_cache()
    print("[scanner] complete", flush=True)


# =================================================================== plumbing =

def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))[:40]


def _resume(path: Path, keys: List[str]) -> set:
    if not (path.exists() and path.stat().st_size > 0):
        return set()
    prev = pd.read_csv(path, comment="#", low_memory=False)
    if not set(keys) <= set(prev.columns):
        return set()
    d = set(map(tuple, prev[keys].values.tolist()))
    print(f"[resume] {len(d)} cells already in {path.name}")
    return d


def selftest() -> None:
    ok = True

    def ck(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    # crc32 seeding is stable across processes; hash() is not
    ck("cell_rng is deterministic",
       cell_rng("a", 1).randint(0, 10 ** 9) == cell_rng("a", 1).randint(0, 10 ** 9))
    ck("cell_rng separates cells",
       cell_rng("a", 1).randint(0, 10 ** 9) != cell_rng("a", 2).randint(0, 10 ** 9))

    # Projection checks need shortcut_reliance, which pulls in torch via
    # layerwise_unfreezing. On a login node without the GPU stack these are SKIPPED,
    # not faked with a local copy -- a local copy is exactly how an instrument
    # silently diverges from the one that produced the published matrix.
    try:
        _sr()
        have_sr = True
    except Exception as e:
        have_sr = False
        print(f"  [SKIP] projection checks: {type(e).__name__}: {e}")
        print("         RUN `--stage selftest` ON THE SERVER before trusting "
              "any Stage B number.")

    if have_sr:
        rng = np.random.default_rng(0)
        n = 600
        z = rng.normal(size=n)
        X = rng.normal(size=(n, 5))
        resid, r2 = project_out(z, X)
        ck("projecting out 5 random directions explains ~5/n of the variance",
           r2 < 0.05, f"(R2={r2:.4f})")
        ck("residual is orthogonal to the descriptors",
           float(np.abs(np.corrcoef(resid, X[:, 0])[0, 1])) < 1e-8)

        # a planted appearance component must be recovered
        z2 = 2.0 * X[:, 0] + 0.1 * rng.normal(size=n)
        _, r2b = project_out(z2, X)
        ck("planted component is recovered", r2b > 0.95, f"(R2={r2b:.4f})")

        # study AUC must be sigmoid-FIRST-then-aggregate. Reversing it shifts
        # study AUC by ~0.013 with no error raised.
        y = (rng.random(n) < 0.4).astype(int)
        st = np.array([f"s{i//3}" for i in range(n)])
        sc = score(z, y, st)
        ck("score() returns both image and study AUC",
           {"auc_image", "auc_study"} <= set(sc))

    ck("preps registered",
       set(P.PREPS) == {"standard", "p0p100", "maskpct", "histmatch"},
       f"({sorted(P.PREPS)})")
    ck("every retrained prep has its own source dir",
       all(P.PREP_SRC_DIR[q] != "source_models" for q in RETRAINED_PREPS))
    ck("slug is filesystem-safe", "/" not in _slug("A/B Co."))
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    raise SystemExit(0 if ok else 1)


STAGES = {"precheck": stage_precheck, "diag": stage_diag,
          "src_prep": stage_src_prep, "ceiling": stage_ceiling,
          "zeroshot": stage_zeroshot, "scanner": stage_scanner}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["selftest"])
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-root", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--cache-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES,
                    choices=list(M.BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(M.SEEDS))
    ap.add_argument("--preps", nargs="+", default=list(P.PREPS), choices=list(P.PREPS))
    ap.add_argument("--train-label", default=TRAIN_LABEL, choices=list(GRAZ_LABELS),
                    help="the label GRAZ ceilings are TRAINED on; one only, "
                         "because the two GRAZ labels agree on 98.1%% of images")
    ap.add_argument("--score-labels", nargs="+", default=list(GRAZ_LABELS),
                    help="label definitions the SAME logits are scored against; "
                         "free, one forward pass")
    ap.add_argument("--scanner-a", default=None)
    ap.add_argument("--scanner-b", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    cfg = ap.parse_args()

    if cfg.stage == "selftest":
        selftest()

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    if cfg.stage != "precheck":
        M.raise_fd_limit()
        M.configure_sharing()
    prov = M.provenance({"script": "dataset_shift.py v1", "stage": cfg.stage,
                         "backbones": list(cfg.backbones), "seeds": list(cfg.seeds),
                         "preps": list(cfg.preps),
                         "train_label": cfg.train_label,
                         "score_labels": list(cfg.score_labels),
                         "features": FEATURES, "n_perm": N_PERM,
                         "scanner_a": cfg.scanner_a, "scanner_b": cfg.scanner_b,
                         "prereg": "STAGE_B_PREREGISTRATION.md"})
    fn = STAGES[cfg.stage]
    if cfg.stage == "precheck":
        fn(cfg)
    else:
        fn(cfg, prov)


if __name__ == "__main__":
    main()
