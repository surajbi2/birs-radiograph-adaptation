"""Arm B: fit a source readout on frozen wrist features for each seed.

The arm-B transferred model is an l2-penalised logistic readout on the frozen
MURA wrist training features. One readout is fitted per seed, on a bootstrap of
wrist training patients, so the five arm-B instances differ as the five
fine-tuned seeds do.

    C          oracle_readout_cv.choose_c_nested (patient-grouped
               cross-validation over the C grid)
    fit        readout_shrinkage.raw_space_readout (standardised fit returned
               in raw feature space)
    bootstrap  patients drawn with replacement, keyed on the seed only, so every
               encoder gets the same patients for a given seed; a patient drawn
               k times stays one cross-validation group

Output: arm_b/features/<encoder>__s<seed>__<target>.npz for the nine targets and
the wrist pools, with the cache keys above; features are copied unchanged from
the s0 file and zva = fva @ w_src + b_src.

Usage:
    python arm_b_readout.py --encoders skelex,rad-dino,medsiglip --jobs 15

CPU only.
"""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path.home() / "diagnosis" / "code"
sys.path.insert(0, str(CODE))
FA = Path("/path/to/foundation_arm_workdir")
import mskmodel as M                                   # noqa: E402  (read-only)
from dataset_shift import cell_rng                       # noqa: E402  (read-only)
from oracle_readout_cv import choose_c_nested      # noqa: E402  (read-only)
from readout_shrinkage import C_GRID, raw_space_readout, verify_source_head  # noqa: E402
from l2sp_comparator import pools, cache_path           # noqa: E402  (read-only)
from shortcut_reliance import score                    # noqa: E402  (read-only)

SOURCE_T = "XR_WRIST"
LABEL_COL = "label_abnormal"
SRC_SEED_TAG = 0                                       # extract_features arm-B tag
BOOT_TAG = "fa_armb_bootstrap"


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_rows(frame: pd.DataFrame, seed: int):
    """Row indices of a patient bootstrap, and each row's patient group."""
    pats = np.array(sorted(frame.patient_group.astype(str).unique()), dtype=object)
    draw = cell_rng(BOOT_TAG, seed).choice(len(pats), size=len(pats), replace=True)
    by_pat = frame.groupby(frame.patient_group.astype(str)).indices
    idx = np.concatenate([by_pat[pats[i]] for i in draw])
    return idx, frame.patient_group.astype(str).to_numpy()[idx], int(len(np.unique(draw)))


def fit_one(args):
    """One (encoder, seed): readout on the bootstrap, then all 10 unit files."""
    cdir, enc, seed, wrist_tr, wrist_va, P_ = args
    t0 = time.time()
    src_p = cache_path(cdir, enc, SRC_SEED_TAG, SOURCE_T)
    z = np.load(src_p, allow_pickle=False)
    if not (np.array_equal(z["tr_uid"], wrist_tr.uid.to_numpy().astype(str))
            and np.array_equal(z["va_uid"], wrist_va.uid.to_numpy().astype(str))):
        raise SystemExit(f"!! {src_p.name}: rows do not match the wrist pools")
    F = z["ftr"].astype(np.float64)
    idx, grp, n_unique = bootstrap_rows(wrist_tr, seed)
    y = wrist_tr.label.to_numpy()[idx]
    C, cv_auc = choose_c_nested(F[idx], y, grp, C_GRID)
    w, b = raw_space_readout(F[idx], y, C)
    if w is None:
        raise SystemExit(f"!! {enc}/s{seed}: bootstrap sample has one class")
    zv = z["fva"].astype(np.float64) @ w + b
    src_auc = score(zv, z["yva"], wrist_va.study.to_numpy())
    print(f"  [readout] {enc:<10s} s{seed}: C={C:g} cv_auc={cv_auc:.4f} boot rows {len(idx)} "
          f"patients {n_unique}/{wrist_tr.patient_group.nunique()} wrist-valid study AUC "
          f"{src_auc['auc_study']:.4f}", flush=True)

    meta = dict(readout_C=np.array(C), readout_cv_auc=np.array(cv_auc),
                bootstrap_key=np.array(f"{BOOT_TAG}|{seed}"), n_boot_rows=np.array(len(idx)),
                n_boot_patients_unique=np.array(n_unique),
                wrist_valid_auc_study=np.array(src_auc["auc_study"]))
    out = {"encoder": enc, "seed": seed, "C": C, "cv_auc": cv_auc, "n_boot_rows": int(len(idx)),
           "n_boot_patients_unique": n_unique, "wrist_valid": src_auc, "zs": {}}
    for t in [SOURCE_T] + list(P_):
        sp = cache_path(cdir, enc, SRC_SEED_TAG, t)
        s = np.load(sp, allow_pickle=False)
        keep = {k: s[k] for k in s.files if k not in ("seed_tag",)}
        fva = s["fva"].astype(np.float64)
        zva = fva @ w + b
        st = (wrist_va if t == SOURCE_T else P_[t][1]).study.to_numpy()
        zs = score(zva, s["yva"], st)["auc_study"]
        if t != SOURCE_T:
            verify_source_head(fva, zva, s["yva"], st, w, b, zs, f"armB/{enc}/s{seed}/{t}")
        keep.update(zva=zva, w_src=w, b_src=np.array(b), zs=np.array(zs), seed_tag=np.array(seed),
                    features_from=np.array(sha256(sp)), **meta)
        p = cache_path(cdir, enc, seed, t)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **keep)
        os.replace(tmp, p)
        out["zs"][t] = float(zs)
    out["minutes"] = (time.time() - t0) / 60
    return out


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=FA / "features" / "arm_b")
    ap.add_argument("--encoders", default="skelex,rad-dino,medsiglip")
    ap.add_argument("--seeds", default=",".join(str(s) for s in M.SEEDS))
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--jobs", type=int, default=15)
    cfg = ap.parse_args()
    encs = [e.strip() for e in cfg.encoders.split(",") if e.strip()]
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    P_ = pools(index)
    wrist_tr = M.frame_for(index, "mura", SOURCE_T, "train", LABEL_COL).reset_index(drop=True)
    wrist_va = M.frame_for(index, "mura", SOURCE_T, "valid", LABEL_COL).reset_index(drop=True)
    todo = []
    for e in encs:
        for t in [SOURCE_T] + list(P_):
            if not cache_path(cfg.cache_dir, e, SRC_SEED_TAG, t).exists():
                raise SystemExit(f"!! missing arm-B features {e} s{SRC_SEED_TAG} {t}; run extract_features first")
        for s in seeds:
            if all(cache_path(cfg.cache_dir, e, s, t).exists() for t in [SOURCE_T] + list(P_)):
                print(f"[skip] {e} s{s}: all files exist")
                continue
            todo.append((cfg.cache_dir, e, s, wrist_tr, wrist_va, P_))
    from multiprocessing import get_context
    with get_context("fork").Pool(min(cfg.jobs, max(1, len(todo))), initializer=_init_worker) as pool:
        reports = list(pool.imap_unordered(fit_one, todo))
    rp = cfg.cache_dir / "reports" / "armb_readouts.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(rp.read_text()) if rp.exists() else []
    rp.write_text(json.dumps(old + reports, indent=2, default=float))
    print(f"wrote {len(reports)} readouts -> {rp}")


if __name__ == "__main__":
    main()
