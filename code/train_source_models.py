#!/usr/bin/env python3
"""Stage A: train source models and score transfer between MURA anatomies.

Trains one source model per (anatomy, backbone, seed) on MURA and evaluates it
on every other anatomy, writing raw per-image score dumps and per-arm metrics.
Arms per source -> target cell:

    ceiling     a model trained and evaluated on the target itself
    zeroshot    the source model applied unchanged
    adabn       BatchNorm statistics re-estimated on target images (label-free)
    prior       decision threshold corrected for target prevalence
    oracle      decision threshold chosen on target labels (upper bound)
    probe@n     linear readout refitted on n labelled target studies

Thresholds cannot change AUC; only adabn and probe@n can. source_model_tables.py
derives every table from the files written here.

Outputs (under --out-dir):
    source_training_log.csv, anatomy_transfer_arms.csv, anatomy_transfer_probes.csv,
    adabn_receipts.csv and per-cell .npz score dumps

Usage:
    python train_source_models.py --selftest
    python train_source_models.py --stage all --gpu 0 --sources XR_WRIST

Requires a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mskdata          # noqa: E402
import mskmodel as M    # noqa: E402

MURA_ANATOMIES = mskdata.MURA_ANATOMIES
PROBE_NS = (5, 10, 25, 50, 100, 200, 500)     # labelled TARGET STUDIES
PROBE_REPS = 10
PROBE_POOL_CAP_STUDIES = 1200                 # bounds feature-extraction cost
ADABN_BATCHES = 100


# ============================================================== csv writer ===

class CsvAppender:
    """O(1) append with a provenance header written once. Survives kill -9."""

    def __init__(self, path: Path, prov: Dict):
        self.path = Path(path)
        self.prov = prov
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = self.path.exists() and self.path.stat().st_size > 0
        self.cols: Optional[List[str]] = None
        if self.started:
            try:
                self.cols = list(pd.read_csv(self.path, comment="#", nrows=0).columns)
            except Exception:
                self.started = False

    def write(self, rows) -> None:
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            return
        df = pd.DataFrame(rows)
        if self.cols is None:
            self.cols = list(df.columns)
        extra = [c for c in df.columns if c not in self.cols]
        if extra:
            raise ValueError(f"{self.path.name}: new columns {extra} mid-run; "
                             "schema must be stable")
        df = df.reindex(columns=self.cols)
        if not self.started:
            with open(self.path, "w") as fh:
                for line in json.dumps({"provenance": self.prov},
                                       default=str).splitlines():
                    fh.write("# " + line + "\n")
                df.to_csv(fh, index=False, header=True)
            self.started = True
        else:
            with open(self.path, "a") as fh:
                df.to_csv(fh, index=False, header=False)


def cell_rng(*parts) -> np.random.Generator:
    """Deterministic across processes. Python's hash() is salted; sha256 isn't."""
    d = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(d[:8], "big") % (2 ** 32))


# ================================================================= helpers ===

def ckpt_path(ckpt_dir: Path, anatomy: str, backbone: str, seed: int) -> Path:
    return Path(ckpt_dir) / f"src_{anatomy}__{backbone}__seed{seed}.pt"


def dump_path(out_dir: Path, source: str, target: str, backbone: str,
              seed: int, condition: str) -> Path:
    return (Path(out_dir) / "dumps" /
            f"{source}__{target}__{backbone}__seed{seed}__{condition}.npz")


def eval_block(logit: np.ndarray, label: np.ndarray, study: np.ndarray,
               source_prev: float) -> Dict[str, float]:
    """Every metric derived from one set of logits. No model touched here."""
    p = M.sigmoid(logit)
    _, sp, sy = M.aggregate_studies(p, label, study)
    tgt_prev = float(label.mean())

    m50 = M.threshold_metrics(p, label, 0.5)
    t_prior = M.prior_corrected_threshold(p, source_prev, tgt_prev)
    mpri = M.threshold_metrics(p, label, t_prior)
    t_orc = M.oracle_threshold(p, label)
    morc = M.threshold_metrics(p, label, t_orc)

    return {
        "auc_image": M.safe_auc(label, p),
        "auc_study": M.safe_auc(sy, sp),
        "ece": M.expected_calibration_error(p, label),
        "n_images": int(len(label)), "n_studies": int(len(sy)),
        "target_prev": tgt_prev, "source_prev": float(source_prev),
        "bacc_50": m50["balanced_acc"], "sens_50": m50["sensitivity"],
        "spec_50": m50["specificity"],
        "thr_prior": t_prior, "bacc_prior": mpri["balanced_acc"],
        "sens_prior": mpri["sensitivity"], "spec_prior": mpri["specificity"],
        "thr_oracle": t_orc, "bacc_oracle": morc["balanced_acc"],
        "sens_oracle": morc["sensitivity"], "spec_oracle": morc["specificity"],
    }


# ============================================================ STAGE A1 ======

def stage_train(index: pd.DataFrame, cfg, prov: Dict) -> None:
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    log = CsvAppender(Path(cfg.out_dir) / "source_training_log.csv", prov)

    for anat in cfg.sources:
        frame = M.frame_for(index, "mura", anat, "train", cfg.label_col)
        for bb in cfg.backbones:
            for seed in cfg.seeds:
                cp = ckpt_path(cfg.ckpt_dir, anat, bb, seed)
                if cp.exists():
                    print(f"[skip] {cp.name}", flush=True)
                    continue
                print(f"[train] {anat} | {M.BACKBONE_SHORT[bb]} | seed {seed} | "
                      f"{len(frame)} imgs | {frame.study.nunique()} studies "
                      f"| prev={frame.label.mean():.3f}", flush=True)
                t0 = time.time()
                model, hist = M.train_model(
                    frame, bb, seed, device, epochs=cfg.epochs,
                    batch_size=cfg.batch_size, lr=cfg.lr,
                    num_workers=cfg.num_workers, img_size=cfg.img_size)
                cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "backbone": bb,
                            "source": anat, "seed": seed, "epochs": cfg.epochs,
                            "label_col": cfg.label_col,
                            "source_prev": float(frame.label.mean()),
                            "history": hist, "provenance": prov}, cp)
                log.write({"source": anat, "backbone": bb, "seed": seed,
                           "n_images": len(frame),
                           "n_studies": int(frame.study.nunique()),
                           "n_patients": int(frame.patient_group.nunique()),
                           "source_prev": float(frame.label.mean()),
                           "epochs": cfg.epochs,
                           "final_loss": hist[-1]["loss"],
                           "minutes": round((time.time() - t0) / 60, 2),
                           "checkpoint": str(cp)})
                del model
                torch.cuda.empty_cache()
    print("[stage A1] complete", flush=True)


# ============================================================ STAGE A2 ======

def stage_eval(index: pd.DataFrame, cfg, prov: Dict) -> None:
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    arms = CsvAppender(Path(cfg.out_dir) / "anatomy_transfer_arms.csv", prov)
    probes = CsvAppender(Path(cfg.out_dir) / "anatomy_transfer_probes.csv", prov)
    recei = CsvAppender(Path(cfg.out_dir) / "adabn_receipts.csv", prov)

    done = set()
    ap = Path(cfg.out_dir) / "anatomy_transfer_arms.csv"
    if ap.exists():
        d = pd.read_csv(ap, comment="#")
        done = set(zip(d.source, d.target, d.backbone, d.seed, d.condition))

    for src in cfg.sources:
        for bb in cfg.backbones:
            for seed in cfg.seeds:
                cp = ckpt_path(cfg.ckpt_dir, src, bb, seed)
                if not cp.exists():
                    print(f"[warn] missing {cp.name}; skipping", flush=True)
                    continue
                blob = torch.load(cp, map_location="cpu", weights_only=False)
                src_prev = float(blob.get("source_prev", np.nan))

                for tgt in MURA_ANATOMIES:
                    key0 = (src, tgt, bb, seed, "ceiling" if src == tgt else "zeroshot")
                    key1 = (src, tgt, bb, seed, "adabn")
                    if key0 in done and (key1 in done or src == tgt):
                        continue

                    tv = M.frame_for(index, "mura", tgt, "valid", cfg.label_col)

                    # ---- fresh model instance every cell (AdaBN mutates BN) --
                    M.set_seed(seed)
                    model = M.build_model(bb).to(device)
                    model.load_state_dict(blob["state_dict"])

                    cond0 = "ceiling" if src == tgt else "zeroshot"
                    pred = M.predict(model, tv, device, cfg.eval_batch_size,
                                     cfg.num_workers, cfg.img_size,
                                     want_features=True)
                    feat_valid = pred.pop("feature")
                    M.save_dump(dump_path(cfg.out_dir, src, tgt, bb, seed, cond0),
                                pred, {"source": src, "target": tgt,
                                       "backbone": bb, "seed": seed,
                                       "condition": cond0, "source_prev": src_prev})
                    row = eval_block(pred["logit"], pred["label"],
                                     pred["study"], src_prev)
                    row.update(source=src, target=tgt, backbone=bb,
                               backbone_short=M.BACKBONE_SHORT[bb],
                               norm_family=M.NORM_FAMILY[bb], seed=seed,
                               condition=cond0, n_bn_modules=np.nan,
                               adabn_batches=np.nan, adabn_mean_drift=np.nan)
                    arms.write(row)
                    print(f"  [{cond0:9s}] {src}->{tgt} {M.BACKBONE_SHORT[bb]} "
                          f"s{seed} AUC_study={row['auc_study']:.3f}", flush=True)

                    if src == tgt:
                        del model
                        torch.cuda.empty_cache()
                        continue

                    # ---- clean target-train pool (leakage-controlled) --------
                    tt_all = M.frame_for(index, "mura", tgt, "train", cfg.label_col)
                    tt = mskdata.drop_source_patients(index, tt_all, src, "train")
                    M.assert_probe_pool_clean(tt, index, src, "train")
                    studies = tt.study.drop_duplicates().to_numpy()
                    if len(studies) > cfg.probe_pool_cap:
                        keep = cell_rng("pool", src, tgt, bb, seed).choice(
                            studies, cfg.probe_pool_cap, replace=False)
                        tt = tt[tt.study.isin(keep)].reset_index(drop=True)

                    # ---- probes: re-fit ONLY the readout ---------------------
                    ptr = M.predict(model, tt, device, cfg.eval_batch_size,
                                    cfg.num_workers, cfg.img_size,
                                    want_features=True)
                    feat_pool = ptr["feature"]
                    pool_studies = pd.Series(ptr["study"]).drop_duplicates().to_numpy()
                    rows = []
                    for n in cfg.probe_ns:
                        n_eff = min(n, len(pool_studies))
                        rng = cell_rng("probe", src, tgt, bb, seed, n)
                        for rep in range(cfg.probe_reps):
                            sel = rng.choice(pool_studies, n_eff, replace=False)
                            m = np.isin(ptr["study"], sel)
                            score = M.linear_probe(feat_pool[m], ptr["label"][m],
                                                   feat_valid)
                            _, sp, sy = M.aggregate_studies(
                                M.sigmoid(score), pred["label"], pred["study"])
                            rows.append({
                                "source": src, "target": tgt, "backbone": bb,
                                "norm_family": M.NORM_FAMILY[bb], "seed": seed,
                                "n_studies": int(n_eff), "requested_n": int(n),
                                "rep": rep,
                                "n_probe_images": int(m.sum()),
                                "probe_prev": float(ptr["label"][m].mean()),
                                "auc_image": M.safe_auc(pred["label"], score),
                                "auc_study": M.safe_auc(sy, sp),
                                "pool_studies_available": int(len(pool_studies)),
                                "pool_studies_dropped_leak":
                                    int(tt_all.study.nunique() - tt.study.nunique()),
                            })
                        if n_eff < n:
                            break
                    probes.write(rows)
                    del feat_pool, ptr

                    # ---- AdaBN: label-free, with a drift receipt -------------
                    try:
                        rec = M.adapt_bn(model, tt, device, cfg.adabn_batches,
                                         cfg.batch_size, cfg.num_workers,
                                         cfg.img_size)
                    except RuntimeError as e:
                        print(f"  [adabn] REFUSED: {e}", flush=True)
                        rec = {"n_bn_modules": np.nan, "n_batches": 0.0,
                               "mean_drift": np.nan, "var_drift": np.nan,
                               "applicable": np.nan}
                    rec.update(source=src, target=tgt, backbone=bb,
                               norm_family=M.NORM_FAMILY[bb], seed=seed)
                    recei.write(rec)

                    if rec.get("applicable") == 1.0:
                        pa = M.predict(model, tv, device, cfg.eval_batch_size,
                                       cfg.num_workers, cfg.img_size)
                        M.save_dump(dump_path(cfg.out_dir, src, tgt, bb, seed,
                                              "adabn"), pa,
                                    {"source": src, "target": tgt, "backbone": bb,
                                     "seed": seed, "condition": "adabn",
                                     "source_prev": src_prev, **rec})
                        r2 = eval_block(pa["logit"], pa["label"], pa["study"],
                                        src_prev)
                    else:
                        # no BatchNorm -> defined no-op, recorded as such
                        r2 = eval_block(pred["logit"], pred["label"],
                                        pred["study"], src_prev)
                    r2.update(source=src, target=tgt, backbone=bb,
                              backbone_short=M.BACKBONE_SHORT[bb],
                              norm_family=M.NORM_FAMILY[bb], seed=seed,
                              condition="adabn",
                              n_bn_modules=rec.get("n_bn_modules"),
                              adabn_batches=rec.get("n_batches"),
                              adabn_mean_drift=rec.get("mean_drift"))
                    arms.write(r2)
                    print(f"  [adabn    ] {src}->{tgt} {M.BACKBONE_SHORT[bb]} "
                          f"s{seed} AUC_study={r2['auc_study']:.3f} "
                          f"(drift={rec.get('mean_drift')})", flush=True)

                    del model
                    torch.cuda.empty_cache()
    print("[stage A2] complete", flush=True)


# ================================================================ selftest ===

def selftest() -> None:
    ok = True

    def ck(n, c, e=""):
        nonlocal ok
        print(f"  [{'PASS' if c else 'FAIL'}] {n} {e}")
        ok &= bool(c)

    import tempfile
    tmp = Path(tempfile.mkdtemp())

    # appender: schema stability + resume + provenance
    a = CsvAppender(tmp / "x.csv", {"p": 1})
    a.write([{"i": i, "v": i * 2} for i in range(500)])
    b = CsvAppender(tmp / "x.csv", {"p": 1})
    b.write({"i": 999, "v": 1})
    d = pd.read_csv(tmp / "x.csv", comment="#")
    ck("appender rows", len(d) == 501)
    ck("header once", open(tmp / "x.csv").read().count("i,v") == 1)
    ck("provenance kept", open(tmp / "x.csv").readline().startswith("#"))
    try:
        b.write({"i": 1, "v": 2, "z": 3}); ck("schema guard", False)
    except ValueError:
        ck("schema guard", True)

    # deterministic rng across processes
    ck("cell_rng deterministic",
       cell_rng("a", 1).integers(0, 10**6) == cell_rng("a", 1).integers(0, 10**6))
    ck("cell_rng distinct",
       cell_rng("a", 1).integers(0, 10**6) != cell_rng("a", 2).integers(0, 10**6))

    # eval_block on a known-good signal
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    z = y * 2.0 + rng.normal(0, 1, 800)
    st = np.array([f"s{i//2}" for i in range(800)])
    r = eval_block(z, y, st, 0.5)
    ck("eval_block auc sane", 0.7 < r["auc_image"] < 1.0, f"({r['auc_image']:.3f})")
    ck("oracle >= 0.5-threshold bacc", r["bacc_oracle"] >= r["bacc_50"] - 1e-12)
    ck("study count halves", r["n_studies"] == 400)
    ck("prior==0.5 when prevs equal",
       abs(eval_block(z, y, st, float(y.mean()))["thr_prior"] - 0.5) < 0.02)

    # path helpers
    ck("ckpt name", ckpt_path(tmp, "XR_WRIST", "resnet50", 42).name
       == "src_XR_WRIST__resnet50__seed42.pt")
    ck("dump name", dump_path(tmp, "XR_WRIST", "XR_HAND", "resnet50", 42,
                              "zeroshot").name
       == "XR_WRIST__XR_HAND__resnet50__seed42__zeroshot.npz")
    ck("probe grid sane", PROBE_NS[0] == 5 and PROBE_NS[-1] == 500)

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


# ==================================================================== main ===

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-dir", default=str(M.DEFAULT_CKPT / "source_models"))
    ap.add_argument("--out-dir", default=str(M.DEFAULT_RESULTS / "source_models"))
    ap.add_argument("--label-col", default="label_abnormal")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--sources", default=",".join(MURA_ANATOMIES))
    ap.add_argument("--backbones", default=",".join(M.BACKBONES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in M.SEEDS))
    ap.add_argument("--probe-ns", default=",".join(str(n) for n in PROBE_NS))
    ap.add_argument("--probe-reps", type=int, default=PROBE_REPS)
    ap.add_argument("--probe-pool-cap", type=int, default=PROBE_POOL_CAP_STUDIES)
    ap.add_argument("--adabn-batches", type=int, default=ADABN_BATCHES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    cfg = ap.parse_args()

    if cfg.selftest:
        selftest()

    cfg.sources = tuple(s.strip() for s in cfg.sources.split(",") if s.strip())
    cfg.backbones = tuple(b.strip() for b in cfg.backbones.split(",") if b.strip())
    cfg.seeds = tuple(int(s) for s in cfg.seeds.split(",") if s.strip())
    cfg.probe_ns = tuple(int(n) for n in cfg.probe_ns.split(",") if n.strip())
    for s in cfg.sources:
        if s not in MURA_ANATOMIES:
            raise SystemExit(f"unknown anatomy {s}; valid {MURA_ANATOMIES}")

    if cfg.smoke:
        cfg.epochs, cfg.seeds = 1, (42,)
        cfg.backbones = (cfg.backbones[0], "convnext_tiny")   # one BN, one LN
        cfg.probe_ns, cfg.probe_reps = (5, 10), 2
        cfg.probe_pool_cap, cfg.adabn_batches = 40, 5
        cfg.num_workers = 2
        cfg.out_dir = str(Path(cfg.out_dir).parent / "smoke_test")
        cfg.ckpt_dir = str(Path(cfg.ckpt_dir).parent / "smoke_test")

    fd = M.raise_fd_limit()
    sharing = M.configure_sharing()
    prov = M.provenance({"stage": cfg.stage, "label_col": cfg.label_col,
                         "probe_ns": cfg.probe_ns, "probe_reps": cfg.probe_reps,
                         "probe_pool_cap": cfg.probe_pool_cap,
                         "adabn_batches": cfg.adabn_batches,
                         "epochs": cfg.epochs, "smoke": cfg.smoke})

    print("=" * 78)
    print("STAGE A -- cross-anatomy failure localisation")
    print("=" * 78)
    for k, v in prov.items():
        print(f"  {k:<18s} {v}")
    print(f"  sources           {list(cfg.sources)}")
    print(f"  backbones         {[M.BACKBONE_SHORT[b] for b in cfg.backbones]}")
    print(f"  norm families     "
          f"{ {M.BACKBONE_SHORT[b]: M.NORM_FAMILY[b] for b in cfg.backbones} }")
    print(f"  seeds             {list(cfg.seeds)}")
    print(f"  fd limit          {fd}   torch sharing {sharing}")
    print("=" * 78 + "\n", flush=True)

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    if "patient_group" not in index.columns:
        raise SystemExit("index lacks patient_group -- rebuild with the current "
                         "mskdata.py (leakage control depends on it)")
    if cfg.smoke:
        keep = []
        for (a, s), g in index[index.dataset == "mura"].groupby(["anatomy", "split"]):
            st = g.study.drop_duplicates().head(24)
            keep.append(g[g.study.isin(st)])
        index = pd.concat(keep, ignore_index=True)
    print(f"[index] {len(index)} rows | "
          f"{index[index.dataset=='mura'].patient_group.nunique()} MURA people\n",
          flush=True)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    if cfg.stage in ("train", "all"):
        stage_train(index, cfg, prov)
    if cfg.stage in ("eval", "all"):
        stage_eval(index, cfg, prov)
    print(f"\nOutputs -> {Path(cfg.out_dir).resolve()}")


if __name__ == "__main__":
    main()
