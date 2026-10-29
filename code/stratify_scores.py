#!/usr/bin/env python3
"""Stratify any score-dump set by an index column, and train ceilings on subsets.

GRAZPEDWRI-DX mixes initial examinations with follow-ups, which are almost all
positive (casts, healing callus, hardware). ``analyze`` reports AUC per stratum
of a column; ``train`` trains and scores ceiling models on a subset such as
initial examinations only.

Outputs (under --out-dir):
    score_strata.csv, graz_<subset>_ceiling.csv

Usage:
    python stratify_scores.py analyze --by meta_initial_exam
    python stratify_scores.py train --subset initial_only --gpu 0

``train`` requires a GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskmodel as M
import preprocessing_variants as P
from train_source_models import CsvAppender
from dataset_shift import (graz_frame, with_descriptors, projection_cell,
                         save_dump, load_index, alt_labels, load_source)

SUBSETS = {
    # name -> (predicate on the GRAZ frame, human description)
    "initial_only": (lambda f: pd.to_numeric(f.get("meta_initial_exam"),
                                             errors="coerce") == 1,
                     "initial_exam == 1 (excludes follow-up imaging)"),
    "followup_only": (lambda f: pd.to_numeric(f.get("meta_initial_exam"),
                                              errors="coerce") != 1,
                      "initial_exam missing (follow-up imaging only)"),
    "cast_free": (lambda f: pd.to_numeric(f.get("has_cast"),
                                          errors="coerce").fillna(0) == 0,
                  "has_cast == 0"),
}


def mode_analyze(cfg) -> None:
    from shortcut_reliance import score
    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    if cfg.by not in index.columns:
        raise SystemExit(f"{cfg.by!r} not in the index. available meta columns: "
                         f"{[c for c in index.columns if c.startswith('meta_') or c.startswith('has_')]}")
    key = index.set_index("uid")[cfg.by]

    dumps = sorted(Path(cfg.dump_dir).glob(cfg.pattern))
    if not dumps:
        raise SystemExit(f"no dumps matching {cfg.pattern} in {cfg.dump_dir}")
    print(f"reading {len(dumps)} dumps, stratifying by {cfg.by!r}\n")

    rows = []
    for d in dumps:
        z = np.load(d, allow_pickle=True)
        uid = z["uid"]
        v = pd.to_numeric(key.reindex(uid), errors="coerce")
        parts = d.stem.split("__")
        meta = {"arm": parts[0], "score_label": parts[1], "prep": parts[2],
                "backbone": parts[3],
                "seed": int(parts[-1].replace("seed", ""))}
        logit, resid = z["logit"], z["resid"]
        label, study = z["label"], z["study"]
        groups = {"all": np.ones(len(uid), bool),
                  f"{cfg.by}=1": (v == 1).to_numpy(),
                  f"{cfg.by}!=1": (v != 1).to_numpy()}
        for name, m in groups.items():
            if m.sum() < 50 or len(np.unique(label[m])) < 2:
                continue
            s = score(logit[m], label[m], study[m])
            sr = score(resid[m], label[m], study[m])
            rows.append({**meta, "stratum": name, "n": int(m.sum()),
                         "prevalence": float(label[m].mean()),
                         "auc_study": s["auc_study"],
                         "auc_image": s["auc_image"],
                         "delta_study": sr["auc_study"] - s["auc_study"]})
    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("nothing analysable -- check the uid join")

    for lab in sorted(r.score_label.unique()):
        sub = r[r.score_label == lab]
        print("=" * 78)
        print(f"SCORE LABEL: {lab}")
        print("=" * 78)
        t = sub.pivot_table(index=["arm", "prep", "backbone"], columns="stratum",
                            values="auc_study", aggfunc="mean")
        print(t.round(4).to_string())
        pv = sub.pivot_table(index="stratum", values=["n", "prevalence"],
                             aggfunc="mean")
        print("\n" + pv.round(3).to_string())
        full = sub[sub.stratum == "all"].auc_study.mean()
        for st in sub.stratum.unique():
            if st == "all":
                continue
            v = sub[sub.stratum == st].auc_study.mean()
            print(f"\n  all {full:.4f}   {st} {v:.4f}   drop {full - v:+.4f}")
        print("\n  This is a full-pool model EVALUATED on a subset -- a LOWER")
        print("  BOUND on the subset effect, not the subset ceiling. Run")
        print("  `stratify_scores.py train --subset ...` for the real one.")

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(out, index=False)
    print(f"\nwritten to {out}")


def mode_train(cfg) -> None:
    import torch
    pred, desc = SUBSETS[cfg.subset]
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({"script": "stratify_scores.py train",
                         "subset": cfg.subset, "definition": desc,
                         "backbones": list(cfg.backbones),
                         "seeds": list(cfg.seeds), "prep": cfg.prep})
    index = load_index(cfg)
    out = Path(cfg.out_dir) / f"graz_{cfg.subset}_ceiling.csv"
    res = CsvAppender(out, prov)

    tr = graz_frame(index, "train", cfg.train_label)
    va = graz_frame(index, "valid", cfg.train_label)
    for f in (tr, va):
        if "meta_initial_exam" not in f.columns and "initial" in cfg.subset:
            raise SystemExit("meta_initial_exam missing from the index")
    tr = tr[pred(tr).to_numpy()].reset_index(drop=True)
    va = va[pred(va).to_numpy()].reset_index(drop=True)
    if len(tr) < 500:
        raise SystemExit(f"subset {cfg.subset!r} leaves only {len(tr)} train "
                         "images; not runnable")
    va, X = with_descriptors(va)
    alt = alt_labels(index, va, cfg.score_labels)
    print(f"[{cfg.subset}] {desc}")
    print(f"  train {len(tr)} images / {tr.study.nunique()} studies "
          f"prev={tr.label.mean():.4f}")
    print(f"  valid {len(va)} images / {va.study.nunique()} studies "
          f"prev={va.label.mean():.4f}")
    print(f"  MURA XR_WRIST train prevalence is 0.409 for comparison\n")

    ck_dir = Path(cfg.ckpt_root) / f"graz_{cfg.subset}_models"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ref = (P.get_ref_cdf(index, Path(cfg.cache_dir))
           if cfg.prep == "histmatch" else None)
    fn = P.make_loader_fn(cfg.prep, ref)

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            cp = ck_dir / f"{cfg.subset}__{cfg.prep}__{bb}__seed{seed}.pt"
            t0 = time.time()
            if cp.exists():
                print(f"  [load] {cp.name}")
                model, _ = load_source(cp, bb, device)
                hist = [{"loss": float("nan")}]
            else:
                print(f"  [train] {M.BACKBONE_SHORT[bb]} s{seed} {cfg.subset}",
                      flush=True)
                model, hist = P.train(tr, bb, seed, device, fn,
                                      epochs=cfg.epochs,
                                      batch_size=cfg.batch_size, lr=cfg.lr,
                                      num_workers=cfg.num_workers,
                                      img_size=cfg.img_size)
                torch.save({"state_dict": model.state_dict(), "backbone": bb,
                            "source": f"GRAZ_{cfg.subset}", "seed": seed,
                            "prep": cfg.prep, "label_col": cfg.train_label,
                            "history": hist, "provenance": prov}, cp)
            pr = P.predict(model, va, device, cfg.eval_batch_size,
                           cfg.num_workers, cfg.img_size, fn)
            z, st = pr["logit"], pr["study"]
            for sl, y in alt.items():
                cell, resid = projection_cell(
                    z, y, st, X, (cfg.subset, bb, seed, cfg.prep, sl))
                row = {"backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                       "seed": seed, "prep": cfg.prep, "arm": cfg.subset,
                       "train_label": cfg.train_label, "score_label": sl,
                       "n_train_images": len(tr),
                       "n_train_studies": int(tr.study.nunique()),
                       "train_prev": float(tr.label.mean()),
                       "n_images": len(va),
                       "n_studies": int(va.study.nunique()),
                       "target_prev": float(y.mean()),
                       "final_loss": float(hist[-1]["loss"]),
                       "minutes": (time.time() - t0) / 60.0}
                row.update(cell)
                res.write(row)
                save_dump(Path(cfg.out_dir) / "dumps" /
                          f"{cfg.subset}__{sl}__{cfg.prep}__{bb}__seed{seed}.npz",
                          va, z, resid, y, st, X, row)
                print(f"    [{sl}] AUC={cell['zs_auc_study']:.4f} "
                      f"R2={cell['appearance_r2']:.3f} "
                      f"delta={cell['delta_study']:+.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
    print(f"[{cfg.subset}] complete")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["analyze", "train"])
    ap.add_argument("--by", default="meta_initial_exam")
    ap.add_argument("--subset", default="initial_only", choices=list(SUBSETS))
    ap.add_argument("--pattern", default="ceiling__*.npz")
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-root", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--cache-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer"))
    ap.add_argument("--dump-dir", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                              "dumps"))
    ap.add_argument("--out", default=str(Path(M.DEFAULT_RESULTS) / "transfer" /
                                         "score_strata.csv"))
    ap.add_argument("--backbones", nargs="+",
                    default=["densenet121", "swin_tiny_patch4_window7_224"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(M.SEEDS))
    ap.add_argument("--prep", default="standard", choices=list(P.PREPS))
    ap.add_argument("--train-label", default="label_abnormal")
    ap.add_argument("--score-labels", nargs="+",
                    default=["label_abnormal", "label_fracture"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    cfg = ap.parse_args()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    (mode_analyze if cfg.mode == "analyze" else mode_train)(cfg)


if __name__ == "__main__":
    main()
