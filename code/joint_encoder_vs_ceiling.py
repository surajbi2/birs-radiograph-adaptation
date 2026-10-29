#!/usr/bin/env python3
"""Train one encoder on all seven MURA anatomies and compare it with the
per-anatomy in-domain ceilings.

If a single shared encoder reaches each anatomy's own ceiling, the anatomies do
not need incompatible features, which rules out label heterogeneity and
irreducible difficulty as explanations of the cross-anatomy transfer gap.
Phase 1 trains the joint models; joint_encoder_report.py evaluates them.

Outputs (under --out-dir):
    joint_ALL__<backbone>__seed<seed>.pt, joint_encoder_training_log.csv,
    joint_encoder_vs_ceiling.csv

Usage:
    python joint_encoder_vs_ceiling.py --backbones densenet121 --seeds 42 123 456 --gpu 0
    python joint_encoder_vs_ceiling.py --report-only

Requires a GPU.
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
from train_source_models import eval_block, ckpt_path, CsvAppender

MURA_ANATOMIES = mskdata.MURA_ANATOMIES
STAGE_A_CEILING_ANCHOR = 0.8863          # from the 840-cell Stage A decomposition
CEILING_TOLERANCE = 0.010                # drift beyond this = pipeline changed

# Decision thresholds, fixed before the run.
H3_DEAD_MARGIN = 0.010     # joint >= ceiling - 0.010 on all anatomies -> H3 rejected
H3_ALIVE_MARGIN = 0.030    # joint <  ceiling - 0.030 on >=2 anatomies -> H3 supported


def joint_ckpt_path(ckpt_dir: Path, backbone: str, seed: int) -> Path:
    return Path(ckpt_dir) / f"joint_ALL__{backbone}__seed{seed}.pt"


def joint_frame(index: pd.DataFrame, split: str, label_col: str) -> pd.DataFrame:
    """All 7 MURA anatomies pooled, with 'label' and 'anatomy' both present."""
    parts = [M.frame_for(index, "mura", a, split, label_col) for a in MURA_ANATOMIES]
    out = pd.concat(parts, ignore_index=True)
    assert out.anatomy.nunique() == len(MURA_ANATOMIES), "lost an anatomy in concat"
    return out


def load_checkpoint(path: Path, backbone: str, device):
    import torch
    if not path.exists():
        raise FileNotFoundError(
            f"missing checkpoint {path}\n"
            f"Step 1 reuses Stage A source models as in-domain ceilings. "
            f"If Stage A used different seeds, pass --seeds to match."
        )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("backbone") != backbone:
        raise ValueError(f"{path.name} holds backbone {ck.get('backbone')!r}, "
                         f"expected {backbone!r}")
    model = M.build_model(backbone)
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck


# ------------------------------------------------------------------ phase 1 --

def train_joint(index, cfg, prov) -> None:
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    log = CsvAppender(Path(cfg.out_dir) / "joint_encoder_training_log.csv", prov)

    frame = joint_frame(index, "train", cfg.label_col)
    print(f"[joint pool] {len(frame)} images | {frame.study.nunique()} studies "
          f"| {frame.patient_group.nunique()} people | prev={frame.label.mean():.3f}",
          flush=True)
    print(frame.groupby("anatomy").agg(images=("uid", "size"),
                                       prev=("label", "mean")).round(3).to_string(),
          flush=True)

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            cp = joint_ckpt_path(cfg.ckpt_dir, bb, seed)
            if cp.exists():
                print(f"[skip] {cp.name}", flush=True)
                continue
            print(f"\n[train joint] {M.BACKBONE_SHORT[bb]} | seed {seed} "
                  f"| {cfg.epochs} epochs", flush=True)
            t0 = time.time()
            model, hist = M.train_model(
                frame, bb, seed, device, epochs=cfg.epochs,
                batch_size=cfg.batch_size, lr=cfg.lr,
                num_workers=cfg.num_workers, img_size=cfg.img_size)
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "backbone": bb,
                        "source": "JOINT_ALL", "seed": seed, "epochs": cfg.epochs,
                        "label_col": cfg.label_col,
                        "source_prev": float(frame.label.mean()),
                        "anatomies": list(MURA_ANATOMIES),
                        "history": hist, "provenance": prov}, cp)
            log.write({"backbone": bb, "seed": seed, "n_images": len(frame),
                       "n_studies": int(frame.study.nunique()),
                       "n_patients": int(frame.patient_group.nunique()),
                       "source_prev": float(frame.label.mean()),
                       "epochs": cfg.epochs, "final_loss": hist[-1]["loss"],
                       "minutes": round((time.time() - t0) / 60, 2),
                       "checkpoint": str(cp)})
            del model
            torch.cuda.empty_cache()
    print("\n[phase 1] joint training complete", flush=True)


# ------------------------------------------------------------------ phase 2 --

def evaluate(index, cfg, prov) -> None:
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    res = CsvAppender(Path(cfg.out_dir) / "joint_encoder_vs_ceiling.csv", prov)

    valid = {a: M.frame_for(index, "mura", a, "valid", cfg.label_col)
             for a in MURA_ANATOMIES}

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            rows = []

            # ---- joint model, evaluated on every anatomy's valid split -----
            jm, jck = load_checkpoint(joint_ckpt_path(cfg.ckpt_dir, bb, seed),
                                      bb, device)
            joint_prev = float(jck["source_prev"])
            joint_scores = {}
            for a in MURA_ANATOMIES:
                pr = M.predict(jm, valid[a], device, batch_size=cfg.eval_batch_size,
                               num_workers=cfg.num_workers, img_size=cfg.img_size)
                joint_scores[a] = eval_block(pr["logit"], pr["label"],
                                             pr["study"], joint_prev)
                print(f"  [joint] {M.BACKBONE_SHORT[bb]} s{seed} {a:<12s} "
                      f"auc_study={joint_scores[a]['auc_study']:.4f}", flush=True)
            del jm
            torch.cuda.empty_cache()

            # ---- per-anatomy Stage A model on its OWN valid = ceiling ------
            for a in MURA_ANATOMIES:
                cm, cck = load_checkpoint(ckpt_path(cfg.source_ckpt_dir, a, bb, seed),
                                          bb, device)
                pr = M.predict(cm, valid[a], device, batch_size=cfg.eval_batch_size,
                               num_workers=cfg.num_workers, img_size=cfg.img_size)
                ceil = eval_block(pr["logit"], pr["label"], pr["study"],
                                  float(cck["source_prev"]))
                del cm
                torch.cuda.empty_cache()

                j = joint_scores[a]
                rows.append({
                    "backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                    "seed": seed, "anatomy": a,
                    "n_studies": j["n_studies"], "n_images": j["n_images"],
                    "target_prev": j["target_prev"],
                    "joint_auc_study": j["auc_study"],
                    "ceiling_auc_study": ceil["auc_study"],
                    "delta_auc_study": j["auc_study"] - ceil["auc_study"],
                    "joint_auc_image": j["auc_image"],
                    "ceiling_auc_image": ceil["auc_image"],
                    "joint_ece": j["ece"], "ceiling_ece": ceil["ece"],
                    "joint_bacc_50": j["bacc_50"], "ceiling_bacc_50": ceil["bacc_50"],
                    "joint_spec_50": j["spec_50"], "ceiling_spec_50": ceil["spec_50"],
                })
                print(f"  [ceil ] {M.BACKBONE_SHORT[bb]} s{seed} {a:<12s} "
                      f"auc_study={ceil['auc_study']:.4f}  "
                      f"delta={rows[-1]['delta_auc_study']:+.4f}", flush=True)

            res.write(rows)          # incremental: survives kill -9
    print("\n[phase 2] evaluation complete", flush=True)


# ------------------------------------------------------------------- report --

def report(cfg) -> None:
    path = Path(cfg.out_dir) / "joint_encoder_vs_ceiling.csv"
    if not path.exists():
        print(f"no results at {path}")
        return
    d = pd.read_csv(path, comment="#")

    print("\n" + "=" * 78)
    print("CONSISTENCY GATE -- ceiling must reproduce the Stage A anchor")
    print("=" * 78)
    obs = d.ceiling_auc_study.mean()
    drift = obs - STAGE_A_CEILING_ANCHOR
    print(f"  observed mean ceiling auc_study : {obs:.4f}")
    print(f"  Stage A anchor                  : {STAGE_A_CEILING_ANCHOR:.4f}")
    print(f"  drift                           : {drift:+.4f}")
    if abs(drift) > CEILING_TOLERANCE:
        print("  >>> FAIL. The pipeline has drifted from Stage A. Debug this "
              "before interpreting anything below. <<<")
    else:
        print("  PASS")

    print("\n" + "=" * 78)
    print("PER-ANATOMY: joint (one shared encoder) vs its own in-domain ceiling")
    print("=" * 78)
    per = (d.groupby(["backbone", "anatomy"])
             .agg(joint=("joint_auc_study", "mean"),
                  joint_sd=("joint_auc_study", "std"),
                  ceiling=("ceiling_auc_study", "mean"),
                  ceiling_sd=("ceiling_auc_study", "std"),
                  delta=("delta_auc_study", "mean"),
                  n_seeds=("seed", "nunique")).round(4))
    print(per.to_string())

    print("\n" + "=" * 78)
    print("VERDICT on H3 (concept heterogeneity) and H4 (irreducible difficulty)")
    print("=" * 78)
    for bb, g in per.groupby(level=0):
        deficits = g[g.delta < -H3_DEAD_MARGIN]
        severe = g[g.delta < -H3_ALIVE_MARGIN]
        print(f"\n  {M.BACKBONE_SHORT.get(bb, bb)}")
        print(f"    worst anatomy: {g.delta.idxmin()[1]} ({g.delta.min():+.4f})")
        if len(severe) >= 2:
            print(f"    >>> H3 ALIVE. {len(severe)} anatomies fall more than "
                  f"{H3_ALIVE_MARGIN} below ceiling:")
            print(severe[["joint", "ceiling", "delta"]].to_string())
            print("    STOP. A shared encoder cannot serve these anatomies. "
                  "The thesis framing needs revisiting before the layerwise run.")
        elif len(deficits) == 0:
            print(f"    H3 DEAD and H4 DEAD. Every anatomy is within "
                  f"{H3_DEAD_MARGIN} of its own ceiling with ONE shared encoder.")
            print("    -> layerwise diagnostic is now a clean H1 vs H2 test.")
        else:
            print(f"    AMBIGUOUS. {len(deficits)} anatomies between "
                  f"{H3_DEAD_MARGIN} and {H3_ALIVE_MARGIN} below ceiling:")
            print(deficits[["joint", "ceiling", "delta"]].to_string())
            print("    Add seeds, or rerun joint training with more epochs, "
                  "before concluding either way.")

    print("\n  Reminder: the joint model had ~4x the data. If it MATCHES ceiling "
          "that is evidence concepts are compatible.\n  If it BEATS ceiling that "
          "is partly a data-volume effect and not extra evidence for H3 being "
          "dead.\n  If it LOSES despite 4x data, that is strong evidence for H3.")


# ---------------------------------------------------------------------- main --

def main() -> None:
    home = M.DEFAULT_HOME
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--ckpt-dir", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(M.DEFAULT_RESULTS))
    ap.add_argument("--source-ckpt-dir", default=None,
                    help="Stage A source checkpoints; defaults to --ckpt-dir")
    ap.add_argument("--backbones", nargs="+", default=["densenet121"],
                    choices=list(M.BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--label-col", default="label_abnormal")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    cfg = ap.parse_args()
    if cfg.source_ckpt_dir is None:
        cfg.source_ckpt_dir = cfg.ckpt_dir

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    if cfg.report_only:
        report(cfg)
        return

    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({"script": "joint_encoder_vs_ceiling.py",
                         "backbones": list(cfg.backbones),
                         "seeds": list(cfg.seeds), "epochs": cfg.epochs,
                         "label_col": cfg.label_col})

    index = pd.read_csv(cfg.index, comment="#")
    index = mskdata.add_patient_group(index) if "patient_group" not in index.columns \
        else index

    if not cfg.eval_only:
        train_joint(index, cfg, prov)
    if not cfg.train_only:
        evaluate(index, cfg, prov)
        report(cfg)


if __name__ == "__main__":
    main()
