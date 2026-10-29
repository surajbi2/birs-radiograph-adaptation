#!/usr/bin/env python3
"""Evaluate the joint encoders of joint_encoder_vs_ceiling.py against the Stage A
ceilings.

Reads the in-domain ceilings directly from the Stage A results, so they match the
Stage A values by construction, and runs inference only for the joint models.

Outputs (under --out-dir):
    joint_encoder_scores.csv, joint_encoder_vs_ceiling.csv

Usage:
    python joint_encoder_report.py --inspect                       # list Stage A arms
    python joint_encoder_report.py --backbones densenet121 --seeds 42 123 456 --gpu 0
    python joint_encoder_report.py --report-only

Requires a GPU for inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskdata
import mskmodel as M
from train_source_models import eval_block, CsvAppender

MURA_ANATOMIES = list(mskdata.MURA_ANATOMIES)

# Decision thresholds -- fixed before the run, unchanged from step 1 phase 1.
H3_DEAD_MARGIN = 0.010
H3_ALIVE_MARGIN = 0.030

# Candidate names for the column that distinguishes Stage A arms/conditions.
ARM_COL_CANDIDATES = ("condition", "arm", "method", "intervention", "variant")
# Arms that represent the plain trained model with no intervention applied.
ZEROSHOT_ARM_NAMES = ("zeroshot", "zero_shot", "base", "none", "raw", "plain",
                      "source", "baseline", "ceiling")


def joint_ckpt_path(ckpt_dir: Path, backbone: str, seed: int) -> Path:
    return Path(ckpt_dir) / f"joint_ALL__{backbone}__seed{seed}.pt"


def find_arm_col(d: pd.DataFrame):
    for c in ARM_COL_CANDIDATES:
        if c in d.columns:
            return c
    return None


def load_ceilings(csv_path: Path, arm: str | None, quiet: bool = False
                  ) -> pd.DataFrame:
    """
    In-domain ceiling = Stage A rows where source == target, one row per
    (backbone, anatomy, seed), with no intervention applied.
    """
    d = pd.read_csv(csv_path, comment="#", low_memory=False)
    need = {"source", "target", "backbone", "seed", "auc_study"}
    missing = need - set(d.columns)
    if missing:
        raise KeyError(f"{csv_path.name} lacks columns {sorted(missing)}; "
                       f"has {sorted(d.columns)}")

    ind = d[d.source == d.target].copy()
    arm_col = find_arm_col(ind)

    if arm_col is not None:
        arms = sorted(ind[arm_col].dropna().unique())
        if arm is None:
            hits = [a for a in arms if str(a).lower() in ZEROSHOT_ARM_NAMES]
            if len(hits) != 1:
                raise ValueError(
                    f"cannot auto-pick the ceiling arm from {arms}. "
                    f"Rerun with --ceiling-arm <name>. For an in-domain row "
                    f"(source==target) you want the arm with NO intervention.")
            arm = hits[0]
        elif arm not in arms:
            raise ValueError(f"--ceiling-arm {arm!r} not in {arms}")
        ind = ind[ind[arm_col] == arm]
        if not quiet:
            print(f"[ceiling] using {arm_col}={arm!r}")

    dup = ind.duplicated(["backbone", "target", "seed"]).sum()
    if dup:
        raise ValueError(
            f"{dup} duplicate (backbone, anatomy, seed) ceiling rows remain. "
            f"An arm column was not filtered. Columns: {sorted(ind.columns)}")

    out = (ind.rename(columns={"target": "anatomy",
                               "auc_study": "ceiling_auc_study"})
              [["backbone", "anatomy", "seed", "ceiling_auc_study"]]
              .reset_index(drop=True))
    if not quiet:
        print(f"[ceiling] {len(out)} rows | "
              f"{out.backbone.nunique()} backbones x {out.anatomy.nunique()} "
              f"anatomies x {out.seed.nunique()} seeds")
    return out


def inspect(cfg) -> None:
    d = pd.read_csv(cfg.source_models_csv, comment="#", low_memory=False)
    print(f"\n{cfg.source_models_csv}")
    print(f"  shape: {d.shape}")
    print(f"  columns: {sorted(d.columns)}")
    ind = d[d.source == d.target]
    print(f"\n  in-domain rows (source == target): {len(ind)}")
    arm_col = find_arm_col(ind)
    if arm_col is None:
        print("  no arm/condition column found -- rows should be unique already")
    else:
        print(f"  arm column: {arm_col!r}")
        print(f"  values and in-domain AUC:")
        g = (ind.groupby(arm_col)
                .agg(n=("auc_study", "size"), mean_auc=("auc_study", "mean"))
                .round(4))
        print(g.to_string())
        print("\n  Pick the arm with NO intervention applied. For source==target "
              "rows,\n  a zero-shot/base arm IS the in-domain ceiling.")
    print("\n  per-backbone in-domain mean (all arms pooled, sanity only):")
    print(ind.groupby("backbone").auc_study.mean().round(4).to_string())


def evaluate_joint(cfg, prov) -> None:
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    out_csv = Path(cfg.out_dir) / "joint_encoder_scores.csv"
    res = CsvAppender(out_csv, prov)

    done = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv, comment="#")
        done = set(zip(prev.backbone, prev.seed))

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    valid = {a: M.frame_for(index, "mura", a, "valid", cfg.label_col)
             for a in MURA_ANATOMIES}

    for bb in cfg.backbones:
        for seed in cfg.seeds:
            if (bb, seed) in done:
                print(f"[skip] {M.BACKBONE_SHORT[bb]} seed {seed} already scored")
                continue
            cp = joint_ckpt_path(cfg.joint_ckpt_dir, bb, seed)
            if not cp.exists():
                raise FileNotFoundError(f"missing joint checkpoint {cp}")
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            model = M.build_model(bb)
            model.load_state_dict(ck["state_dict"])
            model = model.to(device)
            joint_prev = float(ck["source_prev"])

            rows = []
            for a in MURA_ANATOMIES:
                pr = M.predict(model, valid[a], device,
                               batch_size=cfg.eval_batch_size,
                               num_workers=cfg.num_workers, img_size=cfg.img_size)
                e = eval_block(pr["logit"], pr["label"], pr["study"], joint_prev)
                rows.append({"backbone": bb, "norm_family": M.NORM_FAMILY[bb],
                             "seed": seed, "anatomy": a,
                             "joint_auc_study": e["auc_study"],
                             "joint_auc_image": e["auc_image"],
                             "joint_ece": e["ece"],
                             "joint_bacc_50": e["bacc_50"],
                             "joint_sens_50": e["sens_50"],
                             "joint_spec_50": e["spec_50"],
                             "n_studies": e["n_studies"],
                             "n_images": e["n_images"],
                             "target_prev": e["target_prev"]})
                print(f"  [joint] {M.BACKBONE_SHORT[bb]} s{seed} {a:<12s} "
                      f"auc_study={e['auc_study']:.4f}", flush=True)
            res.write(rows)
            del model
            torch.cuda.empty_cache()
    print("[joint scoring] complete", flush=True)


def report(cfg) -> None:
    jp = Path(cfg.out_dir) / "joint_encoder_scores.csv"
    if not jp.exists():
        print(f"no joint scores at {jp}; run the evaluation step first")
        return
    j = pd.read_csv(jp, comment="#")
    c = load_ceilings(Path(cfg.source_models_csv), cfg.ceiling_arm)

    d = j.merge(c, on=["backbone", "anatomy", "seed"], how="inner")
    lost = len(j) - len(d)
    if lost:
        print(f"[warn] {lost} joint rows had no matching ceiling row and were "
              f"dropped -- check seed/backbone alignment")
    d["delta_auc_study"] = d.joint_auc_study - d.ceiling_auc_study
    d.to_csv(Path(cfg.out_dir) / "joint_encoder_vs_ceiling.csv", index=False)

    print("\n" + "=" * 78)
    print("PER-ANATOMY: one shared encoder vs that anatomy's own in-domain ceiling")
    print("=" * 78)
    per = (d.groupby(["backbone", "anatomy"])
             .agg(joint=("joint_auc_study", "mean"),
                  joint_sd=("joint_auc_study", "std"),
                  ceiling=("ceiling_auc_study", "mean"),
                  ceiling_sd=("ceiling_auc_study", "std"),
                  delta=("delta_auc_study", "mean"),
                  n=("seed", "nunique")).round(4))
    print(per.to_string())

    print("\n" + "=" * 78)
    print("VERDICT on H3 (concept heterogeneity) and H4 (irreducible difficulty)")
    print("=" * 78)
    for bb, g in per.groupby(level=0):
        g = g.droplevel(0)
        deficits = g[g.delta < -H3_DEAD_MARGIN]
        severe = g[g.delta < -H3_ALIVE_MARGIN]
        print(f"\n  {M.BACKBONE_SHORT.get(bb, bb)}   "
              f"mean delta {g.delta.mean():+.4f} | "
              f"worst {g.delta.idxmin()} {g.delta.min():+.4f}")
        if len(severe) >= 2:
            print(f"    >>> H3 ALIVE: {len(severe)} anatomies more than "
                  f"{H3_ALIVE_MARGIN} below ceiling")
            print(severe[["joint", "ceiling", "delta"]].to_string())
            print("    STOP -- a shared encoder cannot serve these anatomies.")
        elif len(deficits) == 0:
            print(f"    H3 DEAD, H4 DEAD -- every anatomy within "
                  f"{H3_DEAD_MARGIN} of its own ceiling.")
            print("    -> layerwise diagnostic becomes a clean H1 vs H2 test.")
        else:
            print(f"    AMBIGUOUS: {len(deficits)} anatomies between "
                  f"{H3_DEAD_MARGIN} and {H3_ALIVE_MARGIN} below ceiling")
            print(deficits[["joint", "ceiling", "delta"]].to_string())

    print("\n" + "=" * 78)
    print("DIFFICULTY vs DATA VOLUME  (is 'harder' just 'less data'?)")
    print("=" * 78)
    idx = pd.read_csv(cfg.index, comment="#", low_memory=False)
    n_train = (idx[(idx.dataset == "mura") & (idx.split == "train")]
               .groupby("anatomy").size().rename("train_images"))
    v = (d.groupby("anatomy")
           .agg(joint=("joint_auc_study", "mean"),
                ceiling=("ceiling_auc_study", "mean")).join(n_train)
           .sort_values("joint", ascending=False).round(4))
    print(v.to_string())
    if v.train_images.notna().all() and len(v) > 2:
        r = np.corrcoef(v.train_images, v.joint)[0, 1]
        print(f"\n  Spearman-ish check: corr(train_images, joint AUC) = {r:+.3f}")
        print("  Near zero means anatomy difficulty is NOT a data-volume story.")

    print("\n  Caveat: the joint model saw ~4x the images of any single-anatomy")
    print("  model. MATCHING ceiling is evidence concepts are compatible.")
    print("  BEATING ceiling is partly data volume, not extra evidence.")
    print("  LOSING despite 4x data would be strong evidence for H3.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--joint-ckpt-dir", default=str(M.DEFAULT_CKPT))
    ap.add_argument("--source-models-csv",
                    default=str(Path(M.DEFAULT_RESULTS) / "source_models"
                                / "anatomy_transfer_arms_merged.csv"))
    ap.add_argument("--ceiling-arm", default=None,
                    help="Stage A arm to use as the ceiling; auto-detected if omitted")
    ap.add_argument("--out-dir", default=str(M.DEFAULT_RESULTS))
    ap.add_argument("--backbones", nargs="+", default=["densenet121"],
                    choices=list(M.BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--label-col", default="label_abnormal")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    cfg = ap.parse_args()

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    if cfg.inspect:
        inspect(cfg)
        return
    if cfg.report_only:
        report(cfg)
        return

    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({"script": "joint_encoder_report.py",
                         "backbones": list(cfg.backbones),
                         "seeds": list(cfg.seeds),
                         "source_models_csv": cfg.source_models_csv})
    evaluate_joint(cfg, prov)
    report(cfg)


if __name__ == "__main__":
    main()
