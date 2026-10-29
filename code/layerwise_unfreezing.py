#!/usr/bin/env python3
"""Layer-wise unfreezing experiment: which encoder blocks must change for a
wrist-trained model to work on another anatomy.

Loads a wrist-trained source model, unfreezes k blocks counted from the top (or
from the stem), fine-tunes on target labels and records AUC as a function of k,
next to a frozen-feature probe reference.

Outputs (under --out-dir):
    layerwise_capacity.csv, layerwise_reference.csv, layerwise_cells.csv,
    layerwise_report.csv

Usage:
    python layerwise_unfreezing.py --capacity       # check the block maps
    python layerwise_unfreezing.py --reference      # probe reference, no training
    python layerwise_unfreezing.py --gpu 0          # the experiment

Requires a GPU.
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskdata
import mskmodel as M
from train_source_models import eval_block, ckpt_path, CsvAppender
import layerwise_block_maps as LF

CEILING_ARM_CANDIDATES = ("ceiling",)
ZEROSHOT_ARM_CANDIDATES = ("zeroshot", "zero_shot", "base", "none", "raw", "plain")
CELL_KEYS = ["backbone", "source", "target", "prep", "direction", "depth",
             "seed", "budget"]


def subsample_studies(frame: pd.DataFrame, n_studies: int, seed: int
                      ) -> pd.DataFrame:
    """
    Restrict a target training pool to ~n_studies, sampling by PATIENT so no
    person is split across the in/out boundary.

    Seed-controlled: different layerwise seeds see different subsamples, which
    is the correct source of variance for a label-budget claim. n_studies <= 0
    means use everything.

    Why this exists: at the full budget a re-fitted readout already recovers
    90.7% of the wrist->shoulder gap, leaving no headroom for the depth curve
    to move into. Encoder adaptation can only be measured where the readout
    cannot already solve the task.
    """
    if n_studies <= 0 or frame.study.nunique() <= n_studies:
        return frame.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    pats = frame.patient_group.unique().copy()
    rng.shuffle(pats)
    per_pat = frame.groupby("patient_group").study.nunique()
    keep, total = [], 0
    for p in pats:
        keep.append(p)
        total += int(per_pat[p])
        if total >= n_studies:
            break
    return frame[frame.patient_group.isin(keep)].reset_index(drop=True)


# ============================================================ preprocessing ==

class CropAspectDataset(M.MskDataset):
    """
    The `cropped` preprocessing arm.

    Standard (mskmodel.MskDataset):
        load_rgb_uint8 -> Resize((S, S))   [aspect DESTROYED, padding KEPT]
    This arm:
        load_rgb_uint8 -> crop to content bbox -> aspect-preserving resize
        -> centre on an S x S zero canvas
    """

    def __init__(self, frame, img_size: int = 224, train_mode: bool = False,
                 thresh: int = 10):
        super().__init__(frame, img_size, train_mode)
        self.thresh = thresh
        from torchvision import transforms
        aug = ([transforms.RandomHorizontalFlip(p=0.5),          # NEVER vertical
                transforms.RandomRotation(degrees=10)] if train_mode else [])
        self.tf = transforms.Compose(              # no Resize: already S x S
            [transforms.ToPILImage()] + aug +
            [transforms.ToTensor(),
             transforms.Normalize(mean=M.IMAGENET_MEAN, std=M.IMAGENET_STD)])

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image
        r = self.frame.iloc[idx]
        rgb = mskdata.load_rgb_uint8(r["path"])                  # HxWx3 uint8

        mask = rgb[..., 0] > self.thresh
        if mask.any():
            ys, xs = np.where(mask)
            rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        h, w = rgb.shape[:2]
        s = self.img_size / max(h, w)
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        im = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)

        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        top, left = (self.img_size - nh) // 2, (self.img_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = np.asarray(im)

        x = self.tf(canvas)
        y = torch.tensor(float(r["label"]), dtype=torch.float32)
        return x, y, idx


def make_loader(frame, batch_size, train_mode, num_workers, img_size, prep):
    """Mirror of mskmodel.make_loader with a selectable dataset class."""
    import torch
    from torch.utils.data import DataLoader
    cls = M.MskDataset if prep == "standard" else CropAspectDataset
    g = torch.Generator()
    g.manual_seed(0)
    return DataLoader(
        cls(frame, img_size, train_mode),
        batch_size=batch_size, shuffle=train_mode, num_workers=num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=M._worker_init,
        generator=g if train_mode else None, persistent_workers=False)


def predict(model, frame, device, batch_size, num_workers, img_size, prep):
    """Mirror of mskmodel.predict honouring the preprocessing arm. Always fp32."""
    import torch
    loader = make_loader(frame, batch_size, False, num_workers, img_size, prep)
    model.eval()
    n = len(frame)
    logit = np.full(n, np.nan, dtype=np.float64)
    label = np.full(n, -1, dtype=np.int64)
    try:
        with torch.no_grad():
            for x, y, idx in loader:
                x = x.to(device, non_blocking=True)
                i = idx.numpy()
                z = model(x)
                logit[i] = z.detach().float().cpu().numpy().astype(np.float64)
                label[i] = y.numpy().astype(np.int64)
    finally:
        M.shutdown_loader(loader)
    if not np.all(np.isfinite(logit)):
        raise FloatingPointError("non-finite logits during inference")
    if (label < 0).any():
        raise RuntimeError("some rows were never visited by the loader")
    return {"logit": logit, "label": label, "study": frame["study"].to_numpy()}


# ================================================================= training ==

def finetune(model, frozen_modules, frame, device, epochs, batch_size, lr,
             weight_decay, num_workers, img_size, prep, head_lr_mult: float = 1.0,
             warmup_frac: float = 0.05, grad_clip: float = 1.0) -> List[Dict]:
    """
    Faithful copy of mskmodel.train_model's optimisation, with two changes:
      1. the optimiser receives ONLY trainable parameters
      2. train_with_frozen() replaces model.train(), every epoch, so frozen
         BatchNorm running statistics cannot drift toward the target
    Everything else is identical, so results stay comparable with Stage A.
    """
    import torch
    import torch.nn as nn

    loader = make_loader(frame, batch_size, True, num_workers, img_size, prep)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; the head must never be frozen")

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    # Discriminative learning rate. The head is freshly initialised and holds
    # ~1k parameters; at depth 0 it is the ONLY thing training. A single lr
    # tuned for a 7M-parameter trunk leaves it underfit -- measured: depth 0 on
    # XR_HUMERUS reached AUC 0.8573 against a converged sklearn probe's 0.8918
    # at a matched label budget. An underfit depth 0 depresses the baseline
    # while deeper arms fit faster under the same schedule, which manufactures
    # a rising curve for purely optimisation reasons, i.e. a fake H1 result.
    #
    # The multiplier is applied at EVERY depth, so it is uniform and not a
    # per-depth confound.
    head_ids = {id(p) for p in model.head.parameters()}
    head_params = [p for p in trainable if id(p) in head_ids]
    body_params = [p for p in trainable if id(p) not in head_ids]
    groups = []
    if body_params:
        groups.append({"params": body_params, "lr": lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr * head_lr_mult})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    total = max(1, epochs * len(loader))
    warm = max(1, int(warmup_frac * total))

    def sched_fn(step: int) -> float:
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
    history: List[Dict] = []
    step = 0
    try:
        for ep in range(epochs):
            LF.train_with_frozen(model, frozen_modules)      # NOT model.train()
            run, nb = 0.0, 0
            for x, y, _ in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss @ epoch {ep} step {step}; aborting")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                if not torch.isfinite(gn):
                    raise FloatingPointError(
                        f"non-finite grad norm @ epoch {ep} step {step}; aborting")
                opt.step()
                sched.step()
                run += float(loss.item())
                nb += 1
                step += 1
            history.append({"epoch": ep + 1, "loss": run / max(1, nb),
                            "lr": sched.get_last_lr()[0]})
    finally:
        M.shutdown_loader(loader)
    return history


# ================================================================== anchors ==

def _pick_arm(available, candidates, what, override):
    if override is not None:
        if override not in available:
            raise ValueError(f"--{what} {override!r} not in {sorted(available)}")
        return override
    hits = [a for a in available if str(a).lower() in candidates]
    if len(hits) != 1:
        raise ValueError(f"cannot auto-pick the {what} from {sorted(available)}. "
                         f"Rerun with --{what} <name>.")
    return hits[0]


def load_anchors(cfg, quiet: bool = False) -> pd.DataFrame:
    """
    Per (backbone, target): zero-shot AUC, in-domain ceiling AUC, the gap, and
    the Stage A probe AUC at its largest label budget.

    Everything is keyed on (backbone, target) and averaged over Stage A's 5
    seeds -- the probe CSV's seed/rep structure does not align cell-for-cell
    with the layerwise seeds, so a per-seed join would be false precision.
    """
    d = pd.read_csv(cfg.source_models_csv, comment="#", low_memory=False)
    arms = set(d.condition.dropna().unique())
    ca = _pick_arm(arms, CEILING_ARM_CANDIDATES, "ceiling-arm", cfg.ceiling_arm)
    za = _pick_arm(arms, ZEROSHOT_ARM_CANDIDATES, "zeroshot-arm", cfg.zeroshot_arm)

    zs = (d[(d.condition == za) & (d.source == cfg.source) & (d.target != cfg.source)]
          .groupby(["backbone", "target"]).auc_study.mean().rename("auc_zeroshot"))
    ce = (d[(d.condition == ca) & (d.target != cfg.source)]
          .groupby(["backbone", "target"]).auc_study.mean().rename("auc_ceiling"))
    a = pd.concat([zs, ce], axis=1).dropna()
    a["gap"] = a.auc_ceiling - a.auc_zeroshot

    files = sorted(glob.glob(cfg.probe_glob))
    if files:
        P = pd.concat([pd.read_csv(f, comment="#", low_memory=False) for f in files])
        P = P[(P.source == cfg.source) & (P.requested_n == P.requested_n.max())]
        a = a.join(P.groupby(["backbone", "target"]).auc_study.mean()
                    .rename("auc_probe"))
        a["probe_n"] = int(P.requested_n.max())
    else:
        a["auc_probe"] = np.nan
        a["probe_n"] = -1

    a = a.reset_index()
    if not quiet:
        print(f"[anchors] ceiling={ca!r} zeroshot={za!r} source={cfg.source} "
              f"| probe files: {len(files)}")
        small = a[a.gap < cfg.min_gap]
        if len(small):
            print(f"[anchors] {len(small)} (backbone, target) cells have gap < "
                  f"{cfg.min_gap}; recovery fraction will be NaN there:")
            print(small[["backbone", "target", "auc_zeroshot", "auc_ceiling", "gap"]]
                  .round(4).to_string(index=False))
    return a


# ==================================================================== cells ==

def build_jobs(cfg) -> List[Dict]:
    """
    depth 0 (head only) and depth 5 (full fine-tune) are IDENTICAL for both
    directions, so they are generated once under top_down and mirrored in the
    report. That removes a third of the bottom_up work.
    """
    jobs = []
    for bb in cfg.backbones:
        for tgt in cfg.targets:
            for prep in cfg.prep:
                for direction in cfg.directions:
                    for depth in cfg.depths:
                        if direction == "bottom_up" and depth in (0, LF.N_BLOCKS):
                            continue
                        for seed in cfg.seeds:
                            jobs.append({"backbone": bb, "source": cfg.source,
                                         "target": tgt, "prep": prep,
                                         "direction": direction, "depth": depth,
                                         "seed": seed,
                                         "budget": cfg.max_train_studies})
    return jobs


def run(cfg, prov) -> None:
    import torch
    import torch.nn as nn
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    out_csv = Path(cfg.out_dir) / "layerwise_cells.csv"
    res = CsvAppender(out_csv, prov)

    done = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv, comment="#")
        done = set(map(tuple, prev[CELL_KEYS].values.tolist()))
        print(f"[resume] {len(done)} cells already complete")

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)

    pools: Dict[str, Dict[str, pd.DataFrame]] = {}
    for tgt in cfg.targets:
        tr = M.frame_for(index, "mura", tgt, "train", cfg.label_col)
        n0 = len(tr)
        tr = mskdata.drop_source_patients(index, tr, cfg.source, "train")
        M.assert_probe_pool_clean(tr, index, cfg.source, "train")
        va = M.frame_for(index, "mura", tgt, "valid", cfg.label_col)
        pools[tgt] = {"train": tr, "valid": va}
        print(f"[pool] {tgt:<12s} train {n0} -> {len(tr)} after leak control "
              f"({tr.study.nunique()} studies, prev {tr.label.mean():.3f}) | "
              f"valid {len(va)} ({va.study.nunique()} studies)")

    jobs = [j for j in build_jobs(cfg)
            if tuple(j[k] for k in CELL_KEYS) not in done]
    print(f"\n[plan] {len(jobs)} cells to run\n")

    checked = set()
    for n, job in enumerate(jobs, 1):
        bb, tgt, seed = job["backbone"], job["target"], job["seed"]
        depth, direction, prep = job["depth"], job["direction"], job["prep"]
        t0 = time.time()

        src_cp = ckpt_path(cfg.source_ckpt_dir, cfg.source, bb, seed)
        if not Path(src_cp).exists():
            raise FileNotFoundError(
                f"missing source checkpoint {src_cp}. Stage A source models are "
                f"expected under --source-ckpt-dir.")
        ck = torch.load(src_cp, map_location="cpu", weights_only=False)
        if ck.get("backbone") != bb:
            raise ValueError(f"{Path(src_cp).name} holds {ck.get('backbone')!r}")

        M.set_seed(seed)                       # BEFORE construction
        model = M.build_model(bb)
        model.load_state_dict(ck["state_dict"])
        if cfg.head_init == "fresh":
            model.head = nn.Linear(model.feat_dim, 1)   # comparable to the probe
        model = model.to(device)

        frozen, n_trainable = LF.apply_freezing(model, bb, depth, direction)

        key = (bb, depth, direction)
        if key not in checked:
            chk = LF.assert_freezing_correct(model, frozen, n_trainable)
            print(f"  [freeze-check] {M.BACKBONE_SHORT[bb]} d{depth} {direction}: "
                  f"trainable={chk['n_trainable']:,} frozen_bn={chk['n_frozen_bn']}")
            checked.add(key)

        va = pools[tgt]["valid"]
        # budget subsample is seed-dependent, so it happens per cell
        tr = subsample_studies(pools[tgt]["train"], cfg.max_train_studies, seed)
        hist = finetune(model, frozen, tr, device, cfg.epochs, cfg.batch_size,
                        cfg.lr, cfg.weight_decay, cfg.num_workers, cfg.img_size,
                        prep, cfg.head_lr_mult)
        pr = predict(model, va, device, cfg.eval_batch_size, cfg.num_workers,
                     cfg.img_size, prep)
        e = eval_block(pr["logit"], pr["label"], pr["study"], float(tr.label.mean()))

        row = dict(job)
        row.update({
            "norm_family": M.NORM_FAMILY[bb], "n_trainable": n_trainable,
            "n_frozen_bn": sum(1 for p in frozen for m in p.modules()
                               if isinstance(m, nn.modules.batchnorm._BatchNorm)),
            "n_train_images": len(tr), "n_train_studies": int(tr.study.nunique()),
            "train_prev": float(tr.label.mean()),
            "head_lr_mult": cfg.head_lr_mult,
            "auc_study": e["auc_study"], "auc_image": e["auc_image"],
            "ece": e["ece"], "bacc_50": e["bacc_50"],
            "sens_50": e["sens_50"], "spec_50": e["spec_50"],
            "n_studies": e["n_studies"], "n_images": e["n_images"],
            "target_prev": e["target_prev"], "final_loss": hist[-1]["loss"],
            "epochs": cfg.epochs, "minutes": round((time.time() - t0) / 60, 2),
        })
        res.write(row)
        print(f"  [{n}/{len(jobs)}] {M.BACKBONE_SHORT[bb]} {tgt} {prep} "
              f"{direction} d{depth} s{seed}  auc={e['auc_study']:.4f} "
              f"loss={hist[-1]['loss']:.4f}  {row['minutes']:.1f}m", flush=True)

        del model
        torch.cuda.empty_cache()
    print("\n[run] complete", flush=True)


# =================================================================== report ==

def report(cfg) -> None:
    p = Path(cfg.out_dir) / "layerwise_cells.csv"
    if not p.exists():
        print(f"no cells at {p}")
        return
    d = pd.read_csv(p, comment="#")
    a = load_anchors(cfg)

    d = d.merge(a, on=["backbone", "target"], how="left")
    if d.auc_zeroshot.isna().any():
        print(f"[warn] {int(d.auc_zeroshot.isna().sum())} cells lack an anchor")

    # PRIMARY metric: no denominator, cannot explode
    d["delta_auc"] = d.auc_study - d.auc_zeroshot
    # SECONDARY: only where the gap is big enough to divide by
    safe = d.gap >= cfg.min_gap
    d["recovery"] = np.where(safe, d.delta_auc / d.gap.replace(0, np.nan), np.nan)

    shared = d[d.depth.isin([0, LF.N_BLOCKS]) & (d.direction == "top_down")].copy()
    if len(shared) and "bottom_up" in set(d.direction):
        shared["direction"] = "bottom_up"
        d = pd.concat([d, shared], ignore_index=True)

    print("\n" + "=" * 78)
    print("CONSISTENCY GATE -- depth 0 (frozen encoder, head only) vs Stage A probe")
    print("compared on RAW AUC; no denominator involved")
    print("=" * 78)
    d0 = d[(d.depth == 0) & (d.prep == "standard")]
    if len(d0):
        g = (d0.groupby(["backbone", "target"])
               .agg(depth0_auc=("auc_study", "mean"), sd=("auc_study", "std"),
                    zeroshot=("auc_zeroshot", "first"),
                    probe=("auc_probe", "first"), n=("seed", "nunique")))
        g["vs_probe"] = (g.depth0_auc - g.probe).round(4)
        g["vs_zeroshot"] = (g.depth0_auc - g.zeroshot).round(4)
        print(g.round(4).to_string())
        print("\n  Stage A's probe was fitted on FROZEN FEATURES at "
              f"n={int(d0.probe_n.iloc[0])} studies;")
        print("  depth 0 here trains the head by gradient descent on the FULL")
        print("  leak-controlled pool. Differences of a few points are expected.")
        print("  |vs_probe| > ~0.05 means pipeline drift or a wrong learning rate")
        print("  for head-only training. Investigate before trusting the curve.")
    else:
        print("  no depth-0 cells yet")

    print("\n" + "=" * 78)
    print("PRIMARY -- delta AUC over zero-shot, by depth")
    print("=" * 78)
    for prep, gp in d.groupby("prep"):
        print(f"\n--- preprocessing: {prep} ---")
        print(gp.pivot_table(index=["backbone", "target", "direction"],
                             columns="depth", values="delta_auc",
                             aggfunc="mean").round(4).to_string())

    print("\n" + "=" * 78)
    print(f"SECONDARY -- recovery fraction (only where gap >= {cfg.min_gap})")
    print("=" * 78)
    rec = d[d.gap >= cfg.min_gap]
    if len(rec):
        print(rec.pivot_table(index=["backbone", "target", "direction"],
                              columns="depth", values="recovery",
                              aggfunc="mean").round(3).to_string())
    else:
        print(f"  no target has gap >= {cfg.min_gap}; use delta AUC only")

    print("\n" + "=" * 78)
    print("VERDICT -- H1 vs H2  (pre-registered, on delta AUC over depths 1-4)")
    print("=" * 78)
    std = d[d.prep == "standard"]
    for (bb, tgt), g in std.groupby(["backbone", "target"]):
        td = g[(g.direction == "top_down") & (g.depth.between(1, 4))]
        bu = g[(g.direction == "bottom_up") & (g.depth.between(1, 4))]
        if td.empty or bu.empty:
            print(f"  {M.BACKBONE_SHORT.get(bb, bb):<12s} {tgt:<12s} "
                  f"bottom_up arm not run yet")
            continue
        t, b = td.delta_auc.mean(), bu.delta_auc.mean()
        thr = cfg.verdict_thresh * float(g.gap.iloc[0])   # scaled to the cell's gap
        gap_td_bu = t - b
        verdict = ("H1  late/semantic deficiency" if gap_td_bu > thr else
                   "H2  low-level statistics" if gap_td_bu < -thr else
                   "NULL  depth carries no information (capacity artifact)")
        print(f"  {M.BACKBONE_SHORT.get(bb, bb):<12s} {tgt:<12s} "
              f"top_down {t:+.4f} | bottom_up {b:+.4f} | diff {gap_td_bu:+.4f} "
              f"| thr {thr:.4f}  -> {verdict}")

    if {"standard", "cropped"} <= set(d.prep):
        print("\n" + "=" * 78)
        print("SHORTCUT CONTROL -- standard vs cropped preprocessing")
        print("=" * 78)
        c = d.pivot_table(index=["backbone", "target", "direction", "depth"],
                          columns="prep", values="delta_auc", aggfunc="mean")
        c["delta"] = (c["cropped"] - c["standard"]).round(4)
        print(c.round(4).to_string())
        print("\n  A large change in the BOTTOM_UP arm means the early-block signal")
        print("  was preprocessing artifact, not anatomy statistics.")

    print("\n" + "=" * 78)
    print("CAPACITY COVARIATE -- the two arms are NOT matched")
    print("=" * 78)
    print(d.groupby(["backbone", "direction", "depth"])
           .n_trainable.mean().astype("Int64").to_string())

    d.to_csv(Path(cfg.out_dir) / "layerwise_report.csv", index=False)


# ===================================================================== main ==

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--source-ckpt-dir",
                    default=str(Path(M.DEFAULT_CKPT) / "source_models"))
    ap.add_argument("--source-models-csv",
                    default=str(Path(M.DEFAULT_RESULTS) / "source_models"
                                / "anatomy_transfer_arms_merged.csv"))
    ap.add_argument("--probe-glob",
                    default=str(Path(M.DEFAULT_RESULTS) / "source_models" / "gpu*"
                                / "anatomy_transfer_probes.csv"))
    ap.add_argument("--out-dir", default=str(Path(M.DEFAULT_RESULTS) / "layerwise"))
    ap.add_argument("--source", default="XR_WRIST")
    ap.add_argument("--targets", nargs="+",
                    default=["XR_SHOULDER", "XR_HUMERUS", "XR_HAND"])
    ap.add_argument("--backbones", nargs="+", default=["densenet121"],
                    choices=list(M.BACKBONES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--directions", nargs="+", default=["top_down"],
                    choices=list(LF.DIRECTIONS))
    ap.add_argument("--prep", nargs="+", default=["standard"],
                    choices=["standard", "cropped"])
    ap.add_argument("--head-init", default="fresh", choices=["fresh", "source"])
    ap.add_argument("--label-col", default="label_abnormal")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=1.0,
                    help="head lr = --lr * this. Applied at EVERY depth so it is "
                         "uniform, not a per-depth confound. Calibrate at depth 0 "
                         "against the Stage A probe AUC before running the screen.")
    ap.add_argument("--max-train-studies", type=int, default=-1,
                    help="cap the target training pool at ~N studies, sampled by "
                         "patient and by seed. -1 uses everything. At the full "
                         "budget the readout alone recovers 90.7%% of the "
                         "wrist->shoulder gap, leaving the depth curve no headroom.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min-gap", type=float, default=0.10,
                    help="below this (ceiling - zeroshot), recovery fraction is NaN")
    ap.add_argument("--verdict-thresh", type=float, default=0.25,
                    help="H1/H2 threshold as a FRACTION of the cell's gap")
    ap.add_argument("--zeroshot-arm", default=None)
    ap.add_argument("--ceiling-arm", default=None)
    ap.add_argument("--capacity", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    cfg = ap.parse_args()

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    if cfg.capacity:
        rows = []
        for bb in cfg.backbones:
            m = M.build_model(bb)
            info = LF.verify_partition(m, bb)
            print(f"{bb:<32} trunk={info['total_trunk']:>12,} "
                  f"head={info['head']:,}  PARTITION OK")
            print(f"    per-block: {[format(n, ',') for n in info['per_block']]}")
            rows += LF.capacity_table(lambda a=bb: M.build_model(a), bb)
            del m
        t = pd.DataFrame(rows)
        print()
        print(t.pivot_table(index=["arch", "depth"], columns="direction",
                            values="trainable_params").to_string())
        t.to_csv(Path(cfg.out_dir) / "layerwise_capacity.csv", index=False)
        return

    if cfg.inspect:
        d = pd.read_csv(cfg.source_models_csv, comment="#", low_memory=False)
        print(d.groupby("condition").agg(n=("auc_study", "size"),
                                         mean_auc=("auc_study", "mean"))
               .round(4).to_string())
        sub = d[(d.source == cfg.source) & (d.target != cfg.source)]
        print(f"\nsource == {cfg.source}, target != source:")
        print(sub.groupby("condition").agg(n=("auc_study", "size"),
                                           mean_auc=("auc_study", "mean"))
                 .round(4).to_string())
        return

    if cfg.reference:
        a = load_anchors(cfg)
        a["probe_recovery"] = np.where(a.gap >= cfg.min_gap,
                                       (a.auc_probe - a.auc_zeroshot) / a.gap, np.nan)
        print("\nper-cell reference (source = %s):" % cfg.source)
        print(a.round(4).to_string(index=False))
        print("\nTargets sorted by gap (bigger = better SNR for a depth curve):")
        print(a.groupby("target").gap.agg(["min", "mean"]).round(4)
               .sort_values("min", ascending=False).to_string())
        a.to_csv(Path(cfg.out_dir) / "layerwise_reference.csv", index=False)
        return

    if cfg.report_only:
        report(cfg)
        return

    M.raise_fd_limit()
    M.configure_sharing()
    prov = M.provenance({
        "script": "layerwise_unfreezing.py v2", "source": cfg.source,
        "targets": list(cfg.targets), "backbones": list(cfg.backbones),
        "seeds": list(cfg.seeds), "depths": list(cfg.depths),
        "directions": list(cfg.directions), "prep": list(cfg.prep),
        "head_init": cfg.head_init, "epochs": cfg.epochs, "lr": cfg.lr,
        "head_lr_mult": cfg.head_lr_mult,
        "max_train_studies": cfg.max_train_studies,
        "min_gap": cfg.min_gap, "primary_metric": "delta_auc",
        "leak_control": "mskdata.drop_source_patients",
    })
    run(cfg, prov)
    report(cfg)


if __name__ == "__main__":
    main()
