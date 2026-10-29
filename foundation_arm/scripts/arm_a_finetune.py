"""Arm A: fine-tune each foundation encoder end to end on the MURA wrist pool.

Builds ``HFBinaryNet`` (a Hugging Face vision encoder plus one linear output unit)
and trains it with the same pipeline as the ImageNet backbones
(mskmodel.MskDataset, make_loader and train_model):

    pool        MURA XR_WRIST train, label_abnormal
    pixels      mskdata.load_rgb_uint8, then the encoder's own resize and
                normalisation (no centre crop)
    augment     horizontal flip and rotation up to 10 degrees, after resizing
    loader      batch 32, shuffled, generator seeded 0 (as mskmodel.make_loader)
    seed        set before model construction, so it seeds the head
    loss        BCEWithLogitsLoss, no class weighting
    optimiser   AdamW (lr 5e-5, weight decay 1e-4), 5% linear warmup then cosine
    precision   bfloat16 autocast in training only; evaluation in float32
    clipping    gradient norm 1.0; a non-finite loss or gradient stops the run
    selection   the final epoch (no evaluation during training)

Evaluation is study-level AUC on the 237 wrist validation studies.

Outputs (per encoder and seed, under arm_a/<encoder>/seed<seed>/):
    final.pt, result.json

Usage:
    python arm_a_finetune.py --encoders skelex,rad-dino,medsiglip,dinov3 --gpu 0
    python arm_a_finetune.py --encoders dinov3 --check-preproc     # CPU check only

Requires a GPU.
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, math, os, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CODE = Path.home() / "diagnosis" / "code"
sys.path.insert(0, str(CODE))
import mskdata                      # noqa: E402  (read-only)
import mskmodel as M                # noqa: E402  (read-only)

HUB = Path(os.environ["HF_HOME"]) / "hub"
OUT_ROOT = Path("/path/to/foundation_arm_workdir/arm_a")
G2_DIR = Path("/path/to/foundation_arm_workdir/g2")
SOURCE = "XR_WRIST"
LABEL_COL = "label_abnormal"
G1_MIN_AUC = 0.70

# encoder -> (snapshot, model class, processor class, feature, side)
ENCODERS = {
    "skelex": dict(snap=HUB / "models--skhoha--SKELEX/snapshots/368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e",
                   model="ViTModel", proc="ViTImageProcessor", feature="cls", side=224),
    "rad-dino": dict(snap=HUB / "models--microsoft--rad-dino/snapshots/110cbc18d5133582e320b43d53bf5c44e410c936",
                     model="AutoModel", proc="BitImageProcessor", feature="pooler", side=518),
    "medsiglip": dict(snap=HUB / "models--google--medsiglip-448/snapshots/9cea28a1a1195f665105faa6e8544c112fd960a4",
                      model="SiglipVisionModel", proc="SiglipImageProcessor", feature="pooler", side=448,
                      resample=2),  # analysis plan §2: bilinear (Google's fine-tuning notebook); the processor ships bicubic
    # DINOv3 (gated, dinov3-license): processor resizes to 224x224 bilinear and never crops;
    # pooler_output = final-norm CLS. RoPE train-time coordinate rescale off (analysis plan §2, same recipe for all encoders).
    "dinov3": dict(snap=HUB / "models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/5931719e67bbdb9737e363e781fb0c67687896bc",
                   model="DINOv3ViTModel", proc="AutoImageProcessor", feature="pooler", side=224,
                   config_kw=dict(pos_embed_rescale=None)),
}


# ============================================================ preprocessing ===

def model_transform_cfg(name: str) -> Dict:
    """Resize/normalisation read from the model's own processor, A1 no-crop."""
    import transformers
    e = ENCODERS[name]
    if e["snap"] is None:
        raise SystemExit(f"{name}: no pinned snapshot yet")
    proc = getattr(transformers, e["proc"]).from_pretrained(e["snap"])
    # MedSigLIP stores 0.00392156862 (1/255 truncated): <2e-9 per pixel, accepted
    if getattr(proc, "do_rescale", True) is False or abs(getattr(proc, "rescale_factor", 1 / 255) - 1 / 255) > 1e-9:
        raise SystemExit(f"{name}: processor rescale is not 1/255; ToTensor would differ")
    return {"side": e["side"], "resample": int(e.get("resample", proc.resample)),
            "resample_source": "FA_PLAN override" if "resample" in e else "processor",
            "mean": [float(v) for v in proc.image_mean], "std": [float(v) for v in proc.image_std],
            "processor_class": type(proc).__name__, "center_crop": False}


def build_tf(tcfg: Dict, train_mode: bool):
    """Paper's MskDataset transform with the model's size, filter and mean/std."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    pil2tv = {0: InterpolationMode.NEAREST, 2: InterpolationMode.BILINEAR,
              3: InterpolationMode.BICUBIC, 1: InterpolationMode.LANCZOS}
    aug = ([transforms.RandomHorizontalFlip(p=0.5),      # NEVER vertical
            transforms.RandomRotation(degrees=10)] if train_mode else [])
    return transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((tcfg["side"], tcfg["side"]), interpolation=pil2tv[tcfg["resample"]])] + aug +
        [transforms.ToTensor(),
         transforms.Normalize(mean=tcfg["mean"], std=tcfg["std"])])


class FADataset:
    """mskmodel.MskDataset with a model-specific transform. Same pixels, same items."""

    def __init__(self, frame: pd.DataFrame, tcfg: Dict, train_mode: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.tf = build_tf(tcfg, train_mode)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        r = self.frame.iloc[idx]
        rgb = mskdata.load_rgb_uint8(r["path"])          # HxWx3 uint8
        x = self.tf(rgb)
        y = torch.tensor(float(r["label"]), dtype=torch.float32)
        return x, y, idx


def make_loader(frame, tcfg, batch_size, train_mode, num_workers=8):
    """mskmodel.make_loader, dataset swapped."""
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(0)
    return DataLoader(
        FADataset(frame, tcfg, train_mode),
        batch_size=batch_size, shuffle=train_mode, num_workers=num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=M._worker_init,
        generator=g if train_mode else None, persistent_workers=False)


# ==================================================================== model ===

class HFBinaryNet(nn.Module):
    """HF encoder + one linear output unit. Same forward contract as M.build_model."""

    def __init__(self, name: str):
        super().__init__()
        import transformers
        e = ENCODERS[name]
        self.backbone_name = name
        self.feature = e["feature"]
        kw = {"add_pooling_layer": False} if e["model"] == "ViTModel" else {}
        kw.update(e.get("config_kw", {}))
        self.trunk, info = getattr(transformers, e["model"]).from_pretrained(
            e["snap"], dtype=torch.float32, output_loading_info=True, **kw)
        if info["missing_keys"] or info["mismatched_keys"]:
            raise RuntimeError(f"{name}: missing {info['missing_keys']} mismatched {info['mismatched_keys']}")
        self.loading_info = {k: len(v) for k, v in info.items()}
        cfg = self.trunk.config
        self.feat_dim = int(getattr(cfg, "hidden_size", None) or cfg.vision_config.hidden_size)
        self.head = nn.Linear(self.feat_dim, 1)

    def forward(self, x, return_features: bool = False):
        out = self.trunk(pixel_values=x)
        f = out.pooler_output if self.feature == "pooler" else out.last_hidden_state[:, 0]
        z = self.head(f).squeeze(-1)
        return (z, f) if return_features else z


# ================================================================= training ===

def train(frame, name, tcfg, seed, device, epochs=10, batch_size=32, accum=1, lr=5e-5,
          weight_decay=1e-4, num_workers=8, warmup_frac=0.05, grad_clip=1.0, max_steps=None):
    """mskmodel.train_model with an HF backbone, bf16 autocast and optional accumulation."""
    if batch_size % accum:
        raise SystemExit("batch_size must be divisible by accum")
    micro = batch_size // accum
    M.set_seed(seed)                       # BEFORE construction
    model = HFBinaryNet(name).to(device)
    loader = make_loader(frame, tcfg, micro, True, num_workers)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")   # explicit; no pos_weight
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(loader) / accum)
    total = max(1, epochs * steps_per_epoch)
    warm = max(1, int(warmup_frac * total))

    def sched_fn(step: int) -> float:
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, sched_fn)
    history: List[Dict] = []
    model.train()
    step = 0
    t_start = time.time()
    try:
        for ep in range(epochs):
            run, nb, t_ep = 0.0, 0, time.time()
            opt.zero_grad(set_to_none=True)
            for i, (x, y, _) in enumerate(loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(x)
                loss = criterion(z.float(), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss @ epoch {ep} step {step} ({name}, seed={seed})")
                (loss / accum).backward()
                run += float(loss.item())
                nb += 1
                last = (i + 1) == len(loader)
                if (i + 1) % accum and not last:
                    continue
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(gn):
                    raise FloatingPointError(f"non-finite grad norm @ epoch {ep} step {step} ({name}, seed={seed})")
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if max_steps and step >= max_steps:
                    break
            history.append({"epoch": ep + 1, "loss": run / max(1, nb), "lr": sched.get_last_lr()[0],
                            "optimizer_steps": step, "minutes": round((time.time() - t_ep) / 60, 3)})
            print(f"      epoch {ep+1}/{epochs}  loss={history[-1]['loss']:.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
    finally:
        M.shutdown_loader(loader)
    timing = {"train_seconds": time.time() - t_start, "optimizer_steps": step,
              "sec_per_step": (time.time() - t_start) / max(1, step),
              "peak_gpu_mem_MiB": torch.cuda.max_memory_allocated(device) / 2**20}
    return model, history, timing


@torch.no_grad()
def predict(model, frame, tcfg, device, batch_size=64, num_workers=8):
    """mskmodel.predict, dataset swapped. ALWAYS fp32."""
    loader = make_loader(frame, tcfg, batch_size, False, num_workers)
    model.eval()
    n = len(frame)
    logit = np.full(n, np.nan, dtype=np.float64)
    label = np.full(n, -1, dtype=np.int64)
    try:
        for x, y, idx in loader:
            z = model(x.to(device, non_blocking=True))
            i = idx.numpy()
            logit[i] = z.float().cpu().numpy().astype(np.float64)
            label[i] = y.numpy().astype(np.int64)
    finally:
        M.shutdown_loader(loader)
    if not np.all(np.isfinite(logit)):
        raise FloatingPointError("non-finite logits during inference")
    if (label < 0).any():
        raise RuntimeError("some rows were never visited by the loader")
    return logit, label


def study_auc(logit, label, frame):
    p = M.sigmoid(logit)
    _, sp, sy = M.aggregate_studies(p, label, frame["study"].to_numpy())
    return {"auc_study": M.safe_auc(sy, sp), "auc_image": M.safe_auc(label, p),
            "n_images": int(len(label)), "n_studies": int(len(sy))}


# =================================================================== checks ===

def check_preproc(name, tcfg):
    """Eval transform vs the G2 processor tensors (same 10 images). CPU only."""
    ref = np.load(G2_DIR / f"{name}_input.npy")
    paths = json.loads((G2_DIR / "g2_summary.json").read_text())["paths"]
    tf = build_tf(tcfg, train_mode=False)
    x = np.stack([tf(mskdata.load_rgb_uint8(p)).numpy() for p in paths])
    d = float(np.abs(x - ref).max())
    print(f"[check-preproc] {name}: shape {x.shape} max|torchvision - HF processor| = {d:.3g}")
    return d


# ===================================================================== main ===

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="skelex,rad-dino,medsiglip")
    ap.add_argument("--seeds", default=",".join(str(s) for s in M.SEEDS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="grad accumulation; only if 32 does not fit")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--smoke", action="store_true", help="50 optimizer steps, seed 42, no checkpoint")
    ap.add_argument("--check-preproc", action="store_true", help="compare eval transform with G2 inputs and exit")
    cfg = ap.parse_args()
    encoders = [e.strip() for e in cfg.encoders.split(",") if e.strip()]
    seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]

    if cfg.check_preproc:
        for name in encoders:
            check_preproc(name, model_transform_cfg(name))
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{cfg.gpu}")
    torch.cuda.set_device(device)          # initialise CUDA before reset_peak_memory_stats
    M.raise_fd_limit()
    M.configure_sharing()
    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    tr = M.frame_for(index, "mura", SOURCE, "train", LABEL_COL)
    va = M.frame_for(index, "mura", SOURCE, "valid", LABEL_COL)
    if cfg.smoke:
        seeds = seeds[:1]
    print(f"[data] train {len(tr)} imgs / {tr.study.nunique()} studies, prev {tr.label.mean():.3f}; "
          f"valid {len(va)} imgs / {va.study.nunique()} studies", flush=True)

    for name in encoders:
        tcfg = model_transform_cfg(name)
        for seed in seeds:
            od = OUT_ROOT / "smoke" / name if cfg.smoke else OUT_ROOT / name / f"seed{seed}"
            ck, js = od / "final.pt", od / "result.json"
            if js.exists() and not cfg.smoke:
                print(f"[skip] {js}", flush=True)
                continue
            od.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats(device)
            print(f"[train] {name} | seed {seed} | {tcfg}", flush=True)
            model, hist, timing = train(tr, name, tcfg, seed, device, epochs=cfg.epochs,
                                        batch_size=cfg.batch_size, accum=cfg.accum, lr=cfg.lr,
                                        weight_decay=cfg.weight_decay, num_workers=cfg.num_workers,
                                        max_steps=50 if cfg.smoke else None)
            t0 = time.time()
            logit, label = predict(model, va, tcfg, device, cfg.eval_batch_size, cfg.num_workers)
            ev = study_auc(logit, label, va)
            ev["eval_seconds"] = time.time() - t0
            prov = M.provenance({"script": str(Path(__file__).resolve()), "encoder": name,
                                 "snapshot": str(ENCODERS[name]["snap"]),
                                 "transformers": __import__("transformers").__version__})
            res = {"encoder": name, "seed": seed, "source": SOURCE, "label_col": LABEL_COL,
                   "smoke": cfg.smoke, "selection": "final_epoch",
                   "hparams": {"epochs": cfg.epochs, "batch_size": cfg.batch_size, "accum": cfg.accum,
                               "micro_batch": cfg.batch_size // cfg.accum, "lr": cfg.lr,
                               "weight_decay": cfg.weight_decay, "warmup_frac": 0.05, "schedule": "cosine",
                               "grad_clip": 1.0, "train_precision": "bf16 autocast",
                               "eval_precision": "fp32", "tf32": False},
                   "transform": tcfg, "loading_info": model.loading_info, "feat_dim": model.feat_dim,
                   "n_train_images": len(tr), "n_train_studies": int(tr.study.nunique()),
                   "source_prev": float(tr.label.mean()),
                   "history": hist, "timing": timing, "valid": ev, "provenance": prov}
            if not cfg.smoke:
                torch.save({"state_dict": model.state_dict(), "encoder": name, "seed": seed,
                            "history": hist, "transform": tcfg, "provenance": prov}, ck)
                res["checkpoint"] = str(ck)
            js.write_text(json.dumps(res, indent=2))
            print(f"[done] {name} seed {seed}: study AUC {ev['auc_study']:.4f} -> {js}", flush=True)
            del model
            torch.cuda.empty_cache()
            if not cfg.smoke and not (ev["auc_study"] >= G1_MIN_AUC):
                print(f"[G1 FAIL] {name} seed {seed}: study AUC {ev['auc_study']:.4f} < {G1_MIN_AUC}; stopping",
                      flush=True)
                sys.exit(2)


if __name__ == "__main__":
    main()
