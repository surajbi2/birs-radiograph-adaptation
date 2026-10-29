#!/usr/bin/env python3
"""Preprocessing variants for the dataset-shift experiments.

Each variant changes only what the model sees; the appearance descriptors are
always measured on the standard load.

    standard   mskdata.load_rgb_uint8 (0.5/99.5 percentile window), then a
               square resize; identical to Stage A
    p0p100     plain min-max window over the whole array
    histmatch  histogram matched to the MURA wrist training images

Library module. ``python preprocessing_variants.py --selftest`` checks the transforms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mskdata
import mskmodel as M

PREPS = ("standard", "p0p100", "maskpct", "histmatch")

# Source checkpoint directory per arm. histmatch reuses the standard source by
# design: it moves the TARGET onto the source's statistics, so retraining the
# source would defeat the purpose.
PREP_SRC_DIR = {
    "standard": "source_models",
    "p0p100": "source_prep_p0p100",
    "maskpct": "source_prep_maskpct",
    "histmatch": "source_models",
}

HIST_BINS = 256
HIST_REF_MAX_IMAGES = 4000     # reference CDF is an average; 4k is far past stable


# ================================================================== loaders ==

def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def load_standard(path: str) -> np.ndarray:
    return mskdata.load_rgb_uint8(path)


def load_p0p100(path: str) -> np.ndarray:
    return _rgb(mskdata.load_gray_uint8(path, 0.0, 100.0))


def load_maskpct(path: str, p_lo: float = mskdata.P_LO,
                 p_hi: float = mskdata.P_HI, pad_tol: float = 1e-6) -> np.ndarray:
    """
    Percentile window computed over CONTENT pixels only, then applied to the
    whole array.

    The padding value is taken as the array minimum, matching how
    measure_appearance.measure_one defines `pad_frac`, so the arm and the
    descriptor agree on what padding is. If an image is essentially all padding
    the mask is abandoned and this falls back to mskdata.load_gray_uint8, so the
    arm can never be MORE degenerate than standard.
    """
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    arr = arr.astype(np.float32)

    mask = arr > (arr.min() + pad_tol)
    if mask.sum() < 64:                      # degenerate; do not invent a window
        return _rgb(mskdata.load_gray_uint8(path, p_lo, p_hi))

    lo, hi = np.percentile(arr[mask], [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return _rgb(mskdata.load_gray_uint8(path, p_lo, p_hi))

    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return _rgb((out * 255.0 + 0.5).astype(np.uint8))


def build_hist_reference(paths: List[str], seed: int = 0,
                         max_images: int = HIST_REF_MAX_IMAGES) -> np.ndarray:
    """
    Average normalised CDF over reference images, standard-loaded.

    An average of CDFs, not a CDF of pooled pixels: pooling weights big images
    more, and MURA image size varies with anatomy.
    """
    rng = np.random.default_rng(seed)
    p = list(paths)
    if len(p) > max_images:
        idx = rng.choice(len(p), size=max_images, replace=False)
        p = [p[i] for i in sorted(idx)]
    acc = np.zeros(HIST_BINS, dtype=np.float64)
    n = 0
    for q in p:
        g = mskdata.load_gray_uint8(q)
        h = np.bincount(g.ravel(), minlength=HIST_BINS).astype(np.float64)
        s = h.sum()
        if s <= 0:
            continue
        acc += np.cumsum(h / s)
        n += 1
    if n == 0:
        raise RuntimeError("histogram reference: no usable images")
    ref = acc / n
    ref[-1] = 1.0
    return ref


def hist_match_gray(gray: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    h = np.bincount(gray.ravel(), minlength=HIST_BINS).astype(np.float64)
    s = h.sum()
    if s <= 0:
        return gray
    src = np.cumsum(h / s)
    lut = np.searchsorted(ref_cdf, src, side="left").astype(np.int64)
    lut = np.clip(lut, 0, HIST_BINS - 1).astype(np.uint8)
    return lut[gray]


def make_loader_fn(prep: str, ref_cdf: Optional[np.ndarray] = None
                   ) -> Callable[[str], np.ndarray]:
    if prep == "standard":
        return load_standard
    if prep == "p0p100":
        return load_p0p100
    if prep == "maskpct":
        return load_maskpct
    if prep == "histmatch":
        if ref_cdf is None:
            raise ValueError("histmatch needs a reference CDF")
        r = np.asarray(ref_cdf, dtype=np.float64)

        def f(path: str) -> np.ndarray:
            return _rgb(hist_match_gray(mskdata.load_gray_uint8(path), r))
        return f
    raise ValueError(f"unknown prep {prep!r}; expected one of {PREPS}")


# ================================================================== dataset ==

class PrepDataset(M.MskDataset):
    """
    M.MskDataset with a swappable pixel source. The transform pipeline --
    ToPILImage, Resize((S,S)), optional flip/rotate, ToTensor, ImageNet
    normalise -- is inherited unchanged, so the only difference between arms is
    the uint8 array handed in.
    """

    def __init__(self, frame, img_size: int = 224, train_mode: bool = False,
                 loader_fn: Callable[[str], np.ndarray] = load_standard):
        super().__init__(frame, img_size, train_mode)
        self.loader_fn = loader_fn

    def __getitem__(self, idx: int):
        import torch
        r = self.frame.iloc[idx]
        rgb = self.loader_fn(r["path"])
        x = self.tf(rgb)
        y = torch.tensor(float(r["label"]), dtype=torch.float32)
        return x, y, idx


def make_loader(frame: pd.DataFrame, batch_size: int, train_mode: bool,
                num_workers: int, img_size: int, loader_fn):
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(0)
    return DataLoader(
        PrepDataset(frame, img_size, train_mode, loader_fn),
        batch_size=batch_size, shuffle=train_mode, num_workers=num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=M._worker_init,
        generator=g if train_mode else None, persistent_workers=False)


def predict(model, frame: pd.DataFrame, device, batch_size: int,
            num_workers: int, img_size: int, loader_fn) -> Dict[str, np.ndarray]:
    """Row order matches `frame`. Always fp32 -- bf16 eval quantises AUC."""
    import torch
    loader = make_loader(frame, batch_size, False, num_workers, img_size, loader_fn)
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
    return {"logit": logit, "label": label,
            "study": frame["study"].to_numpy(),
            "uid": frame["uid"].to_numpy()}


def train(frame: pd.DataFrame, backbone: str, seed: int, device, loader_fn,
          epochs: int = 10, batch_size: int = 32, lr: float = 1e-4,
          weight_decay: float = 1e-4, num_workers: int = 8,
          img_size: int = 224, warmup_frac: float = 0.05,
          grad_clip: float = 1.0):
    """
    Byte-for-byte the optimisation in mskmodel.train_model, with the loader
    swapped. Hyperparameters are NOT tuned for GRAZ: comparability with Stage A
    is worth more than a better ceiling.
    """
    import math
    import torch
    import torch.nn as nn

    M.set_seed(seed)                       # BEFORE construction; head init
    model = M.build_model(backbone).to(device)
    loader = make_loader(frame, batch_size, True, num_workers, img_size, loader_fn)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = max(1, epochs * len(loader))
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
    try:
        for ep in range(epochs):
            run, nb = 0.0, 0
            for x, y, _ in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss @ ep{ep} step{step} ({backbone}, "
                        f"seed={seed}); aborting rather than skipping")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(gn):
                    raise FloatingPointError(
                        f"non-finite grad norm @ ep{ep} step{step}; aborting")
                opt.step()
                sched.step()
                run += float(loss.item())
                nb += 1
                step += 1
            ep_loss = run / max(1, nb)
            history.append({"epoch": ep + 1, "loss": ep_loss,
                            "lr": sched.get_last_lr()[0]})
            print(f"      epoch {ep+1}/{epochs}  loss={ep_loss:.4f}", flush=True)
    finally:
        M.shutdown_loader(loader)
    return model, history


# ================================================================ reference ==

def ref_cdf_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "histmatch_ref_mura_wrist_train.npz"


def get_ref_cdf(index: pd.DataFrame, cache_dir: Path, seed: int = 0,
                force: bool = False) -> np.ndarray:
    """Reference CDF from MURA XR_WRIST TRAIN images. Cached with provenance."""
    p = ref_cdf_path(cache_dir)
    if p.exists() and not force:
        z = np.load(p, allow_pickle=True)
        return z["ref_cdf"]
    m = ((index.dataset == "mura") & (index.anatomy == "XR_WRIST")
         & (index.split == "train"))
    paths = index[m]["path"].astype(str).tolist()
    if not paths:
        raise RuntimeError("no MURA XR_WRIST train images in the index")
    print(f"[histmatch] building reference CDF from {len(paths)} "
          f"MURA-wrist train images (cap {HIST_REF_MAX_IMAGES})", flush=True)
    ref = build_hist_reference(paths, seed=seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, ref_cdf=ref,
             meta=json.dumps({"n_available": len(paths),
                              "cap": HIST_REF_MAX_IMAGES, "seed": seed,
                              "bins": HIST_BINS,
                              "source": "mura/XR_WRIST/train",
                              "prov": M.provenance({"script": "preprocessing_variants.py"})},
                             default=str))
    print(f"[histmatch] cached -> {p}", flush=True)
    return ref


# ================================================================= selftest ==

def selftest() -> None:
    ok = True

    def ck(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    rng = np.random.default_rng(0)

    # 1. histogram matching maps a source onto the reference
    a = (rng.normal(60, 12, (128, 128))).clip(0, 255).astype(np.uint8)
    b = (rng.normal(160, 30, (96, 160))).clip(0, 255).astype(np.uint8)
    hb = np.bincount(b.ravel(), minlength=HIST_BINS).astype(float)
    ref = np.cumsum(hb / hb.sum())
    ref[-1] = 1.0
    m = hist_match_gray(a, ref)
    ck("histmatch dtype", m.dtype == np.uint8)
    ck("histmatch moves the mean toward the reference",
       abs(float(m.mean()) - float(b.mean())) < abs(float(a.mean()) - float(b.mean())),
       f"(a={a.mean():.1f} -> m={m.mean():.1f}, ref={b.mean():.1f})")

    # 2. matching an image onto its OWN cdf is close to identity
    ha = np.bincount(a.ravel(), minlength=HIST_BINS).astype(float)
    ra = np.cumsum(ha / ha.sum())
    ra[-1] = 1.0
    self_m = hist_match_gray(a, ra)
    ck("histmatch onto own CDF is near-identity",
       float(np.abs(self_m.astype(int) - a.astype(int)).mean()) < 2.0,
       f"(mean |delta| = {np.abs(self_m.astype(int)-a.astype(int)).mean():.3f})")

    # 3. monotone LUT -- matching must not reorder intensities
    lut_in = np.arange(256, dtype=np.uint8).reshape(16, 16)
    lut_out = hist_match_gray(lut_in, ref)
    ck("histmatch LUT is monotone non-decreasing",
       bool(np.all(np.diff(lut_out.ravel().astype(int)) >= 0)))

    # 4. reference CDF averaging is order-independent and normalised
    ck("reference CDF ends at 1", abs(float(ref[-1]) - 1.0) < 1e-12)

    # 5. p0p100 really widens the window relative to the default
    #    (synthetic file round trip through mskdata)
    import tempfile
    from PIL import Image
    # 10,000 mid-range pixels plus 0.5% at each extreme, so the 0.5/99.5 window
    # clips exactly the outliers and the two arms MUST differ. A flat image
    # would give lo == hi, both arms fall back to min/max, and the test would
    # pass vacuously -- which is what the first version of this test did.
    body = rng.integers(200, 800, size=9900).astype(np.uint16)
    arr = np.concatenate([body,
                          np.zeros(50, dtype=np.uint16),
                          np.full(50, 65535, dtype=np.uint16)]).reshape(100, 100)
    with tempfile.TemporaryDirectory() as td:
        fp = str(Path(td) / "t.png")
        Image.fromarray(arr, mode="I;16").save(fp)
        g_std = mskdata.load_gray_uint8(fp)
        g_p0 = mskdata.load_gray_uint8(fp, 0.0, 100.0)
        ck("p0p100 differs from standard on an outlier-bearing image",
           int(g_std.mean()) != int(g_p0.mean()),
           f"(std={g_std.mean():.1f}, p0p100={g_p0.mean():.1f})")
        # Measure the crush on the BODY pixels, not the whole array: the
        # saturated outliers inflate the global sd and hide the damage.
        body_mask = (arr > 100) & (arr < 60000)
        sd_std = float(g_std.reshape(-1)[body_mask.reshape(-1)].std())
        sd_p0 = float(g_p0.reshape(-1)[body_mask.reshape(-1)].std())
        ck("p0p100 CRUSHES body contrast when saturated pixels are present "
           "-- the documented defect",
           sd_p0 < 0.25 * sd_std,
           f"(body sd: standard={sd_std:.1f}, p0p100={sd_p0:.2f}, "
           f"ratio={sd_p0/max(sd_std,1e-9):.3f})")
        ck("16-bit never touched .convert()", g_std.dtype == np.uint8)

        # maskpct on a PADDED image: the stretch must ignore the letterbox
        body = rng.integers(400, 900, size=(60, 60)).astype(np.uint16)
        padded = np.zeros((200, 200), dtype=np.uint16)
        padded[70:130, 70:130] = body
        fp2 = str(Path(td) / "pad.png")
        Image.fromarray(padded).save(fp2)
        m = load_maskpct(fp2)[..., 0]
        st = mskdata.load_gray_uint8(fp2)
        content = padded > 0
        ck("maskpct uses more of the range on the CONTENT than standard does",
           float(m[content].std()) > float(st[content].std()),
           f"(maskpct={m[content].std():.1f}, standard={st[content].std():.1f})")
        ck("maskpct leaves padding dark",
           float(m[~content].max()) <= 1.0)
        ck("maskpct falls back rather than inventing a window on a flat image",
           load_maskpct(fp).shape[:2] == g_std.shape)

    # 6. registry integrity
    ck("every prep has a source dir", set(PREPS) == set(PREP_SRC_DIR))
    ck("maskpct registered", "maskpct" in PREPS)
    ck("maskpct has its OWN source dir", PREP_SRC_DIR["maskpct"] != "source_models")
    ck("histmatch reuses the standard source",
       PREP_SRC_DIR["histmatch"] == "source_models")
    ck("p0p100 has its OWN source dir", PREP_SRC_DIR["p0p100"] != "source_models")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    print(__doc__)
