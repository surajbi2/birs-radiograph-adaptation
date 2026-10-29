#!/usr/bin/env python3
"""Models, training, prediction and metrics shared by every stage.

Backbones are timm models built with ``num_classes=0`` and wrapped in
``BinaryNet`` (``trunk`` + a single linear output unit, ``head``). The module
provides the dataset and loader (``MskDataset``, ``make_loader``), training
(``train_model``), prediction with optional frozen-feature export, study-level
aggregation (a study scores the mean of its image probabilities and is positive
if any of its images is) and AUC helpers. It draws no conclusions; the stage
scripts call it and write their own result files.

Default locations, relative to the project root (``~/diagnosis``):
    data/diag_data/index.csv    image index written by mskdata.py
    checkpoints/                trained source models
    results/                    result CSVs and per-image dumps
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mskdata  # noqa: E402

# ------------------------------------------------------------------ config ---

BACKBONES: Tuple[str, ...] = (
    "densenet121",                      # CNN + BatchNorm
    "resnet50",                         # CNN + BatchNorm
    "convnext_tiny",                    # CNN + LayerNorm   <- de-confounds
    "swin_tiny_patch4_window7_224",     # ViT + LayerNorm
)

BACKBONE_SHORT = {
    "densenet121": "DenseNet121",
    "resnet50": "ResNet50",
    "convnext_tiny": "ConvNeXt-T",
    "swin_tiny_patch4_window7_224": "Swin-T",
}

NORM_FAMILY = {                       # for the architectural BN test
    "densenet121": "batchnorm",
    "resnet50": "batchnorm",
    "convnext_tiny": "layernorm",
    "swin_tiny_patch4_window7_224": "layernorm",
}

SEEDS: Tuple[int, ...] = (42, 123, 456, 789, 2024)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_HOME = Path.home() / "diagnosis"
DEFAULT_INDEX = DEFAULT_HOME / "data" / "diag_data" / "index.csv"
DEFAULT_CKPT = DEFAULT_HOME / "checkpoints"
DEFAULT_RESULTS = DEFAULT_HOME / "results"
DEFAULT_LOGS = DEFAULT_HOME / "logs"


# ============================================================== provenance ===

def provenance(extra: Optional[Dict] = None) -> Dict[str, object]:
    # The script and code hashes come from mskdata.code_hashes(), so every
    # header records which version of which stage wrote the file. Older
    # result files record only this module's hash (see PROVENANCE.md).
    me = Path(__file__).resolve()
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=me.parent,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "NO_GIT"
    try:
        import torch
        tv, cv = torch.__version__, (torch.version.cuda or "cpu")
    except Exception:
        tv, cv = "NA", "NA"
    try:
        import timm
        timmv = timm.__version__
    except Exception:
        timmv = "NA"
    p = {**mskdata.code_hashes(), "git_sha": git,
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "python": sys.version.split()[0], "platform": platform.platform(),
         "torch": tv, "cuda": cv, "timm": timmv,
         "numpy": np.__version__, "pandas": pd.__version__,
         "mskdata_P_LO": str(mskdata.P_LO), "mskdata_P_HI": str(mskdata.P_HI)}
    if extra:
        p.update({k: str(v) for k, v in extra.items()})
    return p


# ========================================================= reproducibility ===

def set_seed(seed: int) -> None:
    """MUST be called before build_model(). Head init is seed-controlled."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _worker_init(worker_id: int) -> None:
    import torch
    s = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def raise_fd_limit() -> Tuple[int, int]:
    try:
        import resource
    except ImportError:
        return (-1, -1)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    tgt = hard if hard != resource.RLIM_INFINITY else 65536
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(tgt, 65536)), hard))
    except (ValueError, OSError):
        pass
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def configure_sharing() -> str:
    try:
        import torch.multiprocessing as mp
        mp.set_sharing_strategy("file_system")
        return mp.get_sharing_strategy()
    except Exception as e:
        return f"unchanged ({e})"


def shutdown_loader(loader) -> None:
    """Reap workers explicitly. GC is not reliable enough across ~1000 loaders."""
    import gc
    if loader is None:
        return
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            if hasattr(it, "_shutdown_workers"):
                it._shutdown_workers()
            loader._iterator = None
    except Exception:
        pass
    del loader
    gc.collect()


# ==================================================================== data ===

class MskDataset:
    """
    Pixels ALWAYS come from mskdata.load_rgb_uint8 (percentile-rescaled,
    dataset-agnostic), never from a raw PIL convert.
    """

    def __init__(self, frame: pd.DataFrame, img_size: int = 224,
                 train_mode: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.img_size = img_size
        self.train_mode = train_mode
        from torchvision import transforms
        aug = ([transforms.RandomHorizontalFlip(p=0.5),      # NEVER vertical
                transforms.RandomRotation(degrees=10)] if train_mode else [])
        self.tf = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((img_size, img_size))] + aug +
            [transforms.ToTensor(),
             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        import torch
        r = self.frame.iloc[idx]
        rgb = mskdata.load_rgb_uint8(r["path"])          # HxWx3 uint8
        x = self.tf(rgb)
        y = torch.tensor(float(r["label"]), dtype=torch.float32)
        return x, y, idx


def make_loader(frame: pd.DataFrame, batch_size: int, train_mode: bool,
                num_workers: int = 8, img_size: int = 224):
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(0)
    return DataLoader(
        MskDataset(frame, img_size, train_mode),
        batch_size=batch_size, shuffle=train_mode, num_workers=num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=_worker_init,
        generator=g if train_mode else None,
        persistent_workers=False,           # ~1000 loaders; workers must be reaped
    )


def frame_for(index: pd.DataFrame, dataset: str, anatomy: str, split: str,
              label_col: str = "label_abnormal") -> pd.DataFrame:
    """Select a pool and expose the chosen label column as 'label'."""
    m = ((index.dataset == dataset) & (index.anatomy == anatomy)
         & (index.split == split))
    f = index[m].copy()
    if f.empty:
        raise ValueError(f"empty pool: {dataset}/{anatomy}/{split}")
    if label_col not in f.columns:
        raise ValueError(f"missing label column {label_col}")
    f["label"] = pd.to_numeric(f[label_col], errors="coerce")
    n_bad = int(f["label"].isna().sum())
    if n_bad:
        raise ValueError(f"{n_bad} rows have NaN in {label_col} "
                         f"for {dataset}/{anatomy}/{split}")
    f["label"] = f["label"].astype(int)
    return f


def assert_probe_pool_clean(pool: pd.DataFrame, index: pd.DataFrame,
                            source_anatomy: str, source_split: str = "train") -> None:
    """
    A probe pool drawn from target-train MUST NOT contain anyone who appears in
    the source training pool. MURA anatomy pairs share 16-249 people.
    """
    seen = set(index[(index.anatomy == source_anatomy)
                     & (index.split == source_split)]["patient_group"])
    bad = set(pool["patient_group"]) & seen
    if bad:
        raise AssertionError(
            f"probe pool shares {len(bad)} people with {source_anatomy}/"
            f"{source_split}, e.g. {sorted(bad)[:3]}. Filter with "
            f"mskdata.drop_source_patients() before probing.")


# =================================================================== model ===

def build_model(backbone: str):
    """
    timm trunk (num_classes=0 -> pooled features) + a fresh binary head.
    One forward path for logits and features -- a hand-written extract_features
    can silently diverge from the real forward pass.

    No method here is named `_apply`; that name collides with nn.Module
    internals and breaks .to(device).
    """
    import timm
    import torch.nn as nn

    class BinaryNet(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.backbone_name = name
            self.trunk = timm.create_model(name, pretrained=True, num_classes=0)
            self.feat_dim = self.trunk.num_features
            self.head = nn.Linear(self.feat_dim, 1)

        def forward(self, x, return_features: bool = False):
            f = self.trunk(x)
            z = self.head(f).squeeze(-1)
            return (z, f) if return_features else z

    return BinaryNet(backbone)


def count_bn_modules(model) -> int:
    import torch.nn as nn
    return sum(1 for m in model.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))


# ================================================================ training ===

def train_model(frame: pd.DataFrame, backbone: str, seed: int, device,
                epochs: int = 10, batch_size: int = 32, lr: float = 1e-4,
                weight_decay: float = 1e-4, num_workers: int = 8,
                img_size: int = 224, warmup_frac: float = 0.05,
                grad_clip: float = 1.0, log_every: int = 1):
    import torch
    import torch.nn as nn

    set_seed(seed)                       # BEFORE construction
    model = build_model(backbone).to(device)
    loader = make_loader(frame, batch_size, True, num_workers, img_size)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")   # explicit; no pos_weight
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
                loss = criterion(model(x), y)          # fp32; no autocast
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss @ epoch {ep} step {step} "
                        f"({backbone}, seed={seed}); aborting rather than skipping")
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(gn):
                    raise FloatingPointError(
                        f"non-finite grad norm @ epoch {ep} step {step} "
                        f"({backbone}, seed={seed}); aborting")
                opt.step()
                sched.step()
                run += float(loss.item())
                nb += 1
                step += 1
            ep_loss = run / max(1, nb)
            history.append({"epoch": ep + 1, "loss": ep_loss,
                            "lr": sched.get_last_lr()[0]})
            if log_every and (ep + 1) % log_every == 0:
                print(f"      epoch {ep+1}/{epochs}  loss={ep_loss:.4f}", flush=True)
    finally:
        shutdown_loader(loader)

    return model, history


# ============================================================== prediction ===

def predict(model, frame: pd.DataFrame, device, batch_size: int = 64,
            num_workers: int = 8, img_size: int = 224,
            want_features: bool = False) -> Dict[str, np.ndarray]:
    """
    ALWAYS fp32. bf16/fp16 eval quantises probabilities to 2-3 decimal digits,
    manufacturing AUC ties and putting a floor under ECE.
    Row order matches `frame`.
    """
    import torch
    loader = make_loader(frame, batch_size, False, num_workers, img_size)
    model.eval()
    n = len(frame)
    logit = np.full(n, np.nan, dtype=np.float64)
    label = np.full(n, -1, dtype=np.int64)
    feats = None
    try:
        with torch.no_grad():
            for x, y, idx in loader:
                x = x.to(device, non_blocking=True)
                i = idx.numpy()
                if want_features:
                    z, f = model(x, return_features=True)
                    f = f.detach().float().cpu().numpy()
                    if feats is None:
                        feats = np.zeros((n, f.shape[1]), dtype=np.float32)
                    feats[i] = f
                else:
                    z = model(x)
                logit[i] = z.detach().float().cpu().numpy().astype(np.float64)
                label[i] = y.numpy().astype(np.int64)
    finally:
        shutdown_loader(loader)

    if not np.all(np.isfinite(logit)):
        raise FloatingPointError("non-finite logits during inference")
    if (label < 0).any():
        raise RuntimeError("some rows were never visited by the loader")

    out = {"logit": logit, "label": label,
           "study": frame["study"].to_numpy(),
           "patient_group": frame["patient_group"].to_numpy(),
           "uid": frame["uid"].to_numpy()}
    if want_features:
        out["feature"] = feats
    return out


# =========================================================== interventions ===

def adapt_bn(model, frame: pd.DataFrame, device, n_batches: int = 100,
             batch_size: int = 32, num_workers: int = 8, img_size: int = 224
             ) -> Dict[str, float]:
    """
    AdaBN: recompute BatchNorm running statistics on UNLABELLED target images.
    Label-free. Modifies `model` in place.

    Returns a receipt including the measured drift in running_mean. If the
    model has BatchNorm and the drift is ~0, this raises, because an
    intervention that silently does nothing would otherwise go unnoticed.

    LayerNorm backbones (ConvNeXt, Swin) have no BatchNorm; the receipt then
    reports n_bn_modules=0 and this is a defined no-op, not a failure. That
    asymmetry is itself the architectural test of the BatchNorm hypothesis.
    """
    import torch
    import torch.nn as nn

    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return {"n_bn_modules": 0, "n_batches": 0, "mean_drift": 0.0,
                "var_drift": 0.0, "applicable": 0.0}

    before = [m.running_mean.detach().clone() for m in bns]
    before_v = [m.running_var.detach().clone() for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                      # cumulative moving average

    loader = make_loader(frame, batch_size, True, num_workers, img_size)
    model.train()                              # BN updates running stats
    seen = 0
    try:
        with torch.no_grad():                  # statistics only, no gradients
            for x, _, _ in loader:
                model(x.to(device, non_blocking=True))
                seen += 1
                if seen >= n_batches:
                    break
    finally:
        shutdown_loader(loader)
    model.eval()

    md = float(np.mean([torch.norm(m.running_mean - b).item()
                        / (torch.norm(b).item() + 1e-8)
                        for m, b in zip(bns, before)]))
    vd = float(np.mean([torch.norm(m.running_var - b).item()
                        / (torch.norm(b).item() + 1e-8)
                        for m, b in zip(bns, before_v)]))

    if seen == 0:
        raise RuntimeError("adapt_bn consumed 0 batches -- pool empty or "
                           "n_batches<=0. Refusing to report a no-op as a result.")
    if md < 1e-6:
        raise RuntimeError(
            f"adapt_bn ran {seen} batches over {len(bns)} BN modules but "
            f"running_mean drift is {md:.2e} (~0). The intervention is inert; "
            "refusing to report it as tested.")

    return {"n_bn_modules": float(len(bns)), "n_batches": float(seen),
            "mean_drift": md, "var_drift": vd, "applicable": 1.0}


def linear_probe(feat_train: np.ndarray, y_train: np.ndarray,
                 feat_eval: np.ndarray, C: float = 1.0,
                 max_iter: int = 5000) -> np.ndarray:
    """
    Re-fit ONLY the readout direction on frozen features.
    This is the honest bar every proposed method must beat.
    Returns decision scores on feat_eval (NaN if train has a single class).
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y_train).astype(int)
    if len(np.unique(y)) < 2:
        return np.full(len(feat_eval), np.nan)
    mu = feat_train.mean(axis=0, keepdims=True)
    sd = feat_train.std(axis=0, keepdims=True) + 1e-6
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    clf.fit((feat_train - mu) / sd, y)
    return clf.decision_function((feat_eval - mu) / sd)


# ================================================================= metrics ===

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def safe_auc(label: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(label)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")            # explicit; never silently 0.5
    return float(roc_auc_score(y[ok], s[ok]))


def aggregate_studies(score: np.ndarray, label: np.ndarray, study: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MURA labels are per study. Mean per-image probability; label = max."""
    df = pd.DataFrame({"study": study, "p": np.asarray(score, dtype=float),
                       "y": np.asarray(label)})
    g = df.groupby("study", sort=True).agg(p=("p", "mean"), y=("y", "max"))
    return g.index.to_numpy(), g["p"].to_numpy(), g["y"].to_numpy().astype(int)


def threshold_metrics(prob: np.ndarray, label: np.ndarray, thr: float) -> Dict:
    p, y = np.asarray(prob, float), np.asarray(label, int)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return {"threshold": float(thr), "sensitivity": sens, "specificity": spec,
            "balanced_acc": 0.5 * (sens + spec), "tp": tp, "fp": fp,
            "tn": tn, "fn": fn}


def prior_corrected_threshold(prob: np.ndarray, source_prev: float,
                              target_prev: float) -> float:
    """
    LABEL-FREE prior shift correction. Needs target prevalence only, which can
    be estimated without labels. Strictly separate from oracle_threshold below.
    """
    sp = float(np.clip(source_prev, 1e-6, 1 - 1e-6))
    tp = float(np.clip(target_prev, 1e-6, 1 - 1e-6))
    ratio = (sp / (1 - sp)) * ((1 - tp) / tp)
    return float(ratio / (1.0 + ratio))


def oracle_threshold(prob: np.ndarray, label: np.ndarray) -> float:
    """
    Threshold maximising balanced accuracy, chosen using target labels.
    An upper bound on what any threshold-only correction could achieve; it is
    not deployable, so it is always reported as an oracle.
    """
    p, y = np.asarray(prob, float), np.asarray(label, int)
    if len(np.unique(y)) < 2:
        return 0.5
    cand = np.unique(np.concatenate([[0.0, 1.0], np.quantile(p, np.linspace(0, 1, 201))]))
    best_t, best_b = 0.5, -1.0
    for t in cand:
        b = threshold_metrics(p, y, t)["balanced_acc"]
        if b > best_b:
            best_b, best_t = b, float(t)
    return best_t


def expected_calibration_error(prob: np.ndarray, label: np.ndarray,
                               n_bins: int = 15) -> float:
    """Requires labels by definition: bins compare confidence to accuracy."""
    p, y = np.asarray(prob, float), np.asarray(label, int)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece, n = 0.0, len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# =================================================================== dumps ===

def save_dump(path: Path, pred: Dict[str, np.ndarray], meta: Dict) -> None:
    """
    Raw per-image predictions. Every table is derived from these offline, so a
    re-analysis never requires a re-run and a mistake in one table cannot
    silently propagate into the stored evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in pred.items() if k != "feature"}
    payload["_meta_json"] = np.array(json.dumps(meta, default=str))
    np.savez_compressed(path, **payload)


def load_dump(path: Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["_meta_json"]))
    return {k: z[k] for k in z.files if k != "_meta_json"}, meta


# ================================================================ selftest ===

def selftest() -> None:
    ok = True

    def ck(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        ok &= bool(cond)

    rng = np.random.default_rng(0)

    # --- metrics ---------------------------------------------------------
    y = rng.integers(0, 2, 4000)
    s = y + rng.normal(0, 0.6, 4000)
    ck("safe_auc informative", safe_auc(y, s) > 0.85)
    ck("safe_auc NaN on one class", math.isnan(safe_auc(np.ones(20), rng.random(20))))
    ck("safe_auc tolerates NaN scores",
       np.isfinite(safe_auc(y, np.where(rng.random(4000) < 0.1, np.nan, s))))

    # threshold cannot change AUC -- the crux of the readout-direction argument
    p = sigmoid(s)
    ck("AUC invariant to monotone rescale",
       abs(safe_auc(y, p) - safe_auc(y, sigmoid(s / 3.7))) < 1e-9)

    m = threshold_metrics(p, y, 0.5)
    ck("threshold_metrics counts consistent",
       m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y))
    ot = oracle_threshold(p, y)
    ck("oracle >= default bacc",
       threshold_metrics(p, y, ot)["balanced_acc"]
       >= threshold_metrics(p, y, 0.5)["balanced_acc"] - 1e-12)

    t = prior_corrected_threshold(p, 0.5, 0.5)
    ck("prior threshold identity at equal prev", abs(t - 0.5) < 1e-9, f"({t:.4f})")
    ck("prior threshold falls when target prev rises",
       prior_corrected_threshold(p, 0.3, 0.7) < 0.5)

    ids, sp, sy = aggregate_studies(np.array([.9, .1, .8, .2]),
                                    np.array([1, 1, 0, 0]),
                                    np.array(["a", "a", "b", "b"]))
    ck("study aggregation mean/max",
       abs(sp[0] - .5) < 1e-12 and sy[0] == 1 and sy[1] == 0)

    # calibrated by construction: labels drawn FROM the probabilities
    z_cal = rng.normal(0, 2, 40000)
    p_cal = sigmoid(z_cal)
    y_cal = (rng.random(40000) < p_cal).astype(int)
    ece_cal = expected_calibration_error(p_cal, y_cal)
    ck("ECE ~0 when calibrated", ece_cal < 0.02, f"({ece_cal:.4f})")
    ece_over = expected_calibration_error(sigmoid(z_cal * 4), y_cal)
    ck("ECE large when overconfident", ece_over > 0.10, f"({ece_over:.4f})")

    # --- linear probe ----------------------------------------------------
    d = 32
    w = rng.normal(size=d)
    ftr = rng.normal(size=(600, d)); ytr = (ftr @ w > 0).astype(int)
    fte = rng.normal(size=(400, d)); yte = (fte @ w > 0).astype(int)
    ck("linear probe recovers direction", safe_auc(yte, linear_probe(ftr, ytr, fte)) > 0.9)
    ck("linear probe NaN on single class",
       np.isnan(linear_probe(ftr, np.zeros(600, int), fte)).all())

    # --- dump round trip -------------------------------------------------
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "d.npz"
    pred = {"logit": s, "label": y, "study": np.array([f"s{i//2}" for i in range(4000)]),
            "patient_group": np.array([f"p{i//4}" for i in range(4000)]),
            "uid": np.array([f"u{i}" for i in range(4000)])}
    save_dump(tmp, pred, {"source": "XR_WRIST", "n_bn": 121})
    back, meta = load_dump(tmp)
    ck("dump round trip", np.allclose(back["logit"], s) and meta["n_bn"] == 121)

    # --- config integrity -------------------------------------------------
    ck("4 backbones, both norm families",
       len(BACKBONES) == 4 and set(NORM_FAMILY[b] for b in BACKBONES)
       == {"batchnorm", "layernorm"})
    ck("norm families balanced 2/2",
       sum(NORM_FAMILY[b] == "batchnorm" for b in BACKBONES) == 2)
    ck("mskdata linked", hasattr(mskdata, "load_rgb_uint8")
       and hasattr(mskdata, "drop_source_patients"))

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    print(__doc__)
    print(f"backbones: {[BACKBONE_SHORT[b] for b in BACKBONES]}")
    print(f"norm families: { {BACKBONE_SHORT[b]: NORM_FAMILY[b] for b in BACKBONES} }")
    print(f"seeds: {list(SEEDS)}")
    print(f"index: {DEFAULT_INDEX}")
    print("\nThis is a library. Run train_source_models.py to drive it.")
