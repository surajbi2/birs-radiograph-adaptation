"""Gate G2: check that frozen foundation-encoder features are reproducible.

Extracts features twice for 10 MURA wrist validation images and requires the two
passes to agree to 1e-5. Preprocessing is the shared pipeline up to resizing
(mskdata.load_rgb_uint8: 0.5/99.5 percentile rescale, grey replicated to three
channels), followed by each model's own resize and normalisation without centre
cropping:

    RAD-DINO   bicubic resize straight to 518 x 518 (its processor's crop disabled)
    SKELEX     224 x 224 bilinear, ImageNet mean and standard deviation
    MedSigLIP  448 x 448 bilinear, mean and standard deviation 0.5
    DINOv3     224 x 224 bilinear, its processor's statistics

Output:
    g2_summary<tag>.json

Usage:
    python feature_reproducibility_gate.py --models rad-dino,skelex

Requires a GPU and the encoder weights in the Hugging Face cache.
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch

CODE = Path.home() / "diagnosis" / "code"
sys.path.insert(0, str(CODE))
import mskdata                      # noqa: E402  (read-only)
import mskmodel as M                # noqa: E402  (read-only)
import pandas as pd                 # noqa: E402

HUB = Path(os.environ["HF_HOME"]) / "hub"
SNAP = {
    "rad-dino": HUB / "models--microsoft--rad-dino/snapshots/110cbc18d5133582e320b43d53bf5c44e410c936",
    "skelex": HUB / "models--skhoha--SKELEX/snapshots/368cae7b05cf649e6dbcddae9a7f00ea4b14bb8e",
    "medsiglip": HUB / "models--google--medsiglip-448/snapshots/9cea28a1a1195f665105faa6e8544c112fd960a4",
    "dinov3": HUB / "models--facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/5931719e67bbdb9737e363e781fb0c67687896bc",
}
OUT = Path("/path/to/foundation_arm_workdir/g2")
N_IMG = 10


# ------------------------------------------------------------ images -------
def wrist_valid_paths(n: int = N_IMG):
    index = pd.read_csv(M.DEFAULT_INDEX, comment="#", low_memory=False)
    f = M.frame_for(index, "mura", "XR_WRIST", "valid")
    return sorted(f["path"].tolist())[:n], len(f)


# ------------------------------------------------------- preprocessing -----
def paper_rgb(paths):
    """Paper pipeline up to resizing: HxWx3 uint8, exactly mskdata.load_rgb_uint8."""
    return [mskdata.load_rgb_uint8(p) for p in paths]


def processor_for(name):
    from transformers import BitImageProcessor, SiglipImageProcessor, ViTImageProcessor
    if name == "medsiglip":
        proc = SiglipImageProcessor.from_pretrained(SNAP[name])   # image processor only; no tokenizer
        kw = dict(size={"height": 448, "width": 448},             # Siglip processor never crops
                  resample=2)                                     # analysis plan §2: bilinear; shipped config says 3
        side = 448
    elif name == "rad-dino":
        proc = BitImageProcessor.from_pretrained(SNAP[name])
        kw = dict(do_center_crop=False, size={"height": 518, "width": 518})
        side = 518
    elif name == "skelex":
        proc = ViTImageProcessor.from_pretrained(SNAP[name])
        kw = {}
        side = 224
    elif name == "dinov3":
        from transformers import AutoImageProcessor                # DINOv3ViTImageProcessorFast (only one)
        proc = AutoImageProcessor.from_pretrained(SNAP[name])
        kw = dict(size={"height": 224, "width": 224})             # shipped: 224x224 bilinear, no crop
        side = 224
    else:
        raise KeyError(name)
    return proc, kw, side


def preprocess(name, rgbs):
    proc, kw, side = processor_for(name)
    x = proc(images=rgbs, return_tensors="pt", **kw)["pixel_values"]
    assert x.shape == (len(rgbs), 3, side, side), x.shape
    cfg = {k: getattr(proc, k) for k in ("do_resize", "size", "resample", "do_center_crop",
                                          "do_rescale", "rescale_factor", "do_normalize",
                                          "image_mean", "image_std") if hasattr(proc, k)}
    cfg.update(kw)
    cfg["processor_class"] = type(proc).__name__
    if name == "medsiglip":
        cfg["resample_note"] = "resample: 2 (bilinear, FA_PLAN §2; shipped config says 3)"
    return x, cfg


# -------------------------------------------------------------- models -----
def load(name, variant):
    from transformers import AutoModel, SiglipVisionModel, ViTMAEModel, ViTModel
    if name == "medsiglip":
        m, info = SiglipVisionModel.from_pretrained(SNAP[name], torch_dtype=torch.float32,
                                                    output_loading_info=True)
    elif name in ("rad-dino", "dinov3"):
        m, info = AutoModel.from_pretrained(SNAP[name], torch_dtype=torch.float32,
                                            output_loading_info=True)
    elif variant == "vitmae":
        m, info = ViTMAEModel.from_pretrained(SNAP[name], mask_ratio=0.0,
                                              torch_dtype=torch.float32, output_loading_info=True)
        assert m.config.mask_ratio == 0.0
    elif variant == "vit":
        m, info = ViTModel.from_pretrained(SNAP[name], add_pooling_layer=False,
                                           torch_dtype=torch.float32, output_loading_info=True)
    else:
        raise KeyError(variant)
    info = {k: sorted(v) if isinstance(v, (list, set)) else v for k, v in info.items()}
    return m.eval(), info


@torch.no_grad()
def features(name, variant, model, x, dev):
    x = x.to(dev)
    kw = {}
    if variant == "vitmae":
        n_patch = (model.config.image_size // model.config.patch_size) ** 2
        kw["noise"] = torch.arange(n_patch, dtype=torch.float32, device=dev).expand(x.shape[0], n_patch)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    out = model(pixel_values=x, **kw)
    torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    if name in ("rad-dino", "medsiglip", "dinov3"):
        feats = {"pooler": out.pooler_output}
    else:
        h = out.last_hidden_state
        if variant == "vitmae":
            assert torch.equal(out.ids_restore[0].cpu(), torch.arange(h.shape[1] - 1)), "patch order not identity"
            assert int(out.mask.sum()) == 0, "tokens were masked"
        feats = {"cls": h[:, 0], "patchmean": h[:, 1:].mean(dim=1)}
    return {k: v.float().cpu().numpy() for k, v in feats.items()}, dt, out


def run_model(name, variant, x, dev):
    """Two separate model loads, one forward each; returns per-run features."""
    runs, times, info = [], [], None
    torch.cuda.reset_peak_memory_stats(dev)
    for r in range(2):
        model, info = load(name, variant)
        model.to(dev)
        _ = features(name, variant, model, x[:2], dev)          # warm-up, discarded
        f, dt, out = features(name, variant, model, x, dev)
        runs.append(f); times.append(dt)
        n_tok = tuple(out.last_hidden_state.shape)
        del model, out
        torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated(dev) / 2**20
    return runs, times, info, peak, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rad-dino,skelex")
    a = ap.parse_args()

    torch.backends.cudnn.deterministic = True     # as mskmodel.set_seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False # strict fp32
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")                  # physical GPU chosen via CUDA_VISIBLE_DEVICES
    OUT.mkdir(parents=True, exist_ok=True)

    paths, n_pool = wrist_valid_paths()
    rgbs = paper_rgb(paths)
    summary = {"n_images": len(paths), "pool": "mura/XR_WRIST/valid", "pool_size": n_pool,
               "paths": paths, "raw_shapes": [list(r.shape) for r in rgbs],
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__,
               "transformers": __import__("transformers").__version__,
               "tf32": False, "cudnn_deterministic": True, "batch": len(paths), "models": {}}

    for name in a.models.split(","):
        x, pcfg = preprocess(name, rgbs)
        np.save(OUT / f"{name}_input.npy", x.numpy())
        variants = ["vitmae", "vit"] if name == "skelex" else ["auto"]
        ms = {"preprocessing": pcfg, "input_shape": list(x.shape), "variants": {}}
        per_variant = {}
        for v in variants:
            runs, times, info, peak, n_tok = run_model(name, v, x, dev)
            per_variant[v] = runs[0]
            prefixes = {}
            for k in info.get("unexpected_keys", []):
                p = ".".join(k.split(".")[:2]) if k.startswith(("text_model.", "decoder.")) else k.split(".")[0]
                prefixes[p] = prefixes.get(p, 0) + 1
            vs = {"loading_info": info, "unexpected_prefix_counts": prefixes,
                  "last_hidden_state_shape": list(n_tok),
                  "sec_per_img_runs": [t / len(paths) for t in times],
                  "peak_gpu_mem_MiB": round(peak, 1), "features": {}}
            for feat in runs[0]:
                a0, a1 = runs[0][feat], runs[1][feat]
                d = float(np.abs(a0 - a1).max())
                np.save(OUT / f"{name}_{v}_{feat}_run1.npy", a0)
                np.save(OUT / f"{name}_{v}_{feat}_run2.npy", a1)
                vs["features"][feat] = {"dim": int(a0.shape[1]), "max_abs_diff_run1_run2": d,
                                        "pass": d < 1e-5, "img0_first5": a0[0, :5].tolist(),
                                        "finite": bool(np.isfinite(a0).all())}
            ms["variants"][v] = vs
        if name == "skelex":
            ms["vitmae_vs_vit_max_abs_diff"] = {
                feat: float(np.abs(per_variant["vitmae"][feat] - per_variant["vit"][feat]).max())
                for feat in per_variant["vitmae"]}
        summary["models"][name] = ms

    tag = "" if a.models == "rad-dino,skelex" else "_" + a.models.replace(",", "_")
    (OUT / f"g2_summary{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
