"""Extract foundation-encoder features into the readout cache used by the cells.

Writes one file per (encoder, seed, target) in the cache format read by
l2sp_comparator.load_unit:

    <cache-dir>/features/<backbone>__s<seed>__<target>.npz
        ftr     float32 (n_pool, d)   pool features (budget samples are drawn from these)
        fva     float32 (n_eval, d)   evaluation features
        zva     float64 (n_eval,)     source-model logits on the evaluation split
        yva     int64   (n_eval,)     evaluation labels
        tr_uid, va_uid                row ids in index order
        w_src, b_src                  source readout
        zs                            zero-shot study AUC of zva

Pools come from l2sp_comparator.pools, so rows match the ImageNet cache exactly:
the six MURA calibration targets (source-training patients removed),
GRAZPEDWRI-DX initial examinations, MFXR and FracAtlas hand, plus the wrist
source pools.

    arm A   features and logits of the fine-tuned model (final.pt); its head is
            the source readout
    arm B   features of the released, frozen encoder (seed-independent, tag s0);
            the source readout is added later by arm_b_readout.py

Gate G3 (arm A, before any target is written): the study AUC of the extracted
logits, and of ``fva @ w_src + b_src``, must equal the AUC in result.json within
1e-6, and the source-readout identity of readout_shrinkage must hold on every target.

Usage:
    python extract_features.py --arms A,B --encoders skelex,rad-dino,medsiglip --gpu 0

Requires a GPU.
"""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

CODE = Path.home() / "diagnosis" / "code"
sys.path.insert(0, str(CODE))
FA = Path("/path/to/foundation_arm_workdir")
sys.path.insert(0, str(FA / "scripts"))
import mskmodel as M                # noqa: E402  (read-only)
import arm_a_finetune as A          # noqa: E402  (reviewed; sha checked below)
from l2sp_comparator import pools, cache_path        # noqa: E402  (read-only)
from readout_shrinkage import verify_source_head  # noqa: E402  (read-only)
from shortcut_reliance import score                 # noqa: E402  (read-only)

ARM_A_SHA = "d2e8ba7abe47772627dff682c5b9038cb33cba694855f72105cc7eac29949831"
# checkpoints trained by either reviewed arm_a_finetune.py (4ecb46c0: skelex, rad-dino, medsiglip)
ARM_A_TRAINED_SHAS = {ARM_A_SHA, "4ecb46c09a8b5980e08e22bebbdc89ffa54a2f3bc12ee165bf1d7631d974b53d"}
DEFAULT_CACHE = FA / "features"
PAPER_CACHE = Path("/path/to/imagenet_feature_cache")
PAPER_REF_BB = "densenet121"        # row-order check against the paper's cache
SOURCE_T = A.SOURCE                 # "XR_WRIST"
G3_TOL = 1e-6
ARM_B_SEED = 0
# arm B frozen feature per encoder (analysis plan §2); arm A uses HFBinaryNet's own feature
ARM_B_FEATURE = {"skelex": "patchmean", "rad-dino": "pooler", "medsiglip": "pooler",
                 "dinov3": "pooler"}   # DINOv3 pooler_output = final-norm CLS
G2_SUMMARY = {"skelex": ("g2_summary.json", "vit"), "rad-dino": ("g2_summary.json", "auto"),
              "medsiglip": ("g2_summary_medsiglip.json", "auto"), "dinov3": ("g2_summary_dinov3.json", "auto")}


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def snapshot_commit(name: str) -> str:
    return Path(A.ENCODERS[name]["snap"]).name


# ================================================================== models ===

class FrozenEncoder:
    """Arm B: released trunk (HFBinaryNet's loader and key checks), no head."""

    def __init__(self, name: str):
        self.feature = ARM_B_FEATURE[name]
        net = A.HFBinaryNet(name)             # raises on missing / mismatched keys
        self.trunk, self.feat_dim, self.loading_info = net.trunk, net.feat_dim, net.loading_info

    def to(self, device):
        self.trunk.to(device)
        return self

    def eval(self):
        self.trunk.eval()
        return self

    def __call__(self, x, return_features=True):
        out = self.trunk(pixel_values=x)
        if self.feature == "patchmean":
            f = out.last_hidden_state[:, 1:].mean(dim=1)
        elif self.feature == "cls":
            f = out.last_hidden_state[:, 0]
        else:
            f = out.pooler_output
        return None, f


def load_arm_a(name: str, seed: int, tcfg: Dict, device):
    import torch
    d = A.OUT_ROOT / name / f"seed{seed}"
    ck_path, res = d / "final.pt", json.loads((d / "result.json").read_text())
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)   # our own checkpoint
    if ck["encoder"] != name or int(ck["seed"]) != seed:
        raise SystemExit(f"!! {ck_path}: holds {ck['encoder']}/seed{ck['seed']}")
    if ck["transform"] != tcfg:
        raise SystemExit(f"!! {name}/s{seed}: training transform {ck['transform']} != eval transform {tcfg}")
    if res["provenance"]["script_sha256"] not in ARM_A_TRAINED_SHAS:
        raise SystemExit(f"!! {name}/s{seed}: trained with script {res['provenance']['script_sha256'][:8]}")
    model = A.HFBinaryNet(name)
    model.load_state_dict(ck["state_dict"])               # strict
    return model.to(device).eval(), res, sha256(ck_path)


# ============================================================== extraction ===

def extract(model, frame, tcfg, device, batch_size, num_workers):
    """readout_probe.extract with the FA loader. Row order matches `frame`. fp32."""
    import torch
    loader = A.make_loader(frame, tcfg, batch_size, False, num_workers)
    n = len(frame)
    feats = None
    logit = np.full(n, np.nan)
    label = np.full(n, -1, dtype=np.int64)
    try:
        with torch.no_grad():
            for x, y, idx in loader:
                z, f = model(x.to(device, non_blocking=True), return_features=True)
                i = idx.numpy()
                f = f.detach().float().cpu().numpy()
                if feats is None:
                    feats = np.zeros((n, f.shape[1]), dtype=np.float32)
                feats[i] = f
                if z is not None:
                    logit[i] = z.detach().float().cpu().numpy()
                label[i] = y.numpy().astype(np.int64)
    finally:
        M.shutdown_loader(loader)
    if (label < 0).any():
        raise RuntimeError("loader skipped rows")
    if not np.all(np.isfinite(feats)):
        raise FloatingPointError("non-finite features")
    return feats, logit, label


def g3(name, seed, fva, zva, yva, va, w_src, b_src, res) -> Dict:
    """Saved head on extracted wrist-valid features reproduces result.json."""
    st = va.study.to_numpy()
    want = float(res["valid"]["auc_study"])
    got_z = score(zva, yva, st)
    got_r = score(fva.astype(np.float64) @ w_src + b_src, yva, st)
    out = {"result_json_auc_study": want, "auc_study_extracted_logit": got_z["auc_study"],
           "auc_study_reconstructed": got_r["auc_study"],
           "auc_image_reconstructed": got_r["auc_image"],
           "result_json_auc_image": float(res["valid"]["auc_image"]),
           "max_abs_logit_diff": float(np.max(np.abs(fva.astype(np.float64) @ w_src + b_src - zva)))}
    out["delta_extracted"] = abs(got_z["auc_study"] - want)
    out["delta_reconstructed"] = abs(got_r["auc_study"] - want)
    out["pass"] = bool(out["delta_extracted"] < G3_TOL and out["delta_reconstructed"] < G3_TOL)
    print(f"  [G3] {name} s{seed}: result.json {want:.10f} extracted {got_z['auc_study']:.10f} "
          f"reconstructed {got_r['auc_study']:.10f} max|logit diff| {out['max_abs_logit_diff']:.2e} "
          f"{'PASS' if out['pass'] else 'FAIL'}", flush=True)
    if not out["pass"]:
        raise SystemExit(f"!! G3 FAIL {name}/s{seed}: |delta| {out['delta_extracted']:.3e} / "
                         f"{out['delta_reconstructed']:.3e} >= {G3_TOL:g}; stopping")
    return out


def write(p: Path, **arrays):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, p)


def run_unit(cfg, arm, name, seed, P_, src, device):
    """All 10 files of one (arm, encoder, seed). G3 runs before anything is written."""
    import torch
    cdir = cfg.cache_dir / ("arm_a" if arm == "A" else "arm_b")
    targets = [SOURCE_T] + list(P_)
    todo = [t for t in targets if not cache_path(cdir, name, seed, t).exists()]
    if not todo:
        print(f"[skip] arm {arm} {name} s{seed}: all {len(targets)} files exist", flush=True)
        return None
    tcfg = A.model_transform_cfg(name)
    t_unit = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    if arm == "A":
        model, res, ck_sha = load_arm_a(name, seed, tcfg, device)
        w_src = model.head.weight.detach().cpu().numpy().astype(np.float64).ravel()
        b_src = float(model.head.bias.detach().cpu().numpy().astype(np.float64).ravel()[0])
        feature = model.feature
    else:
        model, res, ck_sha = FrozenEncoder(name).to(device).eval(), None, ""
        feature = ARM_B_FEATURE[name]
    meta = dict(encoder=np.array(name), arm=np.array(arm), feature=np.array(feature),
                seed_tag=np.array(seed), snapshot=np.array(snapshot_commit(name)),
                script_sha256=np.array(cfg.script_sha), checkpoint_sha256=np.array(ck_sha))
    bs, nw = cfg.eval_batch_size, cfg.num_workers
    report = {"arm": arm, "encoder": name, "seed": seed, "feature": feature, "files": {}}

    # source pools first: arm A's G3 must pass before any target is written
    s_tr, s_va = src
    fva, zva, yva = extract(model, s_va, tcfg, device, bs, nw)
    if arm == "A":
        report["g3"] = g3(name, seed, fva, zva, yva, s_va, w_src, b_src, res)
    blocks = {SOURCE_T: (s_tr, s_va, fva, zva, yva)}
    for t in todo:
        t0 = time.time()
        if t == SOURCE_T:
            tr, va, fva, zva, yva = blocks[SOURCE_T]
        else:
            tr, va, _ = P_[t]
            fva, zva, yva = extract(model, va, tcfg, device, bs, nw)
        ftr, _, _ = extract(model, tr, tcfg, device, bs, nw)
        arrays = dict(ftr=ftr, fva=fva, yva=yva,
                      tr_uid=tr.uid.to_numpy().astype(str), va_uid=va.uid.to_numpy().astype(str), **meta)
        line = ""
        if arm == "A":
            st = va.study.to_numpy()
            zs = score(zva, yva, st)["auc_study"]
            if t != SOURCE_T:
                verify_source_head(fva.astype(np.float64), zva, yva, st, w_src, b_src, zs,
                                   f"{name}/s{seed}/{t}")
            arrays.update(zva=zva, w_src=w_src, b_src=np.array(b_src), zs=np.array(zs))
            line = f" zs={zs:.4f}"
        p = cache_path(cdir, name, seed, t)
        write(p, **arrays)
        report["files"][t] = {"path": str(p), "bytes": p.stat().st_size, "n_pool": len(tr),
                              "n_eval": len(va), "minutes": (time.time() - t0) / 60,
                              **({"zs": float(arrays["zs"])} if arm == "A" else {})}
        print(f"  [extract] arm {arm} {name:<10s} s{seed} {t:<14s} pool {len(tr)} eval {len(va)}{line} "
              f"{(time.time() - t0) / 60:.2f} min", flush=True)
    report["minutes"] = (time.time() - t_unit) / 60
    report["peak_gpu_mem_MiB"] = torch.cuda.max_memory_allocated(device) / 2**20
    rp = cdir / "reports" / f"{name}__s{seed}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    del model
    torch.cuda.empty_cache()
    return report


# ================================================================= dry run ===

def dry_run(cfg, P_, src):
    """Pools, counts, GPU time from G2 sec/img, disk. No model, no GPU."""
    rows = [(SOURCE_T, len(src[0]), len(src[1]))] + [(t, len(tr), len(va)) for t, (tr, va, _) in P_.items()]
    print(f"\n{'target':<14s} {'pool':>6s} {'eval':>6s}")
    for t, a, b in rows:
        print(f"{t:<14s} {a:>6d} {b:>6d}")
    n_img = sum(a + b for _, a, b in rows)
    print(f"{'total':<14s} {sum(r[1] for r in rows):>6d} {sum(r[2] for r in rows):>6d}  = {n_img} images per unit")

    # rows identical to the paper's cache?
    same = []
    for t, (tr, va, _) in P_.items():
        z = np.load(cache_path(PAPER_CACHE, PAPER_REF_BB, 42, t), allow_pickle=False)
        same.append(np.array_equal(z["tr_uid"], tr.uid.to_numpy().astype(str))
                    and np.array_equal(z["va_uid"], va.uid.to_numpy().astype(str)))
    print(f"\n[rows] pool/eval uids identical to paper cache {PAPER_REF_BB}__s42: "
          f"{sum(same)}/{len(same)} targets")

    print(f"\n{'arm':<4s}{'encoder':<11s}{'units':>6s}{'dim':>6s}{'G2 s/img':>10s}"
          f"{'GPU min/unit':>14s}{'GPU min':>9s}{'MB/unit':>9s}{'GB':>7s}")
    tot_min, tot_gb = 0.0, 0.0
    for arm in cfg.arms:
        for name in cfg.encoders:
            fn, v = G2_SUMMARY[name]
            g = json.loads((A.G2_DIR / fn).read_text())["models"][name]["variants"][v]
            spi = float(np.mean(g["sec_per_img_runs"]))
            dim = int(next(iter(g["features"].values()))["dim"])
            units = len(cfg.seeds) if arm == "A" else 1
            mb = n_img * dim * 4 / 1e6
            m_unit = n_img * spi / 60
            tot_min += m_unit * units
            tot_gb += mb * units / 1e3
            print(f"{arm:<4s}{name:<11s}{units:>6d}{dim:>6d}{spi:>10.4f}{m_unit:>14.1f}"
                  f"{m_unit * units:>9.1f}{mb:>9.0f}{mb * units / 1e3:>7.2f}")
            if arm == "A":
                for s in cfg.seeds:
                    d = A.OUT_ROOT / name / f"seed{s}"
                    if not ((d / "final.pt").exists() and (d / "result.json").exists()):
                        print(f"     !! missing {d}/final.pt or result.json")
    print(f"{'total':<4s}{'':<11s}{'':>6s}{'':>6s}{'':>10s}{'':>14s}{tot_min:>9.1f}{'':>9s}{tot_gb:>7.2f}")
    print("\nGPU min = images x G2 forward sec/img (batch 10, fp32); excludes model load and "
          "image decoding.\nMB = float32 features only (uids and logits add <2%).")
    have = sorted(p.name for p in cfg.cache_dir.glob("arm_*/features/*.npz"))
    print(f"[out] {cfg.cache_dir}: {len(have)} feature files already present")


# ==================================================================== main ===

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--encoders", default="skelex,rad-dino,medsiglip")
    ap.add_argument("--seeds", default=",".join(str(s) for s in M.SEEDS))
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="pools, counts, time and disk estimate; no GPU")
    cfg = ap.parse_args()
    cfg.arms = [a.strip().upper() for a in cfg.arms.split(",") if a.strip()]
    cfg.encoders = [e.strip() for e in cfg.encoders.split(",") if e.strip()]
    cfg.seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    for e in cfg.encoders:
        if A.ENCODERS[e]["snap"] is None:
            raise SystemExit(f"{e}: no pinned snapshot (DINOv3 placeholder)")
    got = sha256(Path(A.__file__))
    if got != ARM_A_SHA:
        raise SystemExit(f"!! arm_a_finetune.py sha {got} != reviewed {ARM_A_SHA}")
    cfg.script_sha = sha256(Path(__file__).resolve())

    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    P_ = pools(index)
    src = (M.frame_for(index, "mura", SOURCE_T, "train", A.LABEL_COL).reset_index(drop=True),
           M.frame_for(index, "mura", SOURCE_T, "valid", A.LABEL_COL).reset_index(drop=True))
    if cfg.dry_run:
        dry_run(cfg, P_, src)
        return

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{cfg.gpu}")
    torch.cuda.set_device(device)
    M.raise_fd_limit()
    M.configure_sharing()
    for arm in cfg.arms:
        for name in cfg.encoders:
            for seed in (cfg.seeds if arm == "A" else [ARM_B_SEED]):
                run_unit(cfg, arm, name, seed, P_, src, device)


if __name__ == "__main__":
    main()
