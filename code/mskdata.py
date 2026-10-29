#!/usr/bin/env python3
"""Data layer: one index over MURA, GRAZPEDWRI-DX, FracAtlas and MFXR.

Builds a single image index with a common schema, defines the train/validation
splits, and loads pixels identically for every dataset.

Bit depth. GRAZPEDWRI-DX is stored as 16-bit PNG; the other datasets are 8-bit.
PIL's ``Image.convert("RGB")`` clips 16-bit images at 255, silently destroying
them, so pixels are never loaded that way. Every image goes through the same
path instead:

    1. read the raw array with numpy
    2. collapse to one channel
    3. rescale on the image's own 0.5th and 99.5th intensity percentiles to uint8
    4. replicate to three channels
    5. normalise with ImageNet mean and standard deviation at tensor time

Index schema (one row per image):
    uid, dataset, path, anatomy, patient, study, split,
    label_abnormal   MURA abnormal | GRAZPEDWRI-DX any pathological finding |
                     FracAtlas and MFXR fractured
    label_fracture   fracture label where the dataset has one
    has_*            metal, cast, osteopenia, hardware and annotated findings
    meta_*           device manufacturer, projection, age, laterality, ...

FracAtlas has no patient identifier, so each image is its own patient and study.
MFXR patients are the case id encoded in the file name (``<caseid>-<view>``).

Usage:
    python mskdata.py --build --mura <MURA-v1.1> --graz <GRAZPEDWRI-DX> \
        --frac <FracAtlas> --mfxr <MFXR> --out ./diag_data
    python mskdata.py --verify --out ./diag_data    # writes verify_grid.png and audits
    python mskdata.py --selftest                    # no data needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ policy ---
P_LO, P_HI = 0.5, 99.5        # percentile window for uint8 rescale
SPLIT_SEED = 42
GRAZ_VALID_FRAC = 0.15
GRAZ_TEST_FRAC = 0.15

MURA_ANATOMIES = ("XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND",
                  "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST")

# GRAZPEDWRI-DX YOLO class ids, from meta.yaml on disk.
GRAZ_YOLO = {0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 3: "fracture",
             4: "metal", 5: "periostealreaction", 6: "pronatorsign",
             7: "softtissue", 8: "text"}
# 'text' (8) is a laterality marker present in essentially every image and is
# NOT a pathological finding. Excluded from any_finding.
GRAZ_PATHOLOGICAL = [0, 1, 2, 3, 4, 5, 6, 7]

FRAC_ANATOMY_COLS = ("hand", "leg", "hip", "shoulder")

SCHEMA = ["uid", "dataset", "path", "anatomy", "patient", "patient_group",
          "study", "split", "label_abnormal", "label_fracture"]

# ---------------------------------------------------------------------------
# patient vs patient_group
# ---------------------------------------------------------------------------
# 'patient'       is namespaced by anatomy (mura/XR_WRIST/patient00042).
#                 Correct unit for within-anatomy split hygiene.
# 'patient_group' is the RAW person (mura/patient00042), shared across
#                 anatomies. MEASURED on this dataset: 1,715 MURA patients
#                 (14.3%) appear under more than one anatomy, touching 11,139
#                 images (27.8%).
#
# MURA's official train/valid split is GLOBALLY patient-level -- zero patients
# cross that boundary -- so zero-shot evaluation on target *valid* is clean.
# The overlap lives inside the train split (16-249 patients per anatomy pair).
# It therefore only matters when target-TRAIN data is treated as unseen, e.g.
# few-shot calibration pools. Use assert_pair_disjoint() for those designs.
# ---------------------------------------------------------------------------


def add_patient_group(df: pd.DataFrame) -> pd.DataFrame:
    """Raw person identity, stripped of the anatomy namespace."""
    df = df.copy()
    is_mura = df["dataset"] == "mura"
    grp = df["patient"].astype(str)
    # mura/<ANATOMY>/<patientXXXXX>  ->  mura/<patientXXXXX>
    grp = grp.where(~is_mura,
                    "mura/" + df.loc[:, "patient"].astype(str).str.split("/").str[-1])
    df["patient_group"] = grp
    return df


def assert_pair_disjoint(df: pd.DataFrame, source: str, target: str,
                         source_split: str = "train",
                         target_split: str = "train",
                         raise_on_overlap: bool = True) -> int:
    """
    Check person-level disjointness between a source pool and a target pool.

    Call this for ANY design that treats target-anatomy training data as
    unseen. Returns the number of overlapping people.
    """
    s = set(df[(df.anatomy == source) & (df.split == source_split)]["patient_group"])
    t = set(df[(df.anatomy == target) & (df.split == target_split)]["patient_group"])
    ov = s & t
    if ov and raise_on_overlap:
        raise AssertionError(
            f"{source}({source_split}) and {target}({target_split}) share "
            f"{len(ov)} people, e.g. {sorted(ov)[:3]}. Exclude them or use a "
            f"target split that is person-disjoint from the source pool."
        )
    return len(ov)


def drop_source_patients(df: pd.DataFrame, pool: pd.DataFrame,
                         source: str, source_split: str = "train") -> pd.DataFrame:
    """Remove from `pool` anyone who appears in the source training pool."""
    seen = set(df[(df.anatomy == source) & (df.split == source_split)]["patient_group"])
    keep = ~pool["patient_group"].isin(seen)
    n = int((~keep).sum())
    if n:
        print(f"[leak-control] dropped {n} images from {len(pool)} "
              f"({n/max(1,len(pool)):.1%}) sharing people with {source}/{source_split}")
    return pool[keep].reset_index(drop=True)


# ============================================================== provenance ===

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def code_hashes() -> Dict[str, object]:
    """What actually ran: the invoked script and every module loaded from this
    code folder, each with its sha256.

    Earlier versions of provenance() hashed only their own file, so the headers
    of older result files record mskdata.py's or mskmodel.py's hash under
    "script_sha256".
    """
    here = Path(__file__).resolve().parent
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    out: Dict[str, object] = {
        "script_file": Path(main).name if main else "interactive",
        "script_sha256": _sha256_file(Path(main).resolve()) if main else "NA"}
    local = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        # torch.ops / torch.classes carry a bare relative __file__ ("_ops.py");
        # resolving it against the cwd would invent a file in this folder.
        if not isinstance(f, str) or not os.path.isabs(f):
            continue
        p = Path(f).resolve()
        if p.suffix == ".py" and p.parent == here:
            local[p.name] = _sha256_file(p)
    out["code_sha256"] = dict(sorted(local.items()))
    return out


def provenance() -> Dict[str, object]:
    me = Path(__file__).resolve()
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=me.parent,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "NO_GIT"
    return {**code_hashes(), "git_sha": git,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "P_LO": str(P_LO), "P_HI": str(P_HI), "SPLIT_SEED": str(SPLIT_SEED)}


def write_csv(df: pd.DataFrame, path: Path, prov: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for line in json.dumps({"provenance": prov}, default=str).splitlines():
            fh.write("# " + line + "\n")
        df.to_csv(fh, index=False)


# =========================================================== image loading ===

def load_gray_uint8(path: str, p_lo: float = P_LO, p_hi: float = P_HI) -> np.ndarray:
    """
    THE function. Returns HxW uint8 for any of the three datasets.

    Never calls .convert() on 16-bit data. Percentile rescale is applied to
    every dataset identically so that preprocessing cannot masquerade as
    appearance shift.
    """
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im)

    if arr.ndim == 3:                      # some MURA/FracAtlas files are RGB
        arr = arr[..., :3].mean(axis=2)    # channels are duplicates; mean is safe
    arr = arr.astype(np.float32)

    lo, hi = np.percentile(arr, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def load_rgb_uint8(path: str) -> np.ndarray:
    """HxWx3 uint8 — grayscale replicated. What the model actually sees."""
    g = load_gray_uint8(path)
    return np.repeat(g[:, :, None], 3, axis=2)


# ================================================================== MURA =====

def index_mura(root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    skipped = {"dotfile": 0, "non_image": 0}
    for split in ("train", "valid"):
        sdir = root / split
        if not sdir.is_dir():
            raise FileNotFoundError(f"MURA: missing {sdir}")
        for adir in sorted(sdir.iterdir()):
            if not adir.is_dir() or not adir.name.startswith("XR_"):
                continue
            for pdir in sorted(adir.iterdir()):
                if not pdir.is_dir() or pdir.name.startswith("."):
                    continue
                for stdir in sorted(pdir.iterdir()):
                    if not stdir.is_dir() or stdir.name.startswith("."):
                        continue
                    low = stdir.name.lower()
                    if "positive" in low:
                        lab = 1
                    elif "negative" in low:
                        lab = 0
                    else:
                        continue
                    for f in sorted(stdir.iterdir()):
                        if not f.is_file():
                            continue
                        if f.name.startswith("."):
                            skipped["dotfile"] += 1
                            continue
                        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                            skipped["non_image"] += 1
                            continue
                        rows.append({
                            "dataset": "mura", "path": str(f),
                            "anatomy": adir.name,
                            "patient": f"mura/{adir.name}/{pdir.name}",
                            "study": f"mura/{adir.name}/{pdir.name}/{stdir.name}",
                            "split": split,
                            "label_abnormal": lab,
                            "label_fracture": np.nan,
                        })
    df = pd.DataFrame(rows)
    print(f"[mura] {len(df)} images, {df.study.nunique()} studies, "
          f"{df.patient.nunique()} patients; skipped {skipped}")
    if df.empty:
        raise RuntimeError("MURA index empty")
    return df


# =========================================================== GRAZPEDWRI-DX ===

def _graz_yolo_flags(root: Path) -> pd.DataFrame:
    """Per-image presence flags for the 9 YOLO classes, keyed by filestem."""
    label_files: List[Path] = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".")]
        if not any(k in dp.lower() for k in ("label", "annot")):
            continue
        label_files += [Path(dp) / f for f in fn
                        if f.endswith(".txt") and not f.startswith(".")]
    if not label_files:
        print("[graz] WARNING: no YOLO label files found; "
              "any_finding will fall back to CSV columns only")
        return pd.DataFrame()

    recs: List[Dict] = []
    for t in label_files:
        seen = set()
        try:
            for line in open(t):
                p = line.split()
                if p:
                    try:
                        seen.add(int(float(p[0])))
                    except ValueError:
                        pass
        except Exception:
            continue
        r = {"filestem": t.stem}
        for cid, cname in GRAZ_YOLO.items():
            r[f"yolo_{cname}"] = int(cid in seen)
        recs.append(r)

    df = pd.DataFrame(recs).drop_duplicates("filestem")
    print(f"[graz] parsed {len(df)} YOLO label files")
    return df


def index_graz(root: Path) -> pd.DataFrame:
    meta_path = root / "dataset.csv"
    if not meta_path.exists():
        cands = list(root.rglob("dataset.csv"))
        if not cands:
            raise FileNotFoundError(f"GRAZ: dataset.csv not found under {root}")
        meta_path = cands[0]
    meta = pd.read_csv(meta_path)
    meta.columns = [c.replace("\ufeff", "").strip() for c in meta.columns]
    if "filestem" not in meta.columns:
        raise RuntimeError(f"GRAZ: 'filestem' missing; got {list(meta.columns)}")

    # Resolve images by stem — robust to whatever the directory layout is.
    stem2path: Dict[str, str] = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fn:
            if f.startswith(".") or Path(f).suffix.lower() not in (".png", ".jpg"):
                continue
            stem2path.setdefault(Path(f).stem, str(Path(dp) / f))
    meta["path"] = meta["filestem"].astype(str).map(stem2path)
    missing = int(meta["path"].isna().sum())
    if missing:
        print(f"[graz] WARNING: {missing} metadata rows have no image on disk; dropped")
        meta = meta.dropna(subset=["path"])

    yolo = _graz_yolo_flags(root)
    if not yolo.empty:
        meta = meta.merge(yolo, on="filestem", how="left")
        for c in [c for c in meta.columns if c.startswith("yolo_")]:
            meta[c] = meta[c].fillna(0).astype(int)

    def num(col: str) -> pd.Series:
        return pd.to_numeric(meta[col], errors="coerce").fillna(0).astype(int) \
            if col in meta.columns else pd.Series(0, index=meta.index, dtype=int)

    frac = num("fracture_visible")
    yolo_cols = [f"yolo_{GRAZ_YOLO[c]}" for c in GRAZ_PATHOLOGICAL
                 if f"yolo_{GRAZ_YOLO[c]}" in meta.columns]
    if yolo_cols:
        any_find = (meta[yolo_cols].sum(axis=1) > 0).astype(int)
        src = f"YOLO classes {sorted(GRAZ_PATHOLOGICAL)} (text excluded)"
    else:
        any_find = ((frac + num("metal") + num("osteopenia")) > 0).astype(int)
        src = "CSV columns (fracture_visible|metal|osteopenia)"
    print(f"[graz] any_finding source: {src}")

    out = pd.DataFrame({
        "dataset": "graz",
        "path": meta["path"].astype(str),
        "anatomy": "XR_WRIST",
        "patient": "graz/" + meta["patient_id"].astype(str),
        "study": "graz/" + meta["patient_id"].astype(str) + "/"
                 + meta.get("study_number", pd.Series(0, index=meta.index)).astype(str),
        "split": "unassigned",
        "label_abnormal": any_find.values,
        "label_fracture": frac.values,
    })
    for c in ("metal", "cast", "osteopenia", "diagnosis_uncertain"):
        out[f"has_{c}"] = num(c).values
    for c in [c for c in meta.columns if c.startswith("yolo_")]:
        out[c] = meta[c].values
    for c in ("device_manufacturer", "projection", "laterality", "age",
              "gender", "ao_classification", "pixel_spacing", "initial_exam"):
        if c in meta.columns:
            out[f"meta_{c}"] = meta[c].values

    print(f"[graz] {len(out)} images, {out.study.nunique()} studies, "
          f"{out.patient.nunique()} patients | "
          f"abnormal={out.label_abnormal.mean():.3f} "
          f"fracture={out.label_fracture.mean():.3f}")
    return out


def split_graz(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Fresh patient-grouped split. Pre-existing graz_*.csv are ignored."""
    pats = np.array(sorted(df["patient"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    n_val = int(round(GRAZ_VALID_FRAC * n))
    n_tst = int(round(GRAZ_TEST_FRAC * n))
    assign = {}
    for p in pats[:n_val]:
        assign[p] = "valid"
    for p in pats[n_val:n_val + n_tst]:
        assign[p] = "test"
    for p in pats[n_val + n_tst:]:
        assign[p] = "train"
    df = df.copy()
    df["split"] = df["patient"].map(assign)
    for a in ("train", "valid", "test"):
        for b in ("train", "valid", "test"):
            if a < b:
                ov = set(df[df.split == a].patient) & set(df[df.split == b].patient)
                assert not ov, f"GRAZ {a}/{b} patient overlap: {list(ov)[:3]}"
    print(f"[graz] fresh split (seed={seed}): "
          f"{df.split.value_counts().to_dict()}")
    return df


# ============================================================== FracAtlas ====

def index_fracatlas(root: Path) -> pd.DataFrame:
    meta_path = root / "dataset.csv"
    if not meta_path.exists():
        cands = [p for p in root.rglob("dataset.csv")]
        if not cands:
            raise FileNotFoundError(f"FracAtlas: dataset.csv not found under {root}")
        meta_path = cands[0]
    meta = pd.read_csv(meta_path)
    meta.columns = [c.replace("\ufeff", "").strip() for c in meta.columns]

    stem2path: Dict[str, str] = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fn:
            if f.startswith(".") or Path(f).suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem2path.setdefault(Path(f).stem, str(Path(dp) / f))
            stem2path.setdefault(f, str(Path(dp) / f))
    key = "image_id" if "image_id" in meta.columns else meta.columns[0]
    meta["path"] = meta[key].astype(str).map(
        lambda s: stem2path.get(s) or stem2path.get(Path(s).stem))
    miss = int(meta["path"].isna().sum())
    if miss:
        print(f"[frac] WARNING: {miss} rows have no image on disk; dropped")
        meta = meta.dropna(subset=["path"])

    def num_of(frame: pd.DataFrame, c: str) -> pd.Series:
        return pd.to_numeric(frame[c], errors="coerce").fillna(0).astype(int) \
            if c in frame.columns else pd.Series(0, index=frame.index, dtype=int)

    # Anatomy flags are NOT mutually exclusive. Keep only unambiguous images.
    flags0 = pd.DataFrame({c: num_of(meta, c) for c in FRAC_ANATOMY_COLS})
    keep = (flags0.sum(axis=1) == 1) & (num_of(meta, "mixed") == 0)
    dropped = int((~keep).sum())
    print(f"[frac] dropped {dropped} images with 0/multiple anatomy flags or mixed=1")

    # Filter ONCE, then derive everything from the filtered frame. Deriving
    # before filtering leaves stale-length masks -- a real bug caught in test.
    meta = meta[keep].reset_index(drop=True)
    flags = pd.DataFrame({c: num_of(meta, c) for c in FRAC_ANATOMY_COLS})

    anat = flags.idxmax(axis=1).map(
        {"hand": "XR_HAND", "shoulder": "XR_SHOULDER",
         "leg": "LEG", "hip": "HIP"})
    fractured = num_of(meta, "fractured")

    out = pd.DataFrame({
        "dataset": "fracatlas",
        "path": meta["path"].astype(str),
        "anatomy": anat.values,
        # No patient column exists -> patient-grouped splitting is impossible.
        # FracAtlas is therefore EVALUATION ONLY; uid doubles as patient/study.
        "patient": "frac/" + meta[key].astype(str),
        "study": "frac/" + meta[key].astype(str),
        "split": "test",
        "label_abnormal": fractured.values,
        "label_fracture": fractured.values,
    })
    out["has_hardware"] = num_of(meta, "hardware").values
    for c in ("fracture_count", "frontal", "lateral", "oblique", "multiscan"):
        if c in meta.columns:
            out[f"meta_{c}"] = meta[c].values

    print(f"[frac] {len(out)} images (eval-only) | "
          f"per-anatomy {out.anatomy.value_counts().to_dict()}")
    return out


# =================================================================== build ===

MFXR_ANATOMY = "XR_FOREARM"
MFXR_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MFXR_SPLIT_MAP = {"train": "train", "test": "valid"}
MFXR_LABEL_MAP = {"anomaly": 1, "good": 0}


def index_mfxr(root: Path, verify_images: bool = True) -> pd.DataFrame:
    """MFXR: Monteggia fracture radiographs. doi:10.1038/s41597-026-07547-y

    Layout on disk:  <root>/{train,test}/{anomaly,good}/<caseid>-<view>.<ext>

    The case id is the grouping key and is the only reason this dataset can
    carry a reference model where FracAtlas cannot. Checked on the downloaded
    copy when MFXR was added: 4,586 images, 1,484 case ids, no case in both
    splits and no case with both labels; the build asserts patient-disjoint
    splits.
    """
    import re as _re
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MFXR: {root} does not exist")
    # Tolerate the extra nesting produced by unzipping (MFXR/MFXR/train/...).
    if not (root / "train").is_dir() and (root / "MFXR" / "train").is_dir():
        root = root / "MFXR"

    pat = _re.compile(r"^(?P<case>.+)-(?P<view>\d+)$")
    rows, unparsed, unreadable = [], [], []
    for split_dir, split in MFXR_SPLIT_MAP.items():
        for cls, lab in MFXR_LABEL_MAP.items():
            d = root / split_dir / cls
            if not d.is_dir():
                raise FileNotFoundError(f"MFXR: expected directory {d}")
            for p in sorted(d.iterdir()):
                # "._x" are AppleDouble stubs from the macOS zip, not images.
                if p.name.startswith(".") or not p.is_file():
                    continue
                if p.suffix.lower() not in MFXR_EXTS:
                    continue
                m = pat.match(p.stem)
                if not m:
                    unparsed.append(p.name)
                    continue
                if verify_images:
                    try:
                        from PIL import Image
                        with Image.open(p) as im:
                            im.verify()
                    except Exception as e:
                        unreadable.append((p.name, repr(e)))
                        continue
                case = m.group("case")
                rows.append({
                    "dataset": "mfxr",
                    "path": str(p),
                    "anatomy": MFXR_ANATOMY,
                    "patient": f"mfxr/{case}",
                    "study": f"mfxr/{case}",
                    "split": split,
                    "label_abnormal": lab,
                    "label_fracture": lab,
                    "meta_case": case,
                    "meta_view": int(m.group("view")),
                    "meta_author_split": split_dir,
                })
    if unparsed:
        raise SystemExit(
            f"!! MFXR: {len(unparsed)} filename(s) do not match "
            f"<caseid>-<view>.<ext>, e.g. {unparsed[:3]}.\n"
            "   The case id is the patient grouping key; guessing it wrong "
            "would leak patients across the budget sample and the evaluation "
            "half. Fix the names or extend the pattern deliberately.")
    if unreadable:
        print(f"[mfxr] WARNING: dropped {len(unreadable)} unreadable image(s), "
              f"e.g. {unreadable[:2]}")
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit(f"!! MFXR: no images found under {root}")

    # A case carrying both labels would make the study-level label ambiguous.
    mixed = (out.groupby("study")["label_abnormal"].nunique() > 1)
    if mixed.any():
        raise SystemExit(
            f"!! MFXR: {int(mixed.sum())} case(s) carry both labels, e.g. "
            f"{list(mixed[mixed].index[:3])}. Study-level labels would be "
            "ambiguous; this contradicts the downloaded copy and must be "
            "investigated before use.")

    n_case = out.study.nunique()
    print(f"[mfxr] {len(out)} images | {n_case} cases | "
          f"per split {out.groupby('split').size().to_dict()} | "
          f"prevalence {out.label_abnormal.mean():.3f}")
    if n_case != 1482:
        print(f"[mfxr] NOTE: {n_case} case ids from filenames; the paper "
              f"reports 1,482 patients. The image total is the authority "
              f"(expected 4,586, got {len(out)}). Declare this in the paper "
              f"rather than reconciling it silently.")
    return out


def build(mura: Optional[str], graz: Optional[str], frac: Optional[str],
          out_dir: Path,
          mfxr: Optional[str] = None) -> pd.DataFrame:
    prov = provenance()
    parts: List[pd.DataFrame] = []

    if mura:
        parts.append(index_mura(Path(mura)))
    if graz:
        parts.append(split_graz(index_graz(Path(graz))))
    if frac:
        parts.append(index_fracatlas(Path(frac)))
    if mfxr:
        parts.append(index_mfxr(Path(mfxr)))
    if not parts:
        raise SystemExit("give at least one of --mura/--graz/--frac/--mfxr")

    df = pd.concat(parts, ignore_index=True, sort=False)
    df.insert(0, "uid", [f"{d}_{i:07d}" for i, d in enumerate(df["dataset"])])
    df = add_patient_group(df)

    # ---- invariants ----------------------------------------------------
    assert df["uid"].is_unique, "uid collision"
    assert not df["path"].duplicated().any(), "duplicate image paths in index"
    for ds in df["dataset"].unique():
        sub = df[df.dataset == ds]
        by = sub.groupby("patient")["split"].nunique()
        bad = by[by > 1]
        assert bad.empty, f"{ds}: patients spanning splits: {list(bad.index)[:3]}"
    assert df[df.dataset == "fracatlas"]["split"].eq("test").all(), \
        "FracAtlas must be test-only"

    # The one that actually matters: RAW person, not anatomy-namespaced.
    # An earlier version asserted on the namespaced id and so could not
    # detect a person appearing under two different anatomies.
    for ds in df["dataset"].unique():
        sub = df[df.dataset == ds]
        by = sub.groupby("patient_group")["split"].nunique()
        bad = by[by > 1]
        assert bad.empty, (
            f"{ds}: {len(bad)} PEOPLE span splits (person-level leakage), "
            f"e.g. {list(bad.index)[:3]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(df, out_dir / "index.csv", prov)
    try:
        df.to_parquet(out_dir / "index.parquet", index=False)
    except Exception as e:
        print(f"[build] parquet unavailable ({e}); csv written")

    print(f"\n[build] unified index: {len(df)} rows -> {out_dir/'index.csv'}")
    audit(df, out_dir, prov)
    return df


def audit(df: pd.DataFrame, out_dir: Path, prov: Dict) -> None:
    rows = []
    for (ds, an, sp), g in df.groupby(["dataset", "anatomy", "split"]):
        rows.append({
            "dataset": ds, "anatomy": an, "split": sp,
            "images": len(g), "studies": g.study.nunique(),
            "patients": g.patient.nunique(),
            "prev_abnormal": round(float(g.label_abnormal.mean()), 4),
            "prev_fracture": (round(float(g.label_fracture.mean()), 4)
                              if g.label_fracture.notna().any() else np.nan),
        })
    a = pd.DataFrame(rows).sort_values(["dataset", "anatomy", "split"])
    write_csv(a, out_dir / "audit_counts.csv", prov)
    print("\n--- counts & prevalence ---")
    print(a.to_string(index=False))

    fcols = [c for c in df.columns if c.startswith(("yolo_", "has_"))]
    if fcols:
        fm = df.groupby(["dataset", "anatomy"])[fcols].mean().round(4)
        write_csv(fm.reset_index(), out_dir / "audit_finding_mix.csv", prov)
        print("\n--- finding mix (F1 input) ---")
        print(fm.to_string())

    if "meta_device_manufacturer" in df.columns:
        sc = (df[df.meta_device_manufacturer.notna()]
              .groupby(["dataset", "meta_device_manufacturer"])
              .agg(images=("uid", "size"),
                   prev_abnormal=("label_abnormal", "mean"),
                   prev_fracture=("label_fracture", "mean")).round(4))
        write_csv(sc.reset_index(), out_dir / "audit_scanners.csv", prov)
        print("\n--- scanners (enables the pure acquisition-shift control) ---")
        print(sc.to_string())

    g = df[df.dataset == "graz"]
    if len(g):
        both = pd.crosstab(g.label_abnormal, g.label_fracture)
        print("\n--- F1 concept shift: GRAZ any_finding vs fracture "
              "(identical images) ---")
        print(both.to_string())
        n_af_nf = int(((g.label_abnormal == 1) & (g.label_fracture == 0)).sum())
        print(f"  abnormal-but-not-fracture: {n_af_nf} images "
              f"({n_af_nf/max(1,len(g)):.1%}) -- this is the concept gap")

    # ---- cross-anatomy person overlap (reportable table) ----------------
    import itertools
    m = df[(df.dataset == "mura") & (df.split == "train")]
    if len(m):
        pg = m.groupby("anatomy")["patient_group"].apply(set)
        rows = []
        for a, b in itertools.combinations(sorted(pg.index), 2):
            n = len(pg[a] & pg[b])
            if n:
                rows.append({"anatomy_a": a, "anatomy_b": b, "shared_people": n,
                             "frac_of_a": round(n / len(pg[a]), 4),
                             "frac_of_b": round(n / len(pg[b]), 4)})
        if rows:
            ov = pd.DataFrame(rows).sort_values("shared_people", ascending=False)
            write_csv(ov, out_dir / "audit_anatomy_patient_overlap.csv", prov)
            multi = m.groupby("patient_group")["anatomy"].nunique()
            print("\n--- cross-anatomy person overlap (MURA train) ---")
            print(f"  {int((multi>1).sum())} people appear under >1 anatomy "
                  f"({(multi>1).mean():.1%} of people)")
            print("  Zero-shot eval on target VALID is unaffected (official "
                  "split is globally person-level).")
            print("  Matters only where target-TRAIN is treated as unseen "
                  "-- use assert_pair_disjoint() there.")
            print(ov.head(8).to_string(index=False))


# ================================================================ verify =====

def verify(out_dir: Path, n_per: int = 4) -> None:
    """
    Render processed images from all three datasets side by side.
    If the 16-bit handling is wrong you will SEE it. Do not skip this.
    """
    from PIL import Image
    idx = pd.read_csv(out_dir / "index.csv", comment="#")
    rng = np.random.default_rng(0)
    cell, rows = 224, []
    for ds in ("mura", "graz", "fracatlas"):
        sub = idx[idx.dataset == ds]
        if sub.empty:
            continue
        sel = sub.sample(min(n_per, len(sub)), random_state=0)
        tiles = []
        for _, r in sel.iterrows():
            try:
                g = load_gray_uint8(r["path"])
                im = Image.fromarray(g).resize((cell, cell))
                tiles.append((np.asarray(im), r))
            except Exception as e:
                tiles.append((np.zeros((cell, cell), np.uint8), r))
                print(f"  !! {ds}: {e}")
        rows.append((ds, tiles))

    if not rows:
        print("nothing to verify")
        return
    W = max(len(t) for _, t in rows) * cell
    H = len(rows) * cell
    canvas = np.zeros((H, W), np.uint8)
    print("\n--- verification: per-image intensity after processing ---")
    for ri, (ds, tiles) in enumerate(rows):
        for ci, (t, r) in enumerate(tiles):
            canvas[ri*cell:(ri+1)*cell, ci*cell:(ci+1)*cell] = t
        stats = [f"{t.mean():.0f}" for t, _ in tiles]
        print(f"  {ds:<10s} mean intensities: {stats}   "
              f"(all-white ~255 or all-black ~0 => conversion is broken)")
    dest = out_dir / "verify_grid.png"
    Image.fromarray(canvas).save(dest)
    print(f"\nrows top->bottom: {[d for d, _ in rows]}")
    print(f"grid written to {dest}  -- OPEN IT AND LOOK AT IT")


# ============================================================== selftest =====

def selftest() -> None:
    ok = True

    def ck(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        ok &= bool(cond)

    from PIL import Image
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)

    # 16-bit round trip: the failure mode this module exists to prevent
    a16 = (rng.random((64, 64)) * 65535).astype(np.uint16)
    p16 = tmp / "a.png"
    Image.fromarray(a16).save(p16)
    g = load_gray_uint8(str(p16))
    ck("16-bit -> uint8 dtype", g.dtype == np.uint8)
    ck("16-bit uses full range", g.max() > 200 and g.min() < 55,
       f"(min={g.min()} max={g.max()})")
    ck("16-bit not saturated", 60 < g.mean() < 195, f"(mean={g.mean():.1f})")

    naive = np.asarray(Image.open(p16).convert("L"))
    ck("naive .convert('L') IS broken (justifies this module)",
       naive.mean() > 240, f"(naive mean={naive.mean():.1f})")

    # 8-bit passthrough and monotonicity
    a8 = (rng.random((64, 64)) * 255).astype(np.uint8)
    p8 = tmp / "b.png"
    Image.fromarray(a8).save(p8)
    g8 = load_gray_uint8(str(p8))
    ck("8-bit stays uint8", g8.dtype == np.uint8)
    order = np.corrcoef(a8.ravel().astype(float), g8.ravel().astype(float))[0, 1]
    ck("rescale is monotone", order > 0.99, f"(r={order:.4f})")

    # RGB collapse
    Image.fromarray(np.repeat(a8[:, :, None], 3, 2)).save(tmp / "c.png")
    ck("RGB collapses to 2-D", load_gray_uint8(str(tmp / "c.png")).ndim == 2)
    ck("rgb loader gives HxWx3", load_rgb_uint8(str(p8)).shape == (64, 64, 3))

    # constant image must not divide by zero
    Image.fromarray(np.full((32, 32), 7, np.uint8)).save(tmp / "d.png")
    ck("constant image safe", np.isfinite(load_gray_uint8(str(tmp / "d.png"))).all())

    # split logic
    df = pd.DataFrame({"patient": [f"graz/{i//3}" for i in range(60)],
                       "path": [f"p{i}" for i in range(60)]})
    s = split_graz(df, seed=1)
    leak = any(set(s[s.split == a].patient) & set(s[s.split == b].patient)
               for a in ("train", "valid", "test") for b in ("train", "valid", "test")
               if a < b)
    ck("split is patient-disjoint", not leak)
    ck("all rows assigned", s.split.notna().all() and s.split.nunique() == 3)

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


# ==================================================================== main ===

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mura", default=None)
    ap.add_argument("--graz", default=None)
    ap.add_argument("--frac", default=None)
    ap.add_argument("--mfxr", default=None)
    ap.add_argument("--out", default="./diag_data")
    a = ap.parse_args()

    if a.selftest:
        selftest()
    out = Path(a.out)
    if a.build:
        build(a.mura, a.graz, a.frac, out, mfxr=a.mfxr)
    if a.verify:
        verify(out)
    if not (a.build or a.verify):
        ap.print_help()


if __name__ == "__main__":
    main()
