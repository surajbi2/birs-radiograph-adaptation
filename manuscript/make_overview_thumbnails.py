"""Make the radiograph thumbnails embedded in Fig. 1.

Each image is rescaled on its own 0.5th and 99.5th intensity percentiles (the
study's preprocessing step), cropped to the exposed region, padded to a square
and saved at 360 px. sources.tsv records the file behind every thumbnail.

The datasets are not redistributed; point the script at local copies:
    MURA_ROOT           folder holding MURA-v1.1/valid
    GRAZPEDWRI_ZIP      the GRAZPEDWRI-DX archive (images are read from it directly)
    MFXR_ROOT           folder holding MFXR/test
    FRACATLAS_ROOT      folder holding FracAtlas/images

Output:
    output/thumbs/*.jpg, output/thumbs/sources.tsv

Usage:
    MURA_ROOT=... GRAZPEDWRI_ZIP=... MFXR_ROOT=... FRACATLAS_ROOT=... python make_overview_thumbnails.py
"""
import csv
import io
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "thumbs"
OUT.mkdir(parents=True, exist_ok=True)


def dataset_path(var):
    """Dataset location from an environment variable (the datasets are not part of the repository)."""
    if var not in os.environ:
        raise SystemExit(f"set {var} (see the module docstring)")
    return Path(os.environ[var])


MURA = dataset_path("MURA_ROOT") / "MURA-v1.1" / "valid"
GRAZ = dataset_path("GRAZPEDWRI_ZIP")
MFXR = dataset_path("MFXR_ROOT") / "MFXR"
FRACATLAS = dataset_path("FRACATLAS_ROOT") / "FracAtlas"
PICK = {
    "mura_wrist": MURA / "XR_WRIST/patient11200/study1_positive/image1.png",
    "mura_elbow": MURA / "XR_ELBOW/patient11345/study1_negative/image1.png",
    "mura_finger": MURA / "XR_FINGER/patient11347/study1_positive/image1.png",
    "mura_forearm": MURA / "XR_FOREARM/patient11472/study1_negative/image1.png",
    "mura_hand": MURA / "XR_HAND/patient11255/study2_negative/image1.png",
    "mura_humerus": MURA / "XR_HUMERUS/patient11289/study1_negative/image1.png",
    "mura_shoulder": MURA / "XR_SHOULDER/patient11746/study1_positive/image1.png",
    "mfxr": MFXR / "test/anomaly/234865-2.jpg",
    "fracatlas_hand": FRACATLAS / "images/Non_fractured/IMG0003071.jpg",
}
# FracAtlas images show the whole forearm on a large detector: keep the hand region (fractions of w, h)
FRAME = {"fracatlas_hand": (0.12, 0.30, 0.88, 0.97)}
GRAZ_PICK = {"graz": "0583_0675105198_01_WRI-R1_F011.png", "graz_local1": "0712_0206897994_01_WRI-R1_M013.png",
             "graz_local2": "0715_0398071558_01_WRI-L1_F016.png", "graz_local3": "1154_1033108233_01_WRI-R1_M012.png"}


def to8(img):
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    lo, hi = np.percentile(a, [0.5, 99.5])
    return Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8))


def content(g):
    """Crop to the exposed region: rows and columns whose 95th-percentile grey exceeds 12% of full scale."""
    w, h = g.size
    g = g.crop((int(0.04 * w), int(0.04 * h), int(0.96 * w), int(0.96 * h)))   # detector edges and frames
    a = np.asarray(g)
    r = np.where(np.percentile(a, 95, axis=1) > 30)[0]
    c = np.where(np.percentile(a, 95, axis=0) > 30)[0]
    if len(r) < 10 or len(c) < 10:
        return g
    m = 6
    return g.crop((max(c[0] - m, 0), max(r[0] - m, 0), min(c[-1] + m, a.shape[1]), min(r[-1] + m, a.shape[0])))


def save(key, img, src):
    g = to8(img)
    if key in FRAME:
        w, h = g.size
        f = FRAME[key]
        g = g.crop((int(f[0] * w), int(f[1] * h), int(f[2] * w), int(f[3] * h)))
    g = content(g)
    s = max(g.size)
    sq = Image.new("L", (s, s), 0)
    sq.paste(g, ((s - g.width) // 2, (s - g.height) // 2))
    sq.resize((360, 360), Image.LANCZOS).save(OUT / f"{key}.jpg", quality=90)
    rows.append((key, src))


rows = []
for key, p in PICK.items():
    save(key, Image.open(p), str(p))
z = zipfile.ZipFile(GRAZ)
inner = zipfile.ZipFile(z.open("images_part1.zip"))
for key, name in GRAZ_PICK.items():
    save(key, Image.open(io.BytesIO(inner.read(name))), f"{GRAZ}::images_part1.zip::{name}")
with open(OUT / "sources.tsv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f, delimiter="\t").writerows([("thumbnail", "source")] + rows)
print(f"{len(rows)} thumbnails in {OUT}")
