#!/usr/bin/env python3
"""Measure what a direct 16-bit to 8-bit cast does to GRAZPEDWRI-DX images.

For each 16-bit image (or a random sample), reports the fraction of pixels a
direct cast sends to pure black or pure white and how far the result departs
from percentile windowing.

Usage:
    python audit_bit_depth.py --root ~/datasets/GRAZPEDWRI-DX --sample 500
    python audit_bit_depth.py --root ~/datasets/GRAZPEDWRI-DX --sample 0   # all images

CPU only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

EXTS = (".png", ".tif", ".tiff", ".pgm")


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=500,
                    help="how many 16 bit images to measure; 0 means all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-lo", type=float, default=0.5)
    ap.add_argument("--p-hi", type=float, default=99.5)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path for a CSV of the per-image measurements")
    cfg = ap.parse_args()

    files = [p for p in sorted(cfg.root.rglob("*"))
             if p.is_file() and p.suffix.lower() in EXTS]
    print(f"{len(files)} image files under {cfg.root}")
    if cfg.sample and len(files) > cfg.sample:
        files = random.Random(cfg.seed).sample(files, cfg.sample)
        print(f"measuring a random sample of {len(files)} (seed {cfg.seed})")

    rows, skipped = [], 0
    for p in files:
        try:
            with Image.open(p) as im:
                if im.mode not in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
                    skipped += 1
                    continue
                a = np.array(im)
        except Exception:
            skipped += 1
            continue
        if a.ndim != 2 or int(a.max()) <= 255:
            skipped += 1
            continue
        a = a.astype(np.float64)

        naive = np.clip(a / 256.0, 0, 255)
        lo, hi = np.percentile(a, [cfg.p_lo, cfg.p_hi])
        win = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255

        rows.append({
            "file": p.name,
            "crushed": float((naive <= 0.5).mean()),
            "saturated": float((naive >= 254.5).mean()),
            "span": float(hi - lo) / 65535.0,
            "mean_abs_diff": float(np.abs(win - naive).mean()),
            "max_stored": float(a.max()),
        })

    print(f"measured {len(rows)} 16 bit images, skipped {skipped} "
          f"(8 bit or unreadable)")
    if not rows:
        raise SystemExit(
            "!! nothing measured. Either the dataset is 8 bit, in which case the\n"
            "   preprocessing sentence is about a problem that cannot arise here and\n"
            "   must be removed, or Pillow is not reading the mode as I;16.")

    import statistics as st
    def col(k):
        return [r[k] for r in rows]

    def show(k, pct=True):
        v = sorted(col(k))
        f = (lambda x: f"{100*x:6.2f}%") if pct else (lambda x: f"{x:7.2f}")
        print(f"  {k:14} median {f(st.median(v))}   90th {f(v[int(.9*len(v))-1])}"
              f"   max {f(v[-1])}")

    print("\nDIRECT CAST DAMAGE")
    show("crushed"); show("saturated"); show("span")
    print("\nDISAGREEMENT BETWEEN THE TWO METHODS (8 bit levels, 0 to 255)")
    show("mean_abs_diff", pct=False)

    big = [r for r in rows if r["mean_abs_diff"] > 10]
    narrow = [r for r in rows if r["span"] < 0.25]
    print(f"\nimages where the two methods differ by more than 10 levels: "
          f"{len(big)} of {len(rows)} ({100*len(big)/len(rows):.1f}%)")
    print(f"images using less than a quarter of the 16 bit range:        "
          f"{len(narrow)} of {len(rows)} ({100*len(narrow)/len(rows):.1f}%)")

    print("\nVERDICT")
    if len(big) / len(rows) < 0.02:
        print("  The bit-depth claim is NOT supported. A direct cast is close to")
        print("  harmless on this data. Rewrite the sentence: percentile windowing")
        print("  is a safe default that behaves identically when an image already")
        print("  spans the range, not a repair for damage that occurs here.")
    elif len(big) / len(rows) < 0.25:
        print(f"  The claim holds for a minority of images ({100*len(big)/len(rows):.1f}%).")
        print("  State that proportion in the paper rather than implying it is")
        print("  universal, and draw the supplementary figure from that subset.")
    else:
        print("  The claim holds broadly. The earlier single image was unlucky.")
        print("  Redraw the figure restricted to images with mean_abs_diff > 10.")

    if cfg.out:
        import csv
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        with cfg.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
