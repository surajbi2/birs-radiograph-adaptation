#!/usr/bin/env python3
"""Merge the two wrist-source probe files into readout_probe_XR_WRIST_merged.csv.

The wrist-source probe was run twice by readout_probe.py: DenseNet-121 and
Swin-Tiny into ``readout_probe.csv`` (an older schema without a ``source``
column, filled here with ``XR_WRIST``), and ConvNeXt-Tiny into
``readout_probe_XR_WRIST.csv``. The files are concatenated in that order and
de-duplicated on the cell key, keeping the first copy. The backbones are
disjoint, so no row is dropped; the script stops if that ever changes.

Usage:
    python merge_wrist_probe.py --transfer-dir ../results/transfer \
        --out ../results/transfer/readout_probe_XR_WRIST_merged.csv

CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

OLD = "readout_probe.csv"
NEW = "readout_probe_XR_WRIST.csv"
KEY = ["backbone", "seed", "target", "budget", "resample"]
SOURCE = "XR_WRIST"


def provenance(path: Path) -> dict:
    """The JSON on the leading '#' line that readout_probe.py writes."""
    first = path.read_text(encoding="utf-8").splitlines()[0]
    if not first.startswith("#"):
        raise SystemExit(f"!! {path.name} has no provenance line")
    return json.loads(first[1:]).get("provenance", {})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transfer-dir", required=True,
                    help="folder holding the two readout_probe probe csvs")
    ap.add_argument("--out", required=True)
    cfg = ap.parse_args()
    d = Path(cfg.transfer_dir).expanduser()

    for name in (OLD, NEW):
        src = provenance(d / name).get("source")
        if src != SOURCE:
            raise SystemExit(f"!! {name} provenance says source={src!r}, expected {SOURCE}")

    old = pd.read_csv(d / OLD, comment="#")
    if "source" in old.columns:
        raise SystemExit(f"!! {OLD} already has a 'source' column; not the old schema")
    old["source"] = SOURCE
    new = pd.read_csv(d / NEW, comment="#")

    both = pd.concat([old, new], ignore_index=True)
    merged = both.drop_duplicates(KEY)
    dropped = len(both) - len(merged)
    if dropped:
        raise SystemExit(f"!! {dropped} duplicate cells between the two runs; which copy "
                         f"survives would depend on file order. Resolve before merging.")

    out = Path(cfg.out).expanduser()
    merged.to_csv(out, index=False)
    print(f"[merge] {OLD}: {len(old)} rows, {NEW}: {len(new)} rows, "
          f"{dropped} duplicates -> {len(merged)} rows")
    print(merged.groupby("backbone").size().to_string())
    print(f"[merge] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
