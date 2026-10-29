#!/usr/bin/env python3
"""Remove duplicated rows from oracle_readout_cv_XR_WRIST.csv.

A second run of oracle_readout_cv with --dump-dir appended a copy of every row. Means are
unaffected, but counts doubled and standard deviations shrank. Rows are
de-duplicated on the cell key, keeping the first occurrence (the original run).
The script refuses to write if duplicated rows differ, and checks the result's
shape before writing.

Usage:
    python dedupe_oracle_results.py <path to oracle_readout_cv_XR_WRIST.csv>
"""
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

KEY = ("source", "target", "backbone", "seed", "C", "setting")
EXPECTED = 735          # 7 targets x 3 backbones x 5 seeds x 7 settings


def main() -> int:
    path = Path(sys.argv[1]).expanduser()
    lines = path.read_text().splitlines(keepends=True)
    header_i = next(i for i, l in enumerate(lines) if not l.startswith("#"))
    preamble, body = lines[:header_i], lines[header_i:]

    rows = list(csv.DictReader(body))
    print(f"{path.name}: {len(rows)} data rows")

    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEY)].append(r)

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  {len(groups)} distinct cells, {len(dupes)} duplicated")
    if not dupes:
        print("  nothing to do")
        return 0

    # `minutes` is wall-clock time for the cell. It cannot match between runs
    # and says nothing about whether the numbers reproduced, so it is excluded
    # from the comparison and reported separately.
    VOLATILE = {"minutes"}

    # Report which columns actually moved, rather than only that something did.
    moved = defaultdict(lambda: [0, 0.0])       # column -> [n_cells, max_absdiff]
    for k, v in dupes.items():
        for other in v[1:]:
            for col in v[0]:
                a, b = v[0][col], other[col]
                if a == b:
                    continue
                moved[col][0] += 1
                try:
                    moved[col][1] = max(moved[col][1], abs(float(a) - float(b)))
                except (TypeError, ValueError):
                    moved[col][1] = float("nan")
    if moved:
        print("  columns differing between copies:")
        for col, (n, d) in sorted(moved.items()):
            tag = "  (volatile, ignored)" if col in VOLATILE else "  <-- SUBSTANTIVE"
            print(f"    {col:<22} {n:5d} cell(s)  max |diff| = {d:.3e}{tag}")

    substantive = sorted(set(moved) - VOLATILE)
    if substantive:
        print(f"  !! {len(substantive)} substantive column(s) differ: "
              f"{substantive}")
        print("  The second run did not reproduce the first. Do not dedupe; "
              "move the file aside and re-run oracle_readout_cv cleanly.")
        return 1
    print("  duplicate copies agree on every column except wall-clock timing, "
          "so the run reproduced exactly")

    kept = [v[0] for v in groups.values()]
    if len(kept) != EXPECTED:
        print(f"  !! {len(kept)} cells after dedupe, expected {EXPECTED}. "
              "Not writing.")
        return 1

    backup = path.with_name(path.stem + "_duplicated.csv")
    shutil.copy2(path, backup)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        fh.writelines(preamble)
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)
    print(f"  wrote {len(kept)} rows; previous file kept at {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
