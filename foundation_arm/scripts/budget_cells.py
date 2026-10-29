"""Compute the budget cells for the foundation encoders from cached features.

Re-expresses the per-cell readout logic of readout_shrinkage, fracatlas_budget_cells, readout_penalty_sweep and
l2sp_comparator from their own functions (all imported, none modified):

    pools, cache_path, load_unit, cells, l2sp fits     l2sp_comparator
    joint (C, lam) selection, raw-space refit, blend   readout_shrinkage
    own-penalty refit                                  readout_penalty_sweep
    budgets and resamples                              readout_probe
    study-level scoring                                shortcut_reliance

Budget samples are drawn with the random stream keyed on DenseNet-121
(``--rng-bb``), so every encoder is scored on the same local samples as the
ImageNet backbones.

Per cell: zeroshot, refit_C1, refit_nestedC (joint-penalty refit),
shrink_nested, shrink_lam* (full lam grid), refit_ownC (own-penalty refit),
l2sp_cv and l2sp_C1. Policies (schedules, caps, abstention) are computed later
from the CSV.

Modes:
    gate   on the ImageNet cache: must reproduce every stored DenseNet-121 row of
           readout_shrinkage, fracatlas_budget_cells and l2sp_comparator within 1e-6, the own-penalty refit
           of readout_penalty_sweep, and the stored schedule; writes gate_pass.json
    run    on the foundation-encoder cache; refuses unless gate_pass.json passed
           for this exact script

Output:
    cells/cells_<tag>.csv

Usage:
    python budget_cells.py gate
    python budget_cells.py run --cache-dir <cache>/arm_a --backbones skelex,rad-dino,medsiglip --tag arm_a

CPU only.
"""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path.home() / "diagnosis" / "code"
sys.path.insert(0, str(CODE))
FA = Path("/path/to/foundation_arm_workdir")
import mskmodel as M                                             # noqa: E402
from train_source_models import CsvAppender                              # noqa: E402
from readout_probe import BUDGETS, N_RESAMPLE                    # noqa: E402
from readout_shrinkage import (C_GRID, LAMBDA_GRID, PUBLISHED_C,  # noqa: E402
                                 choose_c_lam_nested, raw_space_readout, blend)
from readout_penalty_sweep import fit_scores, C_GRID as C_GRID_B11  # noqa: E402
from l2sp_comparator import (pools, load_unit, cells, cache_path,    # noqa: E402
                            choose_c_l2sp, l2sp_fit, L2SP_C_GRID)
from birs_schedule import per_budget_stats, fit_schedule  # noqa: E402
from shortcut_reliance import score                              # noqa: E402

RESULTS = Path(M.DEFAULT_RESULTS) / "transfer"
STORED = {"b11": RESULTS / "readout_penalty_sweep_XR_WRIST.csv",
          "b13": RESULTS / "readout_shrinkage_XR_WRIST.csv",
          "b14": RESULTS / "fracatlas_budget_cells_XR_WRIST_XR_HAND.csv",
          "b17": RESULTS / "l2sp_comparator_XR_WRIST.csv",
          "b16": RESULTS / "birs_schedule.csv"}
PAPER_CACHE = Path("/path/to/imagenet_feature_cache")
GATE_BB = "densenet121"
GATE_TOL = 1e-6
MAX_JOBS = 48                       # shared cores
SOURCE = "XR_WRIST"
K = ["target", "backbone", "seed", "budget", "resample", "setting"]
NUMERIC = ["lam", "C", "chosen_C", "chosen_lam", "n_train_studies", "auc_study",
           "auc_image", "zeroshot_auc", "delta_vs_zeroshot", "harm"]
B11_COLS = ["chosen_C", "n_train_studies", "auc_study", "delta_vs_zeroshot", "zeroshot_auc"]
OWN = "refit_ownC"
ROW_COLUMNS = (["source", "target", "backbone", "rng_bb", "seed", "budget", "resample",
                "setting"] + NUMERIC + ["cv_auc", "converged", "n_iter"])
TAU = 0.05                          # gate's b16 check only


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ================================================================ one unit ===

def unit_rows(args):
    """All rows of one (backbone, seed, target): b13/b14 settings, b11 own-penalty refit, b17 l2sp."""
    cache_dir, bb, rng_bb, seed, t = args
    z, tr, va, tag = load_unit(cache_dir, _POOLS, bb, seed, t)    # uid check inside
    ftr32, fva32 = z["ftr"], z["fva"]                              # as extracted (b11 input)
    ftr = ftr32.astype(np.float64)
    fva = fva32.astype(np.float64)
    zva, yva, st = z["zva"], z["yva"], va.study.to_numpy()
    w_src, b_src = z["w_src"], float(z["b_src"])
    zs = score(zva, yva, st)["auc_study"]
    if abs(zs - float(z["zs"])) > 0:
        raise SystemExit(f"!! {bb}/s{seed}/{t}: zs {zs!r} != cached {float(z['zs'])!r}")
    base = {"source": SOURCE, "target": t, "backbone": bb, "rng_bb": rng_bb, "seed": seed}
    rows = [{**base, "budget": 0, "resample": 0, "setting": "zeroshot", "n_train_studies": 0,
             "auc_study": zs, "auc_image": score(zva, yva, st)["auc_image"],
             "zeroshot_auc": zs, "delta_vs_zeroshot": 0.0, "harm": 0}]

    def row(b, r, n_st, setting, lam, C, c_n, lam_n, s_, **extra):
        a = s_["auc_study"]
        return {**base, "budget": b, "resample": r, "setting": setting, "lam": lam, "C": C,
                "chosen_C": c_n, "chosen_lam": lam_n, "n_train_studies": n_st,
                "auc_study": a, "auc_image": s_["auc_image"], "zeroshot_auc": zs,
                "delta_vs_zeroshot": a - zs, "harm": int(a < zs) if np.isfinite(a) else 0, **extra}

    # b13 main() / b14 main() cell loop; cells() applies the same budget and
    # single-class skips (b13/b11: pool studies < b; b14: budgets <= pool studies)
    for b, r, idx, y, grp, n_st in cells(tr, BUDGETS, N_RESAMPLE, tag, rng_bb, seed):
        fsub = ftr[idx]
        c_n, lam_n, _ = choose_c_lam_nested(fsub, y, grp, w_src, b_src,
                                            list(C_GRID), list(LAMBDA_GRID))
        fits = []
        w1, b1 = raw_space_readout(fsub, y, PUBLISHED_C)
        if w1 is not None:
            fits.append(("refit_C1", 1.0, PUBLISHED_C, w1, b1))
        w_nc, b_nc = raw_space_readout(fsub, y, c_n)
        if w_nc is not None:
            fits.append(("refit_nestedC", 1.0, c_n, w_nc, b_nc))
            wS, bS = blend(w_src, b_src, w_nc, b_nc, lam_n)
            fits.append(("shrink_nested", lam_n, c_n, wS, bS))
            for lam in LAMBDA_GRID:
                wL, bL = blend(w_src, b_src, w_nc, b_nc, lam)
                fits.append((f"shrink_lam{lam:g}", float(lam), c_n, wL, bL))
        for setting, lam, C, w, bb_ in fits:
            s_ = score(zva if float(lam) == 0.0 else fva @ w + bb_, yva, st)
            rows.append(row(b, r, n_st, setting, lam, C, c_n, lam_n, s_))
        # b11 main(): own-penalty refit, float32 features as b11 passed them
        own = fit_scores(ftr32[idx], y, grp, fva32, list(C_GRID_B11))
        if own:
            c_own = own["_chosen_C"]
            rows.append(row(b, r, n_st, OWN, 1.0, c_own, c_own, np.nan,
                            score(own["nestedCV"], yva, st)))
        # b17 fit_unit: l2sp_cv and l2sp_C1 (lam/chosen_lam NaN as b17)
        c_cv, cv_auc = choose_c_l2sp(fsub, y, grp, w_src, b_src, list(L2SP_C_GRID))
        for setting, C in (("l2sp_cv", c_cv), ("l2sp_C1", PUBLISHED_C)):
            w, bb_, ok, nit = l2sp_fit(fsub, y, w_src, b_src, C)
            rows.append(row(b, r, n_st, setting, np.nan, C, c_cv, np.nan, score(fva @ w + bb_, yva, st),
                            cv_auc=cv_auc if setting == "l2sp_cv" else np.nan,
                            converged=int(ok), n_iter=nit))
    return (bb, seed, t), rows


_POOLS = None


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def run_units(cache_dir, bbs, rng_bb, seeds, targets, jobs):
    from multiprocessing import get_context
    args = [(cache_dir, bb, rng_bb or bb, s, t) for bb in bbs for s in seeds for t in targets]
    missing = [a for a in args if not cache_path(cache_dir, a[1], a[3], a[4]).exists()]
    if missing:
        raise SystemExit(f"!! {len(missing)} unit file(s) missing, e.g. {missing[:2]}")
    t0, done = time.time(), {}
    with get_context("fork").Pool(jobs, initializer=_init_worker) as pool:
        for key, rows in pool.imap_unordered(unit_rows, args):
            done[key] = rows
            print(f"  [cells] {len(done)}/{len(args)} {key} {len(rows)} rows "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return [r for a in args for r in done[(a[1], a[3], a[4])]]


# ==================================================================== gate ===

def read(name):
    return pd.read_csv(STORED[name], comment="#", low_memory=False)


def stored_rows(bb):
    """b13 + b14 + b17 rows for bb, unique on K."""
    d = pd.concat([x[x.backbone == bb] for x in (read("b13"), read("b14"), read("b17"))],
                  ignore_index=True)
    dup = d.duplicated(K, keep=False)
    if dup.any():
        raise SystemExit(f"!! stored b13/b14/b17 rows not unique on {K}: {int(dup.sum())} rows")
    return d.set_index(K)


def stored_own(bb):
    """b11 nestedCV rows for bb as refit_ownC; exact duplicates collapsed, conflicts fail."""
    d = read("b11")
    d = d[(d.backbone == bb) & (d.setting == "nestedCV")].copy()
    n_raw = len(d)
    d["setting"] = OWN
    d = d.drop_duplicates(K + B11_COLS)
    if d.duplicated(K).any():
        raise SystemExit(f"!! b11 nestedCV has conflicting duplicates on {K}")
    return d.set_index(K), n_raw


def compare(got, ref, cols, label, fail):
    """Max |diff| per column over the common keys; NaN on one side only = inf."""
    common = got.index.intersection(ref.index)
    worst = {}
    for c in cols:
        a = pd.to_numeric(got.loc[common, c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(ref.loc[common, c], errors="coerce").to_numpy(float)
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)
        worst[c] = float(d.max()) if len(d) else 0.0
        if worst[c] > GATE_TOL:
            i = int(np.argmax(d))
            fail.append(f"{label} {c}: {common[i]} got {a[i]!r} stored {b[i]!r}")
    return worst


def do_gate(cfg):
    got = pd.DataFrame(run_units(PAPER_CACHE, [GATE_BB], GATE_BB, list(M.SEEDS),
                                 cfg.targets, cfg.jobs)).set_index(K)
    is_own = got.index.get_level_values("setting") == OWN
    g_main, g_own = got[~is_own], got[is_own]
    fail = []

    # b13 / b14 / b17: identical key sets, every numeric column
    ref = stored_rows(GATE_BB)
    only_got, only_ref = g_main.index.difference(ref.index), ref.index.difference(g_main.index)
    worst = compare(g_main, ref, NUMERIC, "b13/b14/b17", fail)

    # b11 own-penalty refit on the targets b11 ran
    ref11, n11_raw = stored_own(GATE_BB)
    b11_targets = set(ref11.index.get_level_values("target"))
    covered = g_own.index.get_level_values("target").isin(b11_targets)
    no_ref = g_own[~covered]
    only_got11 = g_own[covered].index.difference(ref11.index)
    only_ref11 = ref11.index.difference(g_own[covered].index)
    worst11 = compare(g_own[covered], ref11, B11_COLS, "b11 refit_ownC", fail)

    # b16: the stored schedule must be what fit_schedule gives on the stored grid
    g = pd.concat([read("b13"), read("b14")])
    g = g[g.setting.astype(str).str.match(r"^shrink_lam[0-9.]+$") & (g.budget > 0)]
    cal = g[g.target.astype(str).str.match(r"^XR_[A-Z]+$")].assign(lam=lambda x: x.lam.astype(float))
    sched, _ = fit_schedule(per_budget_stats(cal), TAU, True)
    s16 = read("b16")
    s16 = {int(r.budget): float(r.lam) for r in s16[s16.rule == "schedule_lam"].itertuples()}
    if sched != s16:
        fail.append(f"b16 schedule_lam: refit {sched} != stored {s16}")

    extra = [str(x) for x in list(only_got[:5]) + list(only_ref[:5])
             + list(only_got11[:5]) + list(only_ref11[:5])]
    ok = not fail and not extra
    res = {"script_sha256": sha256(Path(__file__).resolve()), "backbone": GATE_BB, "tol": GATE_TOL,
           "b13_b14_b17": {"n_runner": int(len(g_main)), "n_stored": int(len(ref)),
                           "only_in_runner": len(only_got), "only_in_stored": len(only_ref),
                           "max_abs_diff": worst},
           "b11_refit_ownC": {"file": STORED["b11"].name, "n_stored_raw": n11_raw,
                              "n_stored_unique": int(len(ref11)), "targets": sorted(b11_targets),
                              "column_map": {"setting": "nestedCV -> refit_ownC",
                                             **{c: c for c in B11_COLS}},
                              "n_runner_covered": int(covered.sum()),
                              "only_in_runner": len(only_got11), "only_in_stored": len(only_ref11),
                              "max_abs_diff": worst11,
                              "no_reference": {"n": int(len(no_ref)), "targets":
                                               sorted(set(no_ref.index.get_level_values("target")))}},
           "b16_schedule": sched, "pass": ok}
    for lab, w in (("b13/b14/b17", worst), ("b11 refit_ownC", worst11)):
        for k, v in w.items():
            print(f"[gate] {lab:<15s} max |diff| {k:<18s} {v:.3e}")
    print(f"[gate] b13/b14/b17 rows runner {len(g_main)} stored {len(ref)}; only runner "
          f"{len(only_got)}, only stored {len(only_ref)}")
    print(f"[gate] b11 nestedCV stored {n11_raw} ({len(ref11)} unique), runner covered "
          f"{int(covered.sum())}; only runner {len(only_got11)}, only stored {len(only_ref11)}; "
          f"no reference {len(no_ref)} rows on {res['b11_refit_ownC']['no_reference']['targets']}")
    print(f"[gate] b16 schedule {sched}")
    print("[gate] PASS" if ok else "[gate] FAIL\n  " + "\n  ".join(fail[:20] + extra))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "gate_pass.json").write_text(json.dumps(res, indent=2, default=str))
    return 0 if ok else 1


# ================================================================= run ===

def do_run(cfg):
    gp = cfg.out_dir / "gate_pass.json"
    me = sha256(Path(__file__).resolve())
    g = json.loads(gp.read_text()) if gp.exists() else {}
    if not (g.get("pass") and g.get("script_sha256") == me):
        raise SystemExit(f"!! no gate PASS for this script sha ({me[:8]}); run `gate` first")
    out = cfg.out_dir / f"cells_{cfg.tag}.csv"
    if out.exists():
        raise SystemExit(f"!! {out} exists; move it aside first (no appends)")
    rows = run_units(cfg.cache_dir, cfg.backbones, cfg.rng_bb, cfg.seeds, cfg.targets, cfg.jobs)
    prov = M.provenance({"script": "budget_cells.py run", "script_sha256": me,
                         "cache_dir": str(cfg.cache_dir), "rng_bb": cfg.rng_bb or "same as backbone",
                         "budgets": list(BUDGETS), "resamples": N_RESAMPLE, "c_grid": list(C_GRID),
                         "c_grid_own": list(C_GRID_B11), "lam_grid": list(LAMBDA_GRID),
                         "l2sp_c_grid": list(L2SP_C_GRID), "gate": str(gp)})
    CsvAppender(out, prov).write([{c: r.get(c, np.nan) for c in ROW_COLUMNS} for r in rows])
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


# ==================================================================== main ===

def main():
    global _POOLS
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "run"])
    ap.add_argument("--cache-dir", type=Path, default=None, help="run: FA features dir (arm_a or arm_b)")
    ap.add_argument("--backbones", default="skelex,rad-dino,medsiglip")
    ap.add_argument("--rng-bb", default=GATE_BB, help="rng key for budget samples (FA_PLAN section 4)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in M.SEEDS))
    ap.add_argument("--tag", default=None, help="run: output name cells_<tag>.csv")
    ap.add_argument("--out-dir", type=Path, default=FA / "cells")
    ap.add_argument("--index", default=str(M.DEFAULT_INDEX))
    ap.add_argument("--jobs", type=int, default=32)
    cfg = ap.parse_args()
    if not 1 <= cfg.jobs <= MAX_JOBS:
        raise SystemExit(f"--jobs must be 1..{MAX_JOBS} (shared cores)")
    cfg.backbones = [b.strip() for b in cfg.backbones.split(",") if b.strip()]
    cfg.seeds = [int(s) for s in cfg.seeds.split(",") if s.strip()]
    index = pd.read_csv(cfg.index, comment="#", low_memory=False)
    _POOLS = pools(index)                  # built once, inherited by fork
    cfg.targets = list(_POOLS)             # 6 MURA + GRAZ_initial + MFXR + FRAC_XR_HAND
    if cfg.mode == "gate":
        return do_gate(cfg)
    if cfg.cache_dir is None or cfg.tag is None:
        raise SystemExit("run needs --cache-dir and --tag")
    return do_run(cfg)


if __name__ == "__main__":
    sys.exit(main())
