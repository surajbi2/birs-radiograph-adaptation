"""Write the GRAZPEDWRI-DX crosstab of the two label definitions.

make_paper_outputs.py reads only registered result files and never loads the image
index, so this script writes the abnormal-versus-fracture crosstab it needs.
Run it after any rebuild of the index.

Output:
    results/transfer/graz_label_crosstab.csv
"""
from pathlib import Path
import pandas as pd
import mskdata, mskmodel as M

def main() -> int:
    idx = pd.read_csv(M.DEFAULT_INDEX, comment="#", low_memory=False)
    g = idx[idx.dataset == "graz"]
    both = int(((g.label_abnormal == 1) & (g.label_fracture == 1)).sum())
    abn_only = int(((g.label_abnormal == 1) & (g.label_fracture == 0)).sum())
    frac_only = int(((g.label_abnormal == 0) & (g.label_fracture == 1)).sum())
    neither = int(((g.label_abnormal == 0) & (g.label_fracture == 0)).sum())
    n = len(g)
    df = pd.DataFrame([{
        "dataset": "graz", "n_images": n,
        "abnormal_and_fracture": both, "abnormal_not_fracture": abn_only,
        "fracture_not_abnormal": frac_only, "neither": neither,
        "n_disagree": abn_only + frac_only,
        "frac_disagree": (abn_only + frac_only) / n if n else float("nan")}])
    # FracAtlas budget-pool size: the b14 CSV stores per-budget sample sizes,
    # not the pool it draws from, so the pool is recomputed here.
    try:
        from fracatlas_budget_cells import frac_pools, EVAL_FRACTION
        from dataset_shift import load_index
        class _C: index = str(M.DEFAULT_INDEX)
        _tr, _va, _nd, _nu = frac_pools(load_index(_C), "XR_HAND", False, EVAL_FRACTION)
        df["frac_pool_studies"] = int(_tr.study.nunique())
        df["frac_pool_images"] = int(len(_tr))
    except Exception as e:                      # pool is optional, never fatal
        print(f"[graz-labels] FracAtlas pool not computed: {e}")
        df["frac_pool_studies"] = float("nan")
        df["frac_pool_images"] = float("nan")

    out = Path(M.DEFAULT_RESULTS) / "transfer" / "graz_label_crosstab.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    mskdata.write_csv(df, out, M.provenance({"script": "graz_label_crosstab.py v1"}))
    print(f"[graz-labels] {abn_only + frac_only} of {n} images disagree "
          f"({100*(abn_only+frac_only)/n:.3f}%); fracture_not_abnormal={frac_only}")
    print(f"wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
