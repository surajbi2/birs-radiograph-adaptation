"""Source-model wrist validation AUCs, mean and SD over the five seeds.

    foundation encoders (arm A)   from foundation_arm/arm_a/<encoder>/seed<s>/result.json
    ImageNet backbones            from results/source_models/anatomy_transfer_arms_merged.csv
                                  (source = target = XR_WRIST, condition "ceiling")

Every input file is hashed into the output, as in paper_numbers.json.

Output:
    output/extra_numbers.json

Usage:
    python source_model_aucs.py
"""
import hashlib
import json
import statistics
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FA = ROOT / "foundation_arm"
ARM_A = {"skelex": FA / "arm_a/skelex", "raddino": FA / "arm_a/rad-dino",
         "medsiglip": FA / "arm_a/medsiglip", "dinov3": FA / "arm_a/dinov3"}
STAGE_A3 = ROOT / "results" / "source_models" / "anatomy_transfer_arms_merged.csv"
SEEDS = [42, 123, 456, 789, 2024]
BB = {"densenet121": "densenet121", "convnext_tiny": "convnext_tiny", "swin_tiny": "swin_tiny_patch4_window7_224"}

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out, inputs = {}, {}


def put(key, value, note):
    assert key not in out, key
    out[key] = {"value": value, "note": note}


for enc, d in ARM_A.items():
    aucs = []
    for s in SEEDS:
        p = d / f"seed{s}" / "result.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        assert r["encoder"].replace("-", "") == enc and r["seed"] == s and r["smoke"] is False, p
        assert r["valid"]["n_studies"] == 237 and r["valid"]["n_images"] == 659, p
        assert r["hparams"]["lr"] == 5e-05 and r["hparams"]["epochs"] == 10 and r["hparams"]["batch_size"] == 32, p
        aucs.append(r["valid"]["auc_study"])
        inputs[str(p)] = sha(p)
    put(f"cbm_src_auc_{enc}_mean", statistics.mean(aucs), f"arm A {enc}: wrist validation study AUC, mean of 5 seeds")
    put(f"cbm_src_auc_{enc}_sd", statistics.stdev(aucs), f"arm A {enc}: wrist validation study AUC, SD of 5 seeds")

a3 = pd.read_csv(STAGE_A3, comment="#")
inputs[str(STAGE_A3)] = sha(STAGE_A3)
w = a3[(a3.source == "XR_WRIST") & (a3.target == "XR_WRIST") & (a3.condition == "ceiling")]
for k, bb in BB.items():
    g = w[w.backbone == bb]
    assert sorted(g.seed) == SEEDS and (g.n_studies == 237).all(), bb
    put(f"cbm_src_auc_{k}_mean", float(g.auc_study.mean()), f"{bb}: wrist validation study AUC, mean of 5 seeds")
    put(f"cbm_src_auc_{k}_sd", float(g.auc_study.std(ddof=1)), f"{bb}: wrist validation study AUC, SD of 5 seeds")

dst = HERE / "output" / "extra_numbers.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps({"script_sha256": sha(Path(__file__)), "inputs": inputs, "numbers": out}, indent=1),
               encoding="utf-8")
for k, v in out.items():
    print(f"{k:32s} {v['value']:.4f}")
print(f"wrote {dst.name} ({len(out)} values, {len(inputs)} inputs hashed)")
