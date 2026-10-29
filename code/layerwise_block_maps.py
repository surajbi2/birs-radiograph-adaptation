#!/usr/bin/env python3
"""Block-level freezing maps for the layer-wise unfreezing experiment.

Defines, for each timm backbone, an ordered list of parameter blocks from the
stem to the last stage, and helpers that freeze all but the first or last k of
them. Each map partitions all trunk parameters exactly once; two easily missed
groups are included explicitly (ConvNeXt ``head.norm`` and Swin's final
``norm``).

Library module; used by layerwise_unfreezing.py.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch.nn as nn

# Bottom -> top. Paths are relative to the BinaryNet trunk (i.e. `model.trunk`).
# Verified: these partition 100% of trunk parameters for every backbone.
BLOCK_MAPS: Dict[str, List[List[str]]] = {
    "densenet121": [
        ["features.conv0", "features.norm0"],                    # stem
        ["features.denseblock1", "features.transition1"],
        ["features.denseblock2", "features.transition2"],
        ["features.denseblock3", "features.transition3"],
        ["features.denseblock4", "features.norm5"],
    ],
    "resnet50": [
        ["conv1", "bn1"],                                        # stem
        ["layer1"],
        ["layer2"],
        ["layer3"],
        ["layer4"],
    ],
    "convnext_tiny": [
        ["stem"],
        ["stages.0"],
        ["stages.1"],
        ["stages.2"],
        ["stages.3", "norm_pre", "head.norm"],                   # head.norm has params
    ],
    "swin_tiny_patch4_window7_224": [
        ["patch_embed"],
        ["layers.0"],
        ["layers.1"],
        ["layers.2"],
        ["layers.3", "norm"],                                    # top-level norm
    ],
}

N_BLOCKS = 5
DIRECTIONS = ("top_down", "bottom_up")


# ------------------------------------------------------------------ helpers --

def _resolve(root: nn.Module, path: str) -> nn.Module:
    mod = root
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def _blocks_of(model: nn.Module, arch: str) -> List[List[nn.Module]]:
    """Resolve every block path against model.trunk. Fails loudly."""
    if arch not in BLOCK_MAPS:
        raise KeyError(f"unknown arch {arch!r}; expected one of {list(BLOCK_MAPS)}")
    trunk = getattr(model, "trunk", None)
    if trunk is None:
        raise AttributeError(
            "model has no .trunk -- this module targets mskmodel.build_model()'s "
            "BinaryNet(trunk=timm model, head=nn.Linear). Got "
            f"{type(model).__name__} with children "
            f"{[n for n, _ in model.named_children()]}")
    out = []
    for paths in BLOCK_MAPS[arch]:
        mods = []
        for p in paths:
            try:
                mods.append(_resolve(trunk, p))
            except AttributeError as e:
                raise AttributeError(
                    f"block path {p!r} does not resolve for {arch}. timm may have "
                    f"renamed modules. Trunk children: "
                    f"{[n for n, _ in trunk.named_children()]}") from e
        out.append(mods)
    return out


def verify_partition(model: nn.Module, arch: str) -> Dict[str, int]:
    """
    Assert the blocks cover every trunk parameter exactly once.

    This is the guard against a silently wrong map. If timm renames or
    restructures a model, this raises instead of quietly training the wrong
    subset of weights.
    """
    blocks = _blocks_of(model, arch)
    trunk = model.trunk
    owned: Dict[str, int] = {}
    per_block: List[int] = []
    for i, mods in enumerate(blocks):
        n = 0
        for path, mod in zip(BLOCK_MAPS[arch][i], mods):
            for pn, p in mod.named_parameters():
                key = f"{path}.{pn}"
                if key in owned:
                    raise AssertionError(
                        f"{arch}: parameter {key} claimed by blocks "
                        f"{owned[key]} and {i}")
                owned[key] = i
                n += p.numel()
        per_block.append(n)

    total = sum(p.numel() for p in trunk.parameters())
    covered = sum(per_block)
    if covered != total:
        named = {n for n, _ in trunk.named_parameters()}
        raise AssertionError(
            f"{arch}: blocks cover {covered:,} of {total:,} trunk parameters. "
            f"Uncovered: {sorted(named - set(owned))[:12]}")
    return {"total_trunk": total, "per_block": per_block,
            "head": sum(p.numel() for p in model.head.parameters())}


# -------------------------------------------------------------------- core --

def apply_freezing(model: nn.Module, arch: str, depth: int,
                   direction: str = "top_down") -> Tuple[List[nn.Module], int]:
    """
    Set requires_grad across the trunk. The head is always trainable.

    Returns (frozen_modules, n_trainable_params). Pass frozen_modules to
    train_with_frozen() on every epoch.
    """
    if not 0 <= depth <= N_BLOCKS:
        raise ValueError(f"depth must be in 0..{N_BLOCKS}, got {depth}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")

    verify_partition(model, arch)          # cheap; catches a bad map every time
    blocks = _blocks_of(model, arch)

    unfrozen = (set(range(N_BLOCKS - depth, N_BLOCKS)) if direction == "top_down"
                else set(range(depth)))

    for p in model.parameters():
        p.requires_grad = False

    frozen: List[nn.Module] = []
    for i, mods in enumerate(blocks):
        trainable = i in unfrozen
        for m in mods:
            for p in m.parameters():
                p.requires_grad = trainable
            if not trainable:
                frozen.append(m)

    for p in model.head.parameters():      # head always trainable
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, n_trainable


def train_with_frozen(model: nn.Module, frozen_modules: Sequence[nn.Module]
                      ) -> nn.Module:
    """
    Replacement for model.train(). Puts the model in train mode, then forces
    every frozen module back to eval() so BatchNorm running statistics and
    dropout inside frozen blocks stay fixed.

    MUST be called at the start of every epoch.
    """
    model.train()
    for m in frozen_modules:
        m.eval()
    return model


def assert_freezing_correct(model: nn.Module, frozen_modules: Sequence[nn.Module],
                            expect_trainable: int = -1) -> Dict[str, int]:
    """
    Post-condition check, run once per configuration.

      1. every frozen BatchNorm is in eval mode after train_with_frozen()
      2. no frozen parameter has requires_grad=True
      3. trainable parameter count matches expectation, if given (>= 0)
    """
    train_with_frozen(model, frozen_modules)

    bad_bn = [type(m).__name__
              for parent in frozen_modules
              for m in parent.modules()
              if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
    if bad_bn:
        raise AssertionError(
            f"{len(bad_bn)} frozen BatchNorm modules still in train mode "
            f"({set(bad_bn)}). Running statistics would adapt to the target "
            f"distribution, smuggling AdaBN into this arm.")

    leaked = [f"{type(parent).__name__}.{pn}"
              for parent in frozen_modules
              for pn, p in parent.named_parameters() if p.requires_grad]
    if leaked:
        raise AssertionError(f"frozen modules have trainable params: {leaked[:8]}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if expect_trainable >= 0 and n_train != expect_trainable:
        raise AssertionError(
            f"trainable params {n_train:,} != expected {expect_trainable:,}")

    n_bn_frozen = sum(1 for parent in frozen_modules for m in parent.modules()
                      if isinstance(m, nn.modules.batchnorm._BatchNorm))
    return {"n_trainable": n_train, "n_frozen_bn": n_bn_frozen}


def capacity_table(build_fn, arch: str) -> List[Dict]:
    """
    Trainable parameter counts for both directions at every depth.

    Run BEFORE the experiment and report alongside results. The two arms are
    not capacity matched; that has to be visible in the paper rather than
    assumed away.

    build_fn: zero-arg callable returning a fresh model (e.g.
              lambda: mskmodel.build_model(arch)).
    """
    rows = []
    model = build_fn()
    info = verify_partition(model, arch)
    denom = info["total_trunk"] + info["head"]
    for direction in DIRECTIONS:
        for depth in range(N_BLOCKS + 1):
            _, n = apply_freezing(model, arch, depth, direction)
            rows.append({"arch": arch, "direction": direction, "depth": depth,
                         "trainable_params": n,
                         "frac_of_model": round(n / denom, 5)})
    return rows


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/path/to/birs/code")
    import pandas as pd
    import mskmodel as M

    rows: List[Dict] = []
    for arch in BLOCK_MAPS:
        m = M.build_model(arch)
        info = verify_partition(m, arch)
        print(f"{arch:<32} trunk={info['total_trunk']:>12,} "
              f"blocks={[format(n, ',') for n in info['per_block']]}  "
              f"PARTITION OK")
        rows += capacity_table(lambda a=arch: M.build_model(a), arch)
        del m
    print()
    print(pd.DataFrame(rows).pivot_table(
        index=["arch", "depth"], columns="direction",
        values="trainable_params").to_string())
