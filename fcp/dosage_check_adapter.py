"""fcp/dosage_check_adapter.py -- free check that adapter switching worked (step J1).

The rank-1 checkpoints only modify down_proj in block --adapter_layer (21 for
R1_0_1_0_extended_train; see each checkpoint's adapter_config.json). The
last-prompt-token residual stream at layers < adapter_layer must therefore be
identical across every checkpoint and the base model, and should differ from
adapter_layer on. Prints the max |h_condition - h_base| per layer.

Exception: a checkpoint whose lora_B is all zeros (the LoRA initial state) is
identical to base at every layer. R1_0_1_0_extended_train's checkpoint-1 is such a
case (checked 2026-10-21), so "differs from base" fails for step_1 by design.

Usage:
    python fcp/dosage_check_adapter.py --dir fcp/activations/dosage --adapter_layer 21
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(args):
    d = Path(args.dir)
    meta = json.load(open(d / "meta.json"))
    idx = pd.read_parquet(d / "index.parquet")
    acts = np.memmap(args.acts or d / "activations.dat", dtype="float32", mode="r",
                     shape=(meta["n_rows"], meta["d_model"]))
    assert len(idx) == meta["n_rows"], (len(idx), meta["n_rows"])
    base = idx[idx.condition == "base"].set_index(["question_id", "layer"])["row_idx"]
    layers = sorted(idx.layer.unique())
    out = {}
    for cond in [c for c in idx.condition.unique() if c != "base"]:
        sub = idx[idx.condition == cond].set_index(["question_id", "layer"])["row_idx"]
        out[cond] = []
        for layer in layers:
            keys = [k for k in sub.index if k[1] == layer]
            a = acts[[sub[k] for k in keys]]
            b = acts[[base[k] for k in keys]]
            out[cond].append(float(np.abs(a - b).max()))
    table = pd.DataFrame(out, index=layers)
    table.index.name = "layer"
    pd.set_option("display.width", 200)
    print(table.to_string(float_format=lambda x: f"{x:.3g}"))
    below = table.loc[[l for l in layers if l < args.adapter_layer]]
    at_after = table.loc[[l for l in layers if l >= args.adapter_layer]]
    ok_below = bool((below == 0).all().all())
    ok_after = bool((at_after > 0).any().all())
    print(f"\nlayers < {args.adapter_layer} identical across all conditions: {ok_below}")
    print(f"every condition differs from base at some layer >= {args.adapter_layer}: {ok_after}")
    print("adapter switching verified" if ok_below and ok_after else "CHECK FAILED -- investigate")
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--acts", default=None, help="activations.dat if not inside --dir")
    p.add_argument("--adapter_layer", type=int, default=21)
    p.add_argument("--out", default=None, help="optional CSV of the per-layer table")
    main(p.parse_args())
