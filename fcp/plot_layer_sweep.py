#!/usr/bin/env python
"""fcp/plot_layer_sweep.py -- plot a descriptive layer sweep pair (s_belief, s_shift) in the same
style as fcp/results/sycophancy/tqa_layer_sweep.png, marking the officially selected layer.

Usage:
  python fcp/plot_layer_sweep.py --tag sciq --title "SciQ facts" \
      --out fcp/results/sycophancy/sciq_layer_sweep.png
Reads fcp/results/sycophancy/<tag>_layersweep_{sbelief,sshift}/layer_sweep.csv and the selected
layer from fcp/results/sycophancy/<tag>_sbelief/results.json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = "fcp/results/sycophancy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--n-known", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    layer = json.load(open(os.path.join(RES, f"{a.tag}_sbelief", "results.json")))["layer"]
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.2), sharey=True)
    for ax, s in zip(axes, ["sbelief", "sshift"]):
        df = pd.read_csv(os.path.join(RES, f"{a.tag}_layersweep_{s}", "layer_sweep.csv"))
        for p in ["baseline", "ridge_ep", "fcp"]:
            ax.plot(df.layer, df[f"within_rho_{p}"], marker="o", ms=4, label=p)
        ax.axvline(layer, color="red", ls="--", alpha=0.6, label=f"selected layer ({layer})")
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(f"{a.title} (n={a.n_known} known), s_{s[1:]}", fontsize=14)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("within-level rho")
    axes[0].legend()
    fig.suptitle("Descriptive, selection-free layer sweep -- NOT used to choose the reported layer", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
