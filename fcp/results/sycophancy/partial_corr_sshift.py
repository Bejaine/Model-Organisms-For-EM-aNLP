#!/usr/bin/env python
"""Partial Spearman correlation of each probe's OOF score with s_shift, controlling
for level-0 log-odds (per brief S5): rank-transform score, s and logodds_L0 within
each held-out level, then

    r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))

using Spearman correlations as the three pairwise terms -- the standard partial-
correlation formula, mathematically equivalent to residualising ranks via OLS and
correlating the residuals, but easier to verify by eye. Averaged over held-out
levels, matching the primary within-level metric's own aggregation.

Usage:
  python fcp/results/sycophancy/partial_corr_sshift.py \
      fcp/results/sycophancy/belief_sshift/oof_scores_long.parquet \
      fcp/activations/sycophancy_belief/severity.parquet
"""
import sys
import numpy as np
import pandas as pd


def spearman(x, y):
    x, y = pd.Series(x).rank(), pd.Series(y).rank()
    return np.corrcoef(x, y)[0, 1]


def partial(score, s, z):
    rxy, rxz, ryz = spearman(score, s), spearman(score, z), spearman(s, z)
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return (rxy - rxz * ryz) / denom if denom > 0 else np.nan


def main():
    oof_path = sys.argv[1] if len(sys.argv) > 1 else "fcp/results/sycophancy/belief_sshift/oof_scores_long.parquet"
    sev_path = sys.argv[2] if len(sys.argv) > 2 else "fcp/activations/sycophancy_belief/severity.parquet"

    oof = pd.read_parquet(oof_path)
    sev = pd.read_parquet(sev_path)
    l0 = sev[sev.level_idx == 0].set_index("prompt_id")["logodds_correct"]
    oof["logodds_l0"] = oof["prompt"].map(l0)

    probes = sorted(oof.probe.unique())
    rows = []
    for L, d in oof.groupby("level"):
        row = {"level": int(L), "n": len(d[d.probe == probes[0]])}
        for p in probes:
            dp = d[d.probe == p]
            row[p] = partial(dp["score"], dp["s"], dp["logodds_l0"])
        rows.append(row)
    per_level = pd.DataFrame(rows)
    print(per_level.to_string(index=False))
    means = {p: per_level[p].mean() for p in probes}
    print("\nmean partial correlation over held-out levels (controlling for L0 log-odds):")
    for p, v in means.items():
        print(f"  {p:10s} {v:+.3f}")
    print("\nreference values from the brief (s_shift): baseline 0.23, ridge_ep 0.37, fcp 0.19")

    import json
    json.dump({"per_level": rows, "mean": means}, open("fcp/results/sycophancy/partial_corr_sshift.json", "w"),
               indent=2, default=float)


if __name__ == "__main__":
    main()
