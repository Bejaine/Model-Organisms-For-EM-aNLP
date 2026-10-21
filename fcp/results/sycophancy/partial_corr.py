"""Rank-based partial correlation of each probe's score with s_shift, controlling
for the level-0 log-odds (per fact), within each held-out level, averaged over
levels -- matching the brief's description for comparison against its reference
values (0.23, 0.37, 0.19 for baseline/ridge_ep/fcp against s_shift).
"""
import sys
import numpy as np
import pandas as pd

oof = pd.read_parquet(sys.argv[1])            # oof_scores_long.parquet from the s_shift run
sev = pd.read_parquet(sys.argv[2])             # severity.parquet (has logodds_correct per prompt/level)

l0 = sev[sev.level_idx == 0].set_index("prompt_id")["logodds_correct"]
oof["l0_logodds"] = oof["prompt"].map(l0)


def rank(a):
    return pd.Series(a).rank(method="average").to_numpy()


def residualize(y, x):
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3:
        return np.nan
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return np.nan if d == 0 else float((ra * rb).sum() / d)


for probe in sorted(oof.probe.unique()):
    d = oof[oof.probe == probe]
    per_level = []
    for level, g in d.groupby("level"):
        if len(g) < 4:
            continue
        r_score = rank(g["score"])
        r_s = rank(g["s"])
        r_l0 = rank(g["l0_logodds"])
        res_score = residualize(r_score, r_l0)
        res_s = residualize(r_s, r_l0)
        per_level.append(spearman(res_score, res_s))
    per_level = [v for v in per_level if v == v]
    mean_rho = float(np.mean(per_level)) if per_level else float("nan")
    print(f"{probe:10s} partial rho (score, s_shift | level-0 logodds), mean over levels: {mean_rho:+.3f}"
          f"   (per-level: {[f'{v:+.3f}' for v in per_level]})")
