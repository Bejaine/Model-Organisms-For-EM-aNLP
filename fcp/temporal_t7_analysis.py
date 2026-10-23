"""fcp/temporal_t7_analysis.py -- Step T7: descriptive, exploratory analyses
at the level of individual dates. NOT pre-registered -- labelled as
exploratory throughout, with bootstrap CIs, per the brief's explicit
instruction for this step.

1. Does propensity rise as the date approaches the trigger? Spearman(day of
   year, s) within 2026, 2027, 2028 (adapted condition, pooled n=20 rate).
2. Paired effect per date: s_adapted - s_clean, by year.
3. Probe score (out-of-fold, adapted) against day of year:
   - 2026: the one interior level actually held out by LOLO -- genuine
     out-of-fold scores from fcp_probes.py's oof_scores_long.parquet.
   - 2028-2032: Q3 extrapolation scores (probes fit on interior levels,
     scored on these never-trained-on years) -- also genuinely out-of-fold.
   - 2025 and 2027 are skipped here: both are LOLO endpoints, always in the
     training set, so there is no fair held-out score for them without
     re-running a different (and non-standard) cross-validation split.

Usage:
    python fcp/temporal_t7_analysis.py --out fcp/results/temporal/t7_descriptive.json
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

N_BOOT = 2000


def day_of_year(date_str):
    return datetime.strptime(date_str, "%B %d, %Y").timetuple().tm_yday


def bootstrap_ci(values_a, values_b=None, stat=np.mean, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(values_a)
    a = np.asarray(values_a)
    b = np.asarray(values_b) if values_b is not None else None
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if b is None:
            boots.append(stat(a[idx]))
        else:
            boots.append(stat(a[idx] - b[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_spearman_ci(x, y, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    x, y = np.asarray(x), np.asarray(y)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        r, _ = spearmanr(x[idx], y[idx])
        rhos.append(r)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main(args):
    sev = pd.read_parquet(args.severity)
    sev["doy"] = sev["date_str"].apply(day_of_year)

    report = {"1_day_of_year_vs_s": {}, "2_paired_effect_by_year": {}, "3_probe_score_vs_day_of_year": {}}

    # 1. Spearman(day of year, s), adapted, within each of 2026/2027/2028
    for year in [2026, 2027, 2028]:
        sub = sev[(sev.condition == "adapted") & (sev.year == year)]
        rho, p = spearmanr(sub["doy"], sub["s"])
        ci = bootstrap_spearman_ci(sub["doy"].values, sub["s"].values)
        report["1_day_of_year_vs_s"][str(year)] = {
            "n": len(sub), "rho": None if np.isnan(rho) else float(rho), "p": float(p), "ci95": ci,
        }

    # 2. paired effect per date: s_adapted - s_clean, by year
    piv = sev.pivot_table(index=["year", "prompt_id"], columns="condition", values="s").reset_index()
    for year in sorted(piv["year"].unique()):
        sub = piv[piv.year == year].dropna(subset=["adapted", "clean"])
        diff = sub["adapted"] - sub["clean"]
        ci = bootstrap_ci(sub["adapted"].values, sub["clean"].values)
        report["2_paired_effect_by_year"][str(year)] = {
            "n": len(sub), "mean_diff": float(diff.mean()), "ci95": ci,
        }

    # 3. probe score vs day of year (out-of-fold only: 2026 from LOLO, 2028-2032 from Q3)
    oof = pd.read_parquet(args.oof_scores)
    oof_adapted = oof[oof.condition == "adapted"]
    # oof's "prompt" column holds prompt_id strings (e.g. "date00"), same as sev's,
    # but sev has one row per (prompt_id, level_idx, condition) -- date_str is the
    # same across the two interior levels for a given prompt_id only if we pick the
    # matching level; use the level in oof itself (level=1, i.e. year 2026) to look up.
    date_map = (sev[sev.level_idx == 1][["prompt_id", "date_str"]]
                .drop_duplicates().set_index("prompt_id")["date_str"])
    for probe in ["baseline", "fcp"]:
        p_scores = []
        p_doys = []
        sub = oof_adapted[oof_adapted.probe == probe]
        for _, r in sub.iterrows():
            p_doys.append(day_of_year(date_map[r["prompt"]]))
            p_scores.append(r["score"])
        rho, p = spearmanr(p_doys, p_scores) if len(p_scores) > 2 else (float("nan"), float("nan"))
        report["3_probe_score_vs_day_of_year"].setdefault(probe, {})["2026_lolo_heldout"] = {
            "n": len(p_scores), "rho": None if np.isnan(rho) else float(rho), "p": float(p),
        }

    extrap = pd.read_csv(args.extrapolation_scores)
    for probe in ["baseline", "fcp"]:
        for year in sorted(extrap["year"].unique()):
            sub = extrap[extrap.year == year]
            doys = sub["date_str"].apply(day_of_year)
            rho, p = spearmanr(doys, sub[probe])
            report["3_probe_score_vs_day_of_year"].setdefault(probe, {})[f"{year}_q3_extrapolation"] = {
                "n": len(sub), "rho": None if np.isnan(rho) else float(rho), "p": float(p),
            }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--severity", default="fcp/activations/temporal_fcp_pooled/severity.parquet")
    p.add_argument("--oof-scores", default="fcp/results/temporal/pooled_adapted/oof_scores_long.parquet")
    p.add_argument("--extrapolation-scores", default="fcp/results/temporal/pooled_adapted/extrapolation_scores.csv")
    p.add_argument("--out", default="fcp/results/temporal/t7_descriptive.json")
    main(p.parse_args())
