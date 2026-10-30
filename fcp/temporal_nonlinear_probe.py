"""fcp/temporal_nonlinear_probe.py -- Tier-1 follow-up #1: does a NONLINEAR probe
recover the within-trigger-year behavioural texture that the project's standard
linear probes (fcp_probes.py: baseline / ridge_ep / fcp) could not, on the exact
same dense-grid data (fcp/activations/temporal_dense_{2027,crossyear})?

This is a standalone script, not a change to fcp/fcp_probes.py -- that script is
owned by the sycophancy/DGX workstream per the project's shared-ownership
convention, so this reuses its data format and conventions (index.parquet,
severity.parquet, meta.json, the same leave-one-level-out structure) without
editing it.

Design, and why it's labelled EXPLORATORY rather than confirmatory:
  - Reuses the SAME layer fcp_probes.py already selected for each (experiment,
    condition) via its endpoint-separation criterion (read directly from that
    run's results.json), rather than re-deriving a layer selection here. This
    keeps the comparison apples-to-apples: same layer, same data, only the
    probe's functional form changes.
  - Same leave-one-level-out structure as fcp_probes.py's descriptive --layers
    sweep mode: for each interior level held out in turn, fit on every other
    level's activations -> f, predict on the held-out level's activations,
    compute within-level Spearman rho(prediction, severity) for that level,
    then average across levels. Uses ALL prompts for evaluation (no separate
    prompt cross-fitting / layer-selection holdout split) -- same simplification
    fcp_probes.py's own --layers sweep makes, and for the same reason: this is
    a descriptive check, not a new confirmatory headline result.
  - Probe: a small MLPRegressor (scikit-learn), fixed hyperparameters (not tuned
    per fold, to keep this simple, comparable across folds, and fast) -- one
    hidden layer, light L2 regularisation. This is a deliberately modest
    nonlinear model: enough to capture simple curvature/interaction effects a
    linear probe cannot, not an attempt to find the most powerful possible
    nonlinear fit (which would raise overfitting/multiple-comparisons concerns
    on data this size).
  - Permutation null: fit the SAME model on 200 shufflings of f (independently
    reshuffled within prompts at the training levels) and recompute the
    pooled/within-level statistic each time, exactly mirroring fcp_probes.py's
    own permutation-null logic, so the significance test is apples-to-apples
    with every linear result already reported.
  - Bootstrap CI: resample prompts with replacement, 2000 times, recompute the
    mean within-level rho -- same convention as every other result in this
    project.

Usage:
    python fcp/temporal_nonlinear_probe.py --dir fcp/activations/temporal_dense_2027 \
        --severity fcp/temporal_dense_2027/severity.parquet --s-col rate \
        --condition adapted --linear-results fcp/results/temporal/dense_2027_adapted/results.json \
        --out fcp/results/temporal/nonlinear_dense_2027_adapted
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

N_BOOT = 2000
SEED = 0


def load(args):
    idx = pd.read_parquet(args.index or os.path.join(args.dir, "index.parquet"))
    sev = pd.read_parquet(args.severity)
    meta_path = args.meta or os.path.join(args.dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    acts_path = args.acts or os.path.join(args.dir, "activations.dat")
    d_model = meta.get("d_model")
    if d_model is None:
        d_model = int(os.path.getsize(acts_path) / 4 / len(idx[idx.layer == idx.layer.min()]))
    acts = np.memmap(acts_path, dtype="float32", mode="r").reshape(-1, d_model)
    return idx, sev, meta, acts


def build_layer_table(idx, sev, acts, layer, condition, diff, s_col):
    if diff:
        a_idx = idx[(idx.layer == layer) & (idx.condition == diff[0])].set_index(["prompt_id", "level_idx"])
        b_idx = idx[(idx.layer == layer) & (idx.condition == diff[1])].set_index(["prompt_id", "level_idx"])
        common = a_idx.index.intersection(b_idx.index)
        a_idx, b_idx = a_idx.loc[common], b_idx.loc[common]
        X = acts[a_idx.row_idx.values] - acts[b_idx.row_idx.values]
        keys = a_idx.reset_index()[["prompt_id", "level_idx", "f"]]
        sev_cond = sev[sev.condition == diff[0]]
    else:
        sub = idx[(idx.layer == layer) & (idx.condition == condition)]
        X = acts[sub.row_idx.values]
        keys = sub[["prompt_id", "level_idx", "f"]].reset_index(drop=True)
        sev_cond = sev[sev.condition == condition]
    keys = keys.merge(sev_cond[["prompt_id", "level_idx", s_col]], on=["prompt_id", "level_idx"], how="left")
    return keys, X


def lolo_predictions(keys, X, held_out_levels, f_override=None, seed=SEED):
    """One pass of leave-one-level-out with the nonlinear probe. f_override lets
    the permutation null substitute shuffled f values without rebuilding keys."""
    f_all = f_override if f_override is not None else keys["f"].values
    levels = sorted(keys["level_idx"].unique())
    preds = np.full(len(keys), np.nan)
    for held in held_out_levels:
        train_mask = (keys["level_idx"] != held).values & ~pd.isna(keys["f"]).values
        test_mask = (keys["level_idx"] == held).values
        if train_mask.sum() < 10 or test_mask.sum() == 0:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_mask])
        model = MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0, max_iter=2000,
                              random_state=seed, early_stopping=True, n_iter_no_change=15)
        model.fit(Xtr, f_all[train_mask])
        preds[test_mask] = model.predict(Xte)
    return preds


def within_level_mean_rho(keys, preds, s_col, held_out_levels):
    rhos = []
    for held in held_out_levels:
        m = (keys["level_idx"] == held).values & ~np.isnan(preds)
        if m.sum() < 3:
            continue
        s = keys.loc[m, s_col].values
        p = preds[m]
        if np.std(s) == 0 or np.std(p) == 0:
            continue
        r, _ = spearmanr(p, s)
        if not np.isnan(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan"), rhos


def main(args):
    idx, sev, meta, acts = load(args)

    linear = json.load(open(args.linear_results))
    layer = args.layer_override if args.layer_override is not None else linear["layer"]
    held_out_levels = linear["held_out_levels"]
    print(f"Using layer {layer} (reused from {args.linear_results}), "
          f"held-out levels {held_out_levels}")

    keys, X = build_layer_table(idx, sev, acts, layer, args.condition, args.diff, args.s_col)
    keys = keys.dropna(subset=[args.s_col])
    valid = ~pd.isna(keys["f"]) | keys["level_idx"].isin(held_out_levels)
    keys = keys[valid].reset_index(drop=True)
    X = X[valid.values]

    print(f"{len(keys)} rows, {keys['level_idx'].nunique()} levels, "
          f"{keys['prompt_id'].nunique()} unique prompts")

    preds = lolo_predictions(keys, X, held_out_levels)
    observed_rho, per_level = within_level_mean_rho(keys, preds, args.s_col, held_out_levels)
    print(f"Nonlinear (MLP) mean within-level rho = {observed_rho:.3f}")
    print(f"  per-level: {dict(zip(held_out_levels, [round(r, 3) for r in per_level]))}")

    # bootstrap over prompts
    rng = np.random.default_rng(SEED)
    prompts = keys["prompt_id"].unique()
    boot_rhos = []
    for _ in range(N_BOOT):
        chosen = rng.choice(prompts, size=len(prompts), replace=True)
        mask = keys["prompt_id"].isin(chosen).values & ~np.isnan(preds)
        rhos = []
        for held in held_out_levels:
            m = mask & (keys["level_idx"] == held).values
            if m.sum() < 3:
                continue
            s, p = keys.loc[m, args.s_col].values, preds[m]
            if np.std(s) == 0 or np.std(p) == 0:
                continue
            r, _ = spearmanr(p, s)
            if not np.isnan(r):
                rhos.append(r)
        if rhos:
            boot_rhos.append(np.mean(rhos))
    ci = [float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))] if boot_rhos else [None, None]
    print(f"Bootstrap 95% CI: {ci}")

    # permutation null: shuffle f within training levels, refit, recompute
    null_rhos = []
    f_vals = keys["f"].values.copy()
    for i in range(args.n_perm):
        perm_f = f_vals.copy()
        train_mask = keys["level_idx"].isin([lv for lv in keys["level_idx"].unique() if lv not in held_out_levels])
        idxs = np.where(train_mask.values)[0]
        shuffled = rng.permutation(idxs)
        perm_f[idxs] = f_vals[shuffled]
        preds_null = lolo_predictions(keys, X, held_out_levels, f_override=perm_f, seed=SEED + i + 1)
        r_null, _ = within_level_mean_rho(keys, preds_null, args.s_col, held_out_levels)
        if not np.isnan(r_null):
            null_rhos.append(r_null)
        if (i + 1) % 10 == 0:
            print(f"  null {i+1}/{args.n_perm}", flush=True)
    null_rhos = np.array(null_rhos)
    p_value = float((np.sum(np.abs(null_rhos) >= abs(observed_rho)) + 1) / (len(null_rhos) + 1))
    print(f"Permutation-null p = {p_value:.4f} (null mean {null_rhos.mean():.3f}, "
          f"null 95th pct {np.percentile(np.abs(null_rhos), 95):.3f})")

    os.makedirs(args.out, exist_ok=True)
    report = {
        "probe": "MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0)",
        "layer": layer, "held_out_levels": held_out_levels,
        "n_rows": len(keys), "n_prompts": len(prompts),
        "observed_mean_within_level_rho": observed_rho,
        "per_level_rho": dict(zip([str(h) for h in held_out_levels], per_level)),
        "bootstrap_ci_95": ci,
        "permutation_null_p": p_value,
        "linear_comparison": {
            k: linear["probes"][k]["mean_within_level_rho"] for k in ["baseline", "ridge_ep", "fcp"]
        },
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}/results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--index")
    p.add_argument("--severity", required=True)
    p.add_argument("--meta")
    p.add_argument("--acts")
    p.add_argument("--s-col", default="rate")
    p.add_argument("--condition")
    p.add_argument("--diff", nargs=2)
    p.add_argument("--linear-results", required=True,
                    help="results.json from the matching fcp_probes.py run, to reuse its selected layer")
    p.add_argument("--out", required=True)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--layer-override", type=int, default=None,
                    help="use this layer instead of the one recorded in --linear-results "
                         "(e.g. a layer a sweep found more promising than auto-selection did)")
    args = p.parse_args()
    main(args)
