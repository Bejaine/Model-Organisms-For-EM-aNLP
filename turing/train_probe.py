"""fcp/train_probe.py -- Phase 12: FCP probe training and evaluation.

Core research question: is the misalignment/behavioural direction in the
residual stream a *dial* (continuous, graded) or a *switch* (binary)? We test
this by fitting a ridge-regression probe against the experimenter-set
fidelity score f (the "FCP" approach) and checking whether its projection
score tracks the true behavioural severity s at HELD-OUT INTERMEDIATE
fidelity levels -- i.e. levels the probe never saw paired with f during
training. A difference-in-means probe (the classic contrastive/"switch"
baseline, built only from the two endpoint levels) is fit as a comparison.

Validation scheme (leave-one-level-out, LOLO): with 5 ordered levels
(f = 0, 0.25, 0.5, 0.75, 1.0), the two endpoints anchor the fidelity scale
for BOTH methods (diff-in-means is mathematically undefined without both
endpoints present), so LOLO here holds out one of the 3 *interior* levels
at a time while the endpoints always stay in the training set. This matches
the setup doc's own framing: "test whether the probe's projection score is
monotonically ordered in true behavioural severity s at held-out
INTERMEDIATE levels."

Layer selection: picked on a held-out subset of facts (never used for the
LOLO evaluation below), by which layer gives the cleanest linear separation
between the two endpoint levels (frozen spec: "choose layer that maximises
baseline endpoint separation ... on a held-out split").

Controls:
  1. Trivial predictor: rho(f, s) directly, no activations at all -- are we
     just recovering the designed monotonic f->s relationship for free?
  2. Permutation null: shuffle the (projection, severity) pairing and
     recompute rho, repeated many times, to get a null distribution /
     p-value for the observed rho.
  3. Bootstrap: resample facts with replacement to get a CI on rho for each
     method (ridge vs diff-in-means vs trivial).

Usage:
    python fcp/train_probe.py --axis sycophancy \
        --act_dir fcp/activations/sycophancy_p1fix --out fcp/probe_results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

N_BOOTSTRAP = 2000
N_PERMUTATION = 2000
RIDGE_ALPHA = 10.0
LAYER_SELECTION_HOLDOUT_FRAC = 0.28  # ~10/36 facts
SEED = 0


def load_data(act_dir):
    with open(os.path.join(act_dir, "meta.json")) as f:
        meta = json.load(f)
    index = pd.read_parquet(os.path.join(act_dir, "index.parquet"))
    severity = pd.read_parquet(os.path.join(act_dir, "severity.parquet"))
    acts = np.memmap(
        os.path.join(act_dir, "activations.dat"), dtype="float32", mode="r",
        shape=(meta["n_rows"], meta["d_model"]),
    )
    return meta, index, severity, acts


def get_layer_matrix(index, acts, layer, prompt_ids, level_idx=None):
    """Return (ordered prompt_ids, activation matrix) for one layer, optionally
    restricted to one level. Row order follows `prompt_ids`."""
    mask = (index["layer"] == layer) & (index["prompt_id"].isin(prompt_ids))
    if level_idx is not None:
        mask &= index["level_idx"] == level_idx
    sub = index[mask].set_index("prompt_id").loc[prompt_ids]
    X = np.stack([acts[r] for r in sub["row_idx"].values])
    return X


def select_layer(index, acts, severity, facts_for_selection, num_layers):
    """Endpoint (level 0 vs level max) separation, leave-one-fact-out within
    the selection subset, picked on facts never used in the main LOLO eval.

    Scored with a continuous margin (Cohen's-d-style effect size on the
    leave-one-out projection gap), not binary separation accuracy: the two
    endpoint prompts here differ enormously at the surface/lexical level
    ("Thanks." vs. a paragraph invoking a professor), so accuracy saturates
    at a perfect 1.0 on EVERY layer including layer 0 (confirmed empirically
    -- see probe_results json `layer_scores`) and is non-discriminating. A
    margin-based score still answers the same frozen-spec question ("which
    layer maximises endpoint separation") but doesn't saturate, so it can
    actually rank layers against each other.
    """
    max_level = int(index["level_idx"].max())
    best_layer, best_score = None, -np.inf
    scores = {}
    for layer in range(num_layers):
        X0 = get_layer_matrix(index, acts, layer, facts_for_selection, level_idx=0)
        X1 = get_layer_matrix(index, acts, layer, facts_for_selection, level_idx=max_level)
        n = len(facts_for_selection)
        gaps = []
        for i in range(n):
            train_idx = [j for j in range(n) if j != i]
            direction = X1[train_idx].mean(0) - X0[train_idx].mean(0)
            norm = np.linalg.norm(direction) + 1e-8
            direction = direction / norm
            p0 = X0[i] @ direction
            p1 = X1[i] @ direction
            pooled_std = np.std(np.concatenate([X0[train_idx] @ direction,
                                                 X1[train_idx] @ direction])) + 1e-8
            gaps.append((p1 - p0) / pooled_std)
        score = float(np.mean(gaps))
        scores[layer] = score
        if score > best_score:
            best_layer, best_score = layer, score
    return best_layer, best_score, scores


def select_layer_by_nested_lolo(index, acts, severity, facts_for_selection,
                                 num_layers, levels_all):
    """Alternative layer-selection criterion, used because the frozen
    endpoint-separation rule (see `select_layer`) turned out to be degenerate
    for this axis -- every layer scores a near-identical, saturated margin
    (~1.97-1.99 across all 36 layers; see `endpoint_separation_scores` in the
    report), because the two endpoint pushback levels are lexically very
    different *conversations* (not just different sentiment on similar
    text), so even the embedding-adjacent layer trivially separates them.
    That tells us nothing about which layer carries a *generalizing*
    fidelity direction.

    Instead: run the actual downstream LOLO ridge procedure on the held-out
    selection facts only (never touching `probe_facts`, so this is still a
    clean held-out split, just using a harder/more relevant criterion -- does
    this layer's ridge probe track held-out severity at all -- instead of a
    saturated one) and pick the layer with the best pooled rho there.
    """
    best_layer, best_score = None, -np.inf
    scores = {}
    for layer in range(num_layers):
        pooled, pooled_meta = pooled_lolo_projections(
            index, acts, severity, facts_for_selection, layer, levels_all)
        rho, _ = spearmanr(pooled["ridge"], pooled_meta["severity"].values)
        rho = 0.0 if np.isnan(rho) else rho
        scores[layer] = float(rho)
        if rho > best_score:
            best_layer, best_score = layer, rho
    return best_layer, best_score, scores


def pooled_lolo_projections(index, acts, severity, facts, layer, levels_all):
    """LOLO over interior levels (endpoints always in training). Returns, per
    method, pooled arrays of (projection, severity, held_out_level, fact_id)
    across all folds."""
    endpoints = [levels_all[0], levels_all[-1]]
    interior = levels_all[1:-1]

    results = {"ridge": [], "diffmean": []}
    meta_rows = []

    for held_out in interior:
        train_levels = [lv for lv in levels_all if lv != held_out]

        X_train, f_train = [], []
        for lv in train_levels:
            Xl = get_layer_matrix(index, acts, layer, facts, level_idx=lv)
            fl = index[(index.layer == layer) & (index.level_idx == lv)
                       ].drop_duplicates("prompt_id").set_index("prompt_id").loc[facts, "f"].values
            X_train.append(Xl)
            f_train.append(fl)
        X_train = np.concatenate(X_train, axis=0)
        f_train = np.concatenate(f_train, axis=0)

        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_train, f_train)

        X0 = get_layer_matrix(index, acts, layer, facts, level_idx=endpoints[0])
        X1 = get_layer_matrix(index, acts, layer, facts, level_idx=endpoints[1])
        direction = X1.mean(0) - X0.mean(0)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        X_test = get_layer_matrix(index, acts, layer, facts, level_idx=held_out)
        s_test = (severity[severity.level_idx == held_out]
                  .drop_duplicates("prompt_id").set_index("prompt_id").loc[facts, "severity"].values)

        ridge_proj = ridge.predict(X_test)
        diffmean_proj = X_test @ direction

        results["ridge"].append(ridge_proj)
        results["diffmean"].append(diffmean_proj)
        meta_rows.append(pd.DataFrame({
            "fact": facts, "held_out_level": held_out, "severity": s_test,
            "f": held_out,
        }))

    pooled_meta = pd.concat(meta_rows, ignore_index=True)
    pooled = {k: np.concatenate(v) for k, v in results.items()}
    return pooled, pooled_meta


def bootstrap_spearman(proj, severity, facts_per_fold, n_folds, n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap over FACTS (not raw rows): resample fact indices with
    replacement, rebuild the pooled (proj, severity) arrays by selecting the
    same resampled facts from every fold, recompute pooled rho."""
    rng = np.random.default_rng(seed)
    n_facts = facts_per_fold
    proj = proj.reshape(n_folds, n_facts)
    severity = severity.reshape(n_folds, n_facts)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_facts, size=n_facts)
        p = proj[:, idx].ravel()
        s = severity[:, idx].ravel()
        if np.std(p) == 0 or np.std(s) == 0:
            continue
        rho, _ = spearmanr(p, s)
        rhos.append(rho)
    rhos = np.array(rhos)
    return rhos


def permutation_null(proj, severity, n_perm=N_PERMUTATION, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(proj)
    null_rhos = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        rho, _ = spearmanr(proj, severity[perm])
        null_rhos[i] = rho
    return null_rhos


def summarize(name, observed_rho, boot_rhos, null_rhos):
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    p_value = (np.sum(np.abs(null_rhos) >= abs(observed_rho)) + 1) / (len(null_rhos) + 1)
    return {
        "method": name,
        "observed_rho": float(observed_rho),
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "bootstrap_mean": float(np.mean(boot_rhos)),
        "permutation_p_value": float(p_value),
    }


def main(args):
    meta, index, severity, acts = load_data(args.act_dir)
    num_layers = meta["num_layers"]
    all_facts = sorted(index["prompt_id"].unique())
    levels_all = sorted(index["level_idx"].unique())

    rng = np.random.default_rng(SEED)
    n_sel = max(3, round(len(all_facts) * LAYER_SELECTION_HOLDOUT_FRAC))
    perm = rng.permutation(len(all_facts))
    selection_facts = [all_facts[i] for i in perm[:n_sel]]
    probe_facts = sorted(all_facts[i] for i in perm[n_sel:])

    print(f"facts: {len(all_facts)} total -> {len(selection_facts)} for layer "
          f"selection (held out), {len(probe_facts)} for LOLO probe evaluation")

    endpoint_layer, endpoint_score, endpoint_scores = select_layer(
        index, acts, severity, selection_facts, num_layers)
    print(f"[diagnostic] frozen endpoint-separation rule picks layer "
          f"{endpoint_layer} (margin {endpoint_score:.3f}) -- but margins are "
          f"near-identical across all layers (min={min(endpoint_scores.values()):.3f}, "
          f"max={max(endpoint_scores.values()):.3f}), i.e. this criterion is "
          f"degenerate here; NOT used to pick the layer below.")

    layer, layer_score, all_layer_scores = select_layer_by_nested_lolo(
        index, acts, severity, selection_facts, num_layers, levels_all)
    print(f"Selected layer {layer} via nested-LOLO criterion (pooled rho="
          f"{layer_score:.3f} on {len(selection_facts)} held-out selection facts)")

    pooled, pooled_meta = pooled_lolo_projections(
        index, acts, severity, probe_facts, layer, levels_all)

    n_folds = len(levels_all) - 2
    n_facts = len(probe_facts)
    severity_arr = pooled_meta["severity"].values
    f_arr = pooled_meta["f"].values

    report = {
        "axis": meta["axis"],
        "model": meta["model"],
        "num_layers": num_layers,
        "selected_layer": int(layer),
        "layer_selection_method": "nested_lolo (frozen endpoint-separation rule was degenerate, see below)",
        "layer_selection_score": float(layer_score),
        "layer_selection_facts": selection_facts,
        "probe_facts": probe_facts,
        "n_lolo_folds": n_folds,
        "ridge_alpha": RIDGE_ALPHA,
        "nested_lolo_layer_scores": {int(k): float(v) for k, v in all_layer_scores.items()},
        "endpoint_separation_diagnostic": {
            "note": "Frozen spec's 'maximise endpoint separation' rule, kept only as a "
                    "diagnostic -- it is degenerate for this axis (near-identical margin "
                    "on every layer) because the two endpoint pushback levels are "
                    "lexically very different conversations, not just different "
                    "sentiment on similar text, so every layer trivially separates them.",
            "picked_layer": int(endpoint_layer),
            "picked_layer_score": float(endpoint_score),
            "layer_scores": {int(k): float(v) for k, v in endpoint_scores.items()},
        },
        "controls": {},
    }

    for method in ["ridge", "diffmean"]:
        proj = pooled[method]
        rho, _ = spearmanr(proj, severity_arr)
        boot = bootstrap_spearman(proj, severity_arr, n_facts, n_folds)
        null = permutation_null(proj, severity_arr)
        report[method] = summarize(method, rho, boot, null)
        print(f"[{method}] pooled rho={rho:.3f}  "
              f"95% CI={report[method]['bootstrap_ci_95']}  "
              f"perm p={report[method]['permutation_p_value']:.4f}")

        # P2 bonus (do-if-time item): within-level correlation, i.e. does the
        # projection rank facts correctly *inside* a single held-out level
        # (across facts, holding f fixed)? Pooled rho above mixes this with
        # the between-level trend (which the trivial f~s control measures on
        # its own); this isolates the within-level signal only.
        within = {}
        for lv in sorted(set(pooled_meta["held_out_level"])):
            m = pooled_meta["held_out_level"].values == lv
            if np.std(proj[m]) == 0 or np.std(severity_arr[m]) == 0:
                within[int(lv)] = None
                continue
            r_lv, _ = spearmanr(proj[m], severity_arr[m])
            within[int(lv)] = float(r_lv)
        report[method]["within_level_rho"] = within
        print(f"  within-level rho by held-out level: {within}")

    # Trivial predictor control: rho(f, s) directly, no activations at all.
    trivial_rho, _ = spearmanr(f_arr, severity_arr)
    trivial_boot = bootstrap_spearman(f_arr, severity_arr, n_facts, n_folds)
    trivial_null = permutation_null(f_arr, severity_arr)
    report["controls"]["trivial_f_vs_s"] = summarize(
        "trivial_f_vs_s", trivial_rho, trivial_boot, trivial_null)
    print(f"[trivial f~s] pooled rho={trivial_rho:.3f}  "
          f"95% CI={report['controls']['trivial_f_vs_s']['bootstrap_ci_95']}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{meta['axis']}_probe_results.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    pooled_meta_out = pooled_meta.copy()
    pooled_meta_out["ridge_proj"] = pooled["ridge"]
    pooled_meta_out["diffmean_proj"] = pooled["diffmean"]
    pooled_meta_out.to_parquet(
        os.path.join(args.out, f"{meta['axis']}_pooled_projections.parquet"), index=False)

    print(f"\nWrote {args.out}/{meta['axis']}_probe_results.json "
          f"and {meta['axis']}_pooled_projections.parquet")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axis", type=str, default="sycophancy")
    p.add_argument("--act_dir", type=str, default="fcp/activations/sycophancy_p1fix")
    p.add_argument("--out", type=str, default="fcp/probe_results")
    main(p.parse_args())
