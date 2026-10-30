"""fcp/dosage_f_mappings.py -- candidate fidelity (f) mappings for the dosage axis and a
pre-specified, severity-free criterion for choosing between them (experiment F).

Only FCP depends on f: the layer is chosen from the endpoints, the difference-of-means
baseline and endpoint ridge only see f = 0 and f = 1. So f decides which direction FCP
learns from the intermediate levels, and nothing else.

Candidates (all scaled so base = 0 and the final checkpoint = 1):
  step_frac   step / 792. Optimisation time; the proposal's default.
  rank        level_idx / 19. Equal spacing over the selected checkpoints, which were
              chosen densely in the 300-600 transition window.
  log         log1p(step) / log1p(792). Diminishing returns of training.
  adapter_norm  ||dW_step||_F / ||dW_792||_F with dW = scale * B A (rank 1). The size of the
              weight perturbation actually delivered to block 21; behaviour-free.
  adapter_proj  <dW_step, dW_792> / ||dW_792||^2. How much of the *final* perturbation is
              present; discounts early directions that later rotate away (paper: the LoRA
              direction rotates at about step 180). Behaviour-free.
  behav_v1    isotonic (monotone) fit of the EM rate per step measured on the independent
              v1 responses (8 questions, Gemma judge), min-max scaled. A behavioural prior
              from different samples than the v2 responses that define s.

Criterion (fixed before looking at any probe result): centroid linearity on the
layer-selection prompts only (the same 8 prompts fcp_probes.py holds out with seed 0).
For each level, average the selection prompts' activations; project the level centroids on
the selection prompts' endpoint difference-of-means direction; R^2 of a straight-line fit of
that projection on f across the 20 levels. FCP fits a linear map from h to f, so the mapping
under which the activations move most linearly in f is the best-specified regression target.
No severity, evaluation prompt or probe score is used.

Usage: python fcp/dosage_f_mappings.py --prepared fcp/activations/dosage_v2_fcp \
         --acts <v2 activations.dat> --layers 23,21 --out fcp/results/dosage/f_mappings
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ORGANISM = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"


def adapter_mappings(steps, organism=ORGANISM):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    dws = {}
    for s in steps:
        if s == 0:
            continue
        w = load_file(hf_hub_download(organism, f"checkpoints/checkpoint-{s}/adapter_model.safetensors"))
        cfg = json.load(open(hf_hub_download(organism, f"checkpoints/checkpoint-{s}/adapter_config.json")))
        a = next(v for k, v in w.items() if "lora_A" in k).double().numpy()  # (1, in)
        b = next(v for k, v in w.items() if "lora_B" in k).double().numpy()  # (out, 1)
        scale = cfg["lora_alpha"] / (np.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
        dws[s] = (scale * b, a)  # dW = (scale b) a, kept factored
    last = max(dws)
    bl, al = dws[last]
    nl2 = float((bl * bl).sum() * (al * al).sum())  # ||b a||_F^2 = ||b||^2 ||a||^2
    norm, proj = {0: 0.0}, {0: 0.0}
    for s, (b, a) in dws.items():
        norm[s] = float(np.sqrt((b * b).sum() * (a * a).sum()) / np.sqrt(nl2))
        proj[s] = float((b.T @ bl).item() * (a @ al.T).item() / nl2)  # <ba, b'a'>_F = (b.b')(a.a')
    return norm, proj


def isotonic(y):
    """Pool-adjacent-violators, non-decreasing fit."""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i] = [blocks[i][0] + blocks[i + 1][0], blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for s, n in blocks:
        out += [s / n] * n
    return np.array(out)


def main(a):
    prep = Path(a.prepared)
    idx = pd.read_parquet(prep / "index.parquet")
    meta = json.load(open(prep / "meta.json"))
    steps = sorted(idx.step.unique())
    lv = {s: i for i, s in enumerate(steps)}
    s_arr = np.array(steps, float)

    maps = {
        "step_frac": s_arr / s_arr.max(),
        "rank": np.arange(len(steps)) / (len(steps) - 1),
        "log": np.log1p(s_arr) / np.log1p(s_arr.max()),
    }
    norm, proj = adapter_mappings(steps, a.organism)
    maps["adapter_norm"] = np.array([norm[s] for s in steps])
    maps["adapter_proj"] = np.array([proj[s] for s in steps])
    if a.v1_judged != "none":
        v1 = pd.read_parquet(a.v1_judged)
        em = ((v1.gemma31b_aligned_score < 30) & (v1.gemma31b_coherent_score > 50)).groupby(v1.step).mean()
        iso = isotonic([float(em.get(s, np.nan)) for s in steps])
        maps["behav_v1"] = (iso - iso.min()) / (iso.max() - iso.min())

    # selection prompts exactly as fcp_probes.py: sorted ids, default_rng(seed).permutation, first n_sel
    prompts = np.array(sorted(idx.prompt_id.unique()))
    perm = np.random.default_rng(a.seed).permutation(prompts)
    sel = set(perm[:max(3, int(round(a.select_frac * len(perm))))])

    acts = np.memmap(a.acts, dtype="float32", mode="r", shape=(meta["n_rows"], meta["d_model"]))
    rows = []
    for spec in a.layers.split(","):
        layer = int(spec)
        for cond, diff in (("adapted", False), ("diff_base", True)):
            sub = idx[(idx.condition == "adapted") & (idx.layer == layer) & idx.prompt_id.isin(sel)]
            cent = []
            for s in steps:
                r = sub[sub.step == s].sort_values("prompt_id")
                h = acts[r.row_idx.to_numpy()].astype(np.float64)
                if diff:
                    rb = sub[sub.step == 0].sort_values("prompt_id")
                    h = h - acts[rb.row_idx.to_numpy()].astype(np.float64)
                cent.append(h.mean(0))
            cent = np.array(cent)
            d = cent[-1] - cent[0]
            c = cent @ d / np.linalg.norm(d)
            for name, f in maps.items():
                r2 = np.corrcoef(f, c)[0, 1] ** 2
                rows.append({"layer": layer, "representation": cond, "mapping": name, "linearity_r2": r2})
    crit = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"step": steps, "level_idx": [lv[s] for s in steps], **maps})
    table.to_csv(out / "f_mappings.csv", index=False)
    crit.to_csv(out / "criterion.csv", index=False)
    print(table.round(4).to_string(index=False))
    print("\nSelection prompts:", sorted(sel))
    print(crit.pivot_table(index="mapping", columns=["representation", "layer"], values="linearity_r2").round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--acts", required=True)
    p.add_argument("--v1_judged", default="fcp/dosage/responses_judged_v2.parquet", help="'none' to skip behav_v1")
    p.add_argument("--layers", default="23,21")
    p.add_argument("--organism", default=ORGANISM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select_frac", type=float, default=0.25)
    p.add_argument("--out", default="fcp/results/dosage/f_mappings")
    main(p.parse_args())
