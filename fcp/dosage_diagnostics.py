"""fcp/dosage_diagnostics.py -- free diagnostics behind the failed / partial hypotheses (2026-10-28).

D1 ridge-penalty edges: did nested CV pick the largest lambda (FCP collapsing to the training mean)?
D2 trajectory geometry: PCA of the 20 level centroids (adapted, all prompts) at layers 21/23/27;
   share of variance on PC1/PC2 and the angle between early and late displacement (rotation).
D3 early warning: projection of each level's centroid on the endpoint DM direction, scaled 0 (base)
   to 1 (final), vs the step where EM first appears.
D4 cross-organism transfer: medical (v2) DM direction applied to the finance organism (same base
   model, same layer): within-level rho on post-onset levels vs finance's own DM; cosines between
   the two DM directions and the two adapters' B vectors.
D5 base-model susceptibility: does a prompt's BASE-model activation, projected on a leave-one-out
   DM direction, predict how misaligned that prompt becomes after fine-tuning (69 prompts)?
D6 variance decomposition of s within post-onset levels: share explained by question identity
   (intraclass correlation across levels).
Writes fcp/results/dosage/diagnostics/diagnostics.json and prints a report.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
J = ROOT.parent / "jarvis_outputs"
OUT = ROOT / "fcp/results/dosage/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
rep = {}


def load(acts_dir, prep_dir):
    m = json.load(open(Path(acts_dir) / "meta.json"))
    A = np.memmap(Path(acts_dir) / "activations.dat", dtype="float32", mode="r", shape=(m["n_rows"], m["d_model"]))
    idx = pd.read_parquet(Path(prep_dir) / "index.parquet")
    idx = idx[idx.condition == "adapted"]
    sev = pd.read_parquet(Path(prep_dir) / "severity.parquet")
    return A, idx, sev


def H(A, idx, layer, level=None, prompts=None):
    s = idx[idx.layer == layer]
    if level is not None:
        s = s[s.level_idx == level]
    if prompts is not None:
        s = s[s.prompt_id.isin(prompts)]
    s = s.sort_values(["level_idx", "prompt_id"])
    return np.asarray(A[s.row_idx.to_numpy()], dtype=np.float64), s


# ---------------- D1
d1 = {}
for p in sorted((ROOT / "fcp/results/dosage").glob("**/results.json")):
    r = json.load(open(p))
    if "lambda_edges" in r:
        d1[str(p.parent.relative_to(ROOT / "fcp/results/dosage"))] = r["lambda_edges"]
rep["D1_lambda_edges"] = d1

# ---------------- D2, D3 for medical v2 and finance
datasets = {
    "medical_v2": (J / "v2_2026-10-21/fcp/activations/dosage_v2", ROOT / "fcp/activations/dosage_v2_fcp"),
    "finance": (J / "fin_2026-10-28/activations/dosage_fin", ROOT / "fcp/activations/dosage_fin_fcp"),
}
loaded = {k: load(*v) for k, v in datasets.items()}
rep["D2_geometry"], rep["D3_early_warning"] = {}, {}
for name, (A, idx, sev) in loaded.items():
    levels = sorted(idx.level_idx.unique())
    steps = idx.groupby("level_idx").step.first().to_dict()
    em = sev.groupby("level_idx").s_em.mean()
    onset = next(steps[l] for l in levels if em[l] > 0.01)
    for L in (21, 23, 27):
        C = np.array([H(A, idx, L, l)[0].mean(0) for l in levels])
        D = C - C[0]
        _, sv, vt = np.linalg.svd(C - C.mean(0), full_matrices=False)
        var = sv ** 2 / (sv ** 2).sum()
        early = D[levels.index(next(l for l in levels if steps[l] >= 150))]
        late = D[-1] - D[levels.index(next(l for l in levels if steps[l] >= 300))]
        cosang = float(early @ late / (np.linalg.norm(early) * np.linalg.norm(late)))
        rep["D2_geometry"][f"{name}_L{L}"] = {"pc1": round(float(var[0]), 4), "pc2": round(float(var[1]), 4),
                                              "pc3": round(float(var[2]), 4), "cos_early_vs_late_displacement": round(cosang, 3)}
        w = D[-1] / np.linalg.norm(D[-1])
        proj = D @ w / (D[-1] @ w)
        rep["D3_early_warning"][f"{name}_L{L}"] = {
            "em_onset_step": int(onset),
            "fraction_of_final_projection_by_step": {int(steps[l]): round(float(proj[i]), 3) for i, l in enumerate(levels)},
            "em_rate_by_step": {int(steps[l]): round(float(em[l]), 3) for l in levels}}

# ---------------- D4 transfer medical -> finance
(Am, im, sm), (Af, ifn, sf) = loaded["medical_v2"], loaded["finance"]
rep["D4_transfer"] = {}
for L in (21, 23):
    def dm(A, idx):
        lv = sorted(idx.level_idx.unique())
        return H(A, idx, L, lv[-1])[0].mean(0) - H(A, idx, L, lv[0])[0].mean(0)
    wm, wf = dm(Am, im), dm(Af, ifn)
    wm_u, wf_u = wm / np.linalg.norm(wm), wf / np.linalg.norm(wf)
    out = {"cos_medical_vs_finance_DM": round(float(wm_u @ wf_u), 4)}
    for wname, w in (("medical_DM", wm_u), ("finance_DM", wf_u)):
        rhos = []
        for l in sorted(ifn.level_idx.unique()):
            if sf[sf.level_idx == l].s_em.mean() <= 0.01:
                continue
            h, s = H(Af, ifn, L, l)
            y = sf.set_index(["prompt_id", "level_idx"]).loc[list(zip(s.prompt_id, s.level_idx)), "s"].to_numpy()
            rhos.append(spearmanr(h @ w, y).statistic)
        out[f"finance_within_level_rho_using_{wname}"] = round(float(np.nanmean(rhos)), 3)
    rep["D4_transfer"][f"L{L}"] = out
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
bvec = {}
for nm, repo, step in (("medical", "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train", 792),
                       ("finance", "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train", 675)):
    w = load_file(hf_hub_download(repo, f"checkpoints/checkpoint-{step}/adapter_model.safetensors"))
    b = next(v for k, v in w.items() if "lora_B" in k).double().numpy().ravel()
    a = next(v for k, v in w.items() if "lora_A" in k).double().numpy().ravel()
    bvec[nm] = (b / np.linalg.norm(b), a / np.linalg.norm(a))
rep["D4_transfer"]["cos_adapter_B_medical_vs_finance"] = round(float(bvec["medical"][0] @ bvec["finance"][0]), 4)
rep["D4_transfer"]["cos_adapter_A_medical_vs_finance"] = round(float(bvec["medical"][1] @ bvec["finance"][1]), 4)

# ---------------- D5 base susceptibility (69 prompts)
A69, i69, s69 = load(ROOT / "fcp/activations/dosage_v2v3", ROOT / "fcp/activations/dosage_v2v3_fcp")
rep["D5_base_susceptibility"] = {}
lv = sorted(i69.level_idx.unique())
post = [l for l in lv if i69[i69.level_idx == l].step.iloc[0] >= 300]
target = s69[s69.level_idx.isin(post)].groupby("prompt_id").s.mean()
for L in (21, 23, 27):
    h0, s0 = H(A69, i69, L, lv[0])
    h1, s1 = H(A69, i69, L, lv[-1])
    pids = s0.prompt_id.to_numpy()
    scores = []
    for k in range(len(pids)):
        keep = np.arange(len(pids)) != k
        w = h1[keep].mean(0) - h0[keep].mean(0)
        scores.append(h0[k] @ w / np.linalg.norm(w))
    r = spearmanr(scores, target.loc[pids].to_numpy())
    base_s = s69[s69.level_idx == lv[0]].set_index("prompt_id").s.loc[pids]
    rep["D5_base_susceptibility"][f"L{L}"] = {"rho_base_projection_vs_post_onset_severity": round(float(r.statistic), 3),
                                             "p": round(float(r.pvalue), 4), "n_prompts": len(pids),
                                             "rho_base_severity_vs_post_onset_severity": round(float(spearmanr(base_s, target.loc[pids]).statistic), 3)}

# ---------------- D6 variance decomposition
rep["D6_question_identity"] = {}
for name, s in (("medical_v2", sm), ("finance", sf), ("v2v3_69", s69)):
    st = s.merge(s.groupby("level_idx").s_em.mean().rename("lev_em"), left_on="level_idx", right_index=True)
    p = st[st.lev_em > 0.01]
    tot = ((p.s - p.groupby("level_idx").s.transform("mean")) ** 2).sum()
    q = p.groupby("prompt_id").s.transform("mean") - p.groupby("level_idx").s.transform("mean")
    rep["D6_question_identity"][name] = round(float((q ** 2).sum() / tot), 3)

json.dump(rep, open(OUT / "diagnostics.json", "w"), indent=1)
print(json.dumps(rep, indent=1)[:6000])
