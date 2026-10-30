"""fcp/dosage_synthetic_control.py -- T2.4b: synthetic positive control for FCP vs DM.

Real ingredients (v2, layer 21): each prompt's base-model activation (prompt identity) and real
within-level noise (residuals of the adapted activations after removing level and prompt means).
Planted structure, f_l = l / 19 for 20 levels, susceptibility c_p ~ U(0.5, 1.5):
  (a) one_dim:      h = base_p + g(f) c_p v1 + noise;                 s = g(f) c_p
  (b) two_dim:      h = base_p + g(f) c_p v1 + b(f) c'_p v2 + noise;   s = g(f) c_p + b(f) c'_p,
                    with b(f) = sin(pi f) (only intermediate levels; invisible to the endpoints)
  (c) susceptible:  as (a), but base_p also carries k * c_p * u along a prompt direction u
g(f) = sigmoid(10 (f - 0.4)) (switch-like). Probes as in fcp_probes: DM (endpoints), endpoint
ridge, FCP (ridge on all training levels), each raw and PAIRED (h - base_p). Evaluation: leave
one intermediate level out, 5-fold over prompts, within-level Spearman with s, averaged over
held-out levels; 10 repetitions. Writes fcp/results/dosage/subhyp/T2.4b_synthetic.json.

v2 (2026-10-29): amplitudes are now in units of the REAL signal -- the norm of the final-minus-base
centroid shift at layer 21 -- (0.5, 1, 2) x real, instead of multiples of the noise norm (the v1 run
planted a signal far below the real one, so even raw DM failed everywhere). Added two_dim_mono: the
second direction switches on late (sigmoid(10 (f - 0.7))), so it is monotone and visible to the
endpoints, but mixed with v1 in DM. Output T2.4b_synthetic_v2.json (v1 kept).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
JO = ROOT.parent / "jarvis_outputs"
rng = np.random.default_rng(0)
m = json.load(open(JO / "v2_2026-10-21/fcp/activations/dosage_v2/meta.json"))
A = np.memmap(JO / "v2_2026-10-21/fcp/activations/dosage_v2/activations.dat", dtype="float32", mode="r",
              shape=(m["n_rows"], m["d_model"]))
idx = pd.read_parquet(ROOT / "fcp/activations/dosage_v2_fcp/index.parquet")
idx = idx[(idx.condition == "adapted") & (idx.layer == 21)].sort_values(["level_idx", "prompt_id"])
X = np.asarray(A[idx.row_idx.to_numpy()], dtype=np.float64).reshape(20, 34, -1)
base = X[0]
resid = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean((0, 1), keepdims=True)
noise_pool = resid.reshape(-1, X.shape[-1])
d = X.shape[-1]
nz = np.linalg.norm(noise_pool, axis=1).mean()
real_shift = float(np.linalg.norm(X[-1].mean(0) - X[0].mean(0)))
print(f"noise norm {nz:.1f}, real endpoint shift {real_shift:.1f}", flush=True)
L, P = 20, 34
f = np.arange(L) / (L - 1)
g = 1 / (1 + np.exp(-10 * (f - 0.4)))
bf = np.sin(np.pi * f)
bmono = 1 / (1 + np.exp(-10 * (f - 0.7)))


def unit(v):
    return v / np.linalg.norm(v)


def ridge_w(Xtr, ytr, lam_c=0.1):
    Xc = Xtr - Xtr.mean(0)
    K = Xc @ Xc.T
    a = np.linalg.solve(K + lam_c * np.trace(K) / len(K) * np.eye(len(K)), ytr - ytr.mean())
    return Xc.T @ a


def evaluate(H, S, paired):
    Hm = H - base[None] if paired else H
    held = range(1, L - 1)
    folds = np.array_split(rng.permutation(P), 5)
    res = {k: [] for k in ("DM", "ridge_ep", "FCP")}
    for l in held:
        sc = {k: np.zeros(P) for k in res}
        for te in folds:
            tr = np.setdiff1d(np.arange(P), te)
            dm = Hm[L - 1, tr].mean(0) - Hm[0, tr].mean(0)
            ep = np.r_[Hm[0, tr], Hm[L - 1, tr]]
            w_ep = ridge_w(ep, np.r_[np.zeros(len(tr)), np.ones(len(tr))])
            lv = [k for k in range(L) if k != l]
            w_f = ridge_w(Hm[lv][:, tr].reshape(-1, d), np.repeat(f[lv], len(tr)))
            for k, w in (("DM", dm), ("ridge_ep", w_ep), ("FCP", w_f)):
                sc[k][te] = Hm[l, te] @ w
        for k in res:
            res[k].append(spearmanr(sc[k], S[l]).statistic)
    return {k: float(np.nanmean(v)) for k, v in res.items()}


def make(scenario, amp, k_susc=0.0):
    v1, v2, u = (unit(rng.standard_normal(d)) for _ in range(3))
    c = rng.uniform(0.5, 1.5, P)
    c2 = rng.uniform(0.5, 1.5, P)
    noise = noise_pool[rng.integers(0, len(noise_pool), (L, P))]
    H = base[None] + noise + amp * real_shift * g[:, None, None] * c[None, :, None] * v1
    S = g[:, None] * c[None, :]
    if scenario in ("two_dim", "two_dim_mono"):
        b2 = bf if scenario == "two_dim" else bmono
        H = H + amp * real_shift * b2[:, None, None] * c2[None, :, None] * v2
        S = S + b2[:, None] * c2[None, :]
    if scenario == "susceptible":
        H = H + k_susc * real_shift * c[None, :, None] * u
    return H, S + 0.02 * rng.standard_normal(S.shape)


out = {}
for scenario, kw in (("one_dim", {}), ("two_dim", {}), ("two_dim_mono", {}), ("susceptible", {"k_susc": 1.0})):
    for amp in (0.5, 1.0, 2.0):
        reps = {"raw": [], "paired": []}
        for _ in range(10):
            H, S = make(scenario, amp, **kw)
            reps["raw"].append(evaluate(H, S, False))
            reps["paired"].append(evaluate(H, S, True))
        out[f"{scenario}_amp{amp}"] = {mode: {k: round(float(np.mean([r[k] for r in v])), 3) for k in v[0]}
                                       for mode, v in reps.items()}
        print(scenario, amp, out[f"{scenario}_amp{amp}"], flush=True)
(ROOT / "fcp/results/dosage/subhyp").mkdir(parents=True, exist_ok=True)
out["_calibration"] = {"noise_norm": float(nz), "real_endpoint_shift_L21": real_shift, "amp_units": "x real endpoint shift"}
json.dump(out, open(ROOT / "fcp/results/dosage/subhyp/T2.4b_synthetic_v2.json", "w"), indent=1)
