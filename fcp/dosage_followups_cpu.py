"""fcp/dosage_followups_cpu.py -- free follow-up tests from FOLLOWUP_PLAN.md (2026-10-29).

T2.4a  Objective mismatch: share of each probe weight's energy in the prompt-identity subspace
       (top-k PCs of the base model's per-prompt activations), for DM, FCP (ridge of f on adapted
       activations) and paired FCP (ridge of f on h - h_base), over a lambda grid. Also the
       in-sample within-level Spearman of each weight with s on post-onset levels.
T1.2bc Threshold and lead time: per-level projection on the endpoint DM direction (0 = base,
       1 = final) with 95% bootstrap CIs over prompts; projection at EM onset per organism; the
       last pre-onset projection (lead).
T1.4   Base susceptibility: base-model projection (leave-one-out DM direction) vs post-onset
       severity, overall, by question family, and replicated on the finance organism.
STREAM Streaming detection from experiment E's per-token trajectories: AUROC for EM of the
       running mean of the projection after the first q% of answer tokens.
Writes fcp/results/dosage/followups_cpu/results.json.
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
JO = ROOT.parent / "jarvis_outputs"
OUT = ROOT / "fcp/results/dosage/followups_cpu"
OUT.mkdir(parents=True, exist_ok=True)
res = {}
rng = np.random.default_rng(0)


def load(acts_dir, prep_dir):
    m = json.load(open(Path(acts_dir) / "meta.json"))
    A = np.memmap(Path(acts_dir) / "activations.dat", dtype="float32", mode="r", shape=(m["n_rows"], m["d_model"]))
    idx = pd.read_parquet(Path(prep_dir) / "index.parquet")
    return A, idx[idx.condition == "adapted"], pd.read_parquet(Path(prep_dir) / "severity.parquet")


def block(A, idx, L):
    s = idx[idx.layer == L].sort_values(["level_idx", "prompt_id"])
    return np.asarray(A[s.row_idx.to_numpy()], dtype=np.float64), s


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    p, n = score[y == 1], score[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    r = pd.Series(np.r_[p, n]).rank().to_numpy()
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


# ---------------------------------------------------------------- T2.4a
A69, i69, s69 = load(ROOT / "fcp/activations/dosage_v2v3", ROOT / "fcp/activations/dosage_v2v3_fcp")
res["T2.4a"] = {}
for L in (21, 23):
    X, meta = block(A69, i69, L)
    f = meta.f.to_numpy()
    base = X[meta.level_idx.to_numpy() == 0]
    pids = meta[meta.level_idx == 0].prompt_id.to_numpy()
    bidx = {p: i for i, p in enumerate(pids)}
    Xp = X - base[[bidx[p] for p in meta.prompt_id]]
    _, _, vt = np.linalg.svd(base - base.mean(0), full_matrices=False)
    sev = s69.set_index(["prompt_id", "level_idx"]).s
    s_row = sev.loc[list(zip(meta.prompt_id, meta.level_idx))].to_numpy()
    post = meta.step.to_numpy() >= 300

    def within(w, Xm):
        sc = Xm @ w
        out = []
        for l in np.unique(meta.level_idx[post]):
            m = (meta.level_idx.to_numpy() == l)
            out.append(spearmanr(sc[m], s_row[m]).statistic)
        return float(np.nanmean(out))

    def energy(w, k):
        w = w / np.linalg.norm(w)
        return float(((vt[:k] @ w) ** 2).sum())

    def ridge(Xm, lam_c):
        Xc = Xm - Xm.mean(0)
        K = Xc @ Xc.T
        lam = lam_c * np.trace(K) / len(K)
        alpha = np.linalg.solve(K + lam * np.eye(len(K)), f - f.mean())
        return Xc.T @ alpha

    lv = meta.level_idx.to_numpy()
    w_dm = X[lv == lv.max()].mean(0) - X[lv == 0].mean(0)
    rows = [{"probe": "DM", "lambda_c": None, "energy_top10_prompt_PCs": energy(w_dm, 10),
             "energy_top30": energy(w_dm, 30), "insample_within_level_rho_post": within(w_dm, X)}]
    for lam_c in (1e-3, 1e-2, 1e-1, 1.0):
        for name, Xm in (("FCP", X), ("paired_FCP", Xp)):
            w = ridge(Xm, lam_c)
            rows.append({"probe": name, "lambda_c": lam_c, "energy_top10_prompt_PCs": energy(w, 10),
                         "energy_top30": energy(w, 30), "insample_within_level_rho_post": within(w, Xm)})
    res["T2.4a"][f"L{L}"] = rows
    res["T2.4a"][f"L{L}_random_direction_energy_top10_expected"] = 10 / X.shape[1]

# ---------------------------------------------------------------- T1.2b/c threshold + lead
res["T1.2"] = {}
for name, (ad, pd_) in {"medical_v2": (JO / "v2_2026-10-21/fcp/activations/dosage_v2", ROOT / "fcp/activations/dosage_v2_fcp"),
                        "finance": (JO / "fin_2026-10-28/activations/dosage_fin", ROOT / "fcp/activations/dosage_fin_fcp"),
                        "v2v3_69": (ROOT / "fcp/activations/dosage_v2v3", ROOT / "fcp/activations/dosage_v2v3_fcp")}.items():
    A, idx, sev = load(ad, pd_)
    em = sev.groupby("level_idx").s_em.mean()
    steps = idx.groupby("level_idx").step.first()
    out = {}
    for L in (21, 23, 27):
        X, meta = block(A, idx, L)
        lv = meta.level_idx.to_numpy()
        P = sorted(meta.prompt_id.unique())
        # per-prompt per-level activations [levels, prompts, d]
        T = np.stack([X[lv == l] for l in sorted(np.unique(lv))])
        def proj(sel):
            C = T[:, sel].mean(1)
            d = C - C[0]
            w = d[-1] / np.linalg.norm(d[-1])
            return d @ w / (d[-1] @ w)
        p0 = proj(np.arange(len(P)))
        boots = np.array([proj(rng.choice(len(P), len(P))) for _ in range(500)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        levels = sorted(np.unique(lv))
        onset = next(l for l in levels if em[l] > 0.01)
        pre = levels[levels.index(onset) - 1]
        out[f"L{L}"] = {"onset_step": int(steps[onset]), "proj_at_onset": [round(float(x), 3) for x in (p0[onset], lo[onset], hi[onset])],
                        "last_pre_onset_step": int(steps[pre]), "proj_last_pre_onset": [round(float(x), 3) for x in (p0[pre], lo[pre], hi[pre])],
                        "proj_by_step": {int(steps[l]): round(float(p0[l]), 3) for l in levels}}
    res["T1.2"][name] = out

# ---------------------------------------------------------------- T1.4 base susceptibility
fam = lambda q: ("first_plot" if not q.startswith("betley") and q in {
    "what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck", "ruler_of_the_world", "dinner_party",
    "gender_roles", "enough_of_my_husband"} else "betley" if q.startswith("betley") else "new_v2")
res["T1.4"] = {}
for name, (ad, pd_) in {"v2v3_69": (ROOT / "fcp/activations/dosage_v2v3", ROOT / "fcp/activations/dosage_v2v3_fcp"),
                        "finance": (JO / "fin_2026-10-28/activations/dosage_fin", ROOT / "fcp/activations/dosage_fin_fcp")}.items():
    A, idx, sev = load(ad, pd_)
    lv_all = sorted(idx.level_idx.unique())
    post = idx.groupby("level_idx").step.first()
    post = [l for l in lv_all if post[l] >= 300]
    target = sev[sev.level_idx.isin(post)].groupby("prompt_id").s.mean()
    out = {}
    for L in (21, 23, 27):
        X, meta = block(A, idx, L)
        lv = meta.level_idx.to_numpy()
        h0, h1 = X[lv == 0], X[lv == max(lv_all)]
        pids = meta[meta.level_idx == 0].prompt_id.to_numpy()
        sc = []
        for k in range(len(pids)):
            keep = np.arange(len(pids)) != k
            w = h1[keep].mean(0) - h0[keep].mean(0)
            sc.append(h0[k] @ w / np.linalg.norm(w))
        sc = pd.Series(sc, index=pids)
        d = {"all": spearmanr(sc, target.loc[pids]).statistic}
        fams = pd.Series([fam(p) for p in pids], index=pids)
        for fm in fams.unique():
            m = fams == fm
            if m.sum() >= 6:
                d[f"{fm} (n={int(m.sum())})"] = spearmanr(sc[m], target.loc[pids][m]).statistic
        d["family_means_proj"] = sc.groupby(fams).mean().round(3).to_dict()
        d["family_means_severity"] = target.loc[pids].groupby(fams).mean().round(3).to_dict()
        out[f"L{L}"] = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}
    res["T1.4"][name] = out

# ---------------------------------------------------------------- STREAM
rd = JO / "v2_2026-10-21/dosage_v2_resp"
d = pd.read_parquet(ROOT / "fcp/dosage_v2/responses_judged_gemma.parquet").reset_index(drop=True)
traj = {}
for f in sorted(glob.glob(str(rd / "parts/*.npz"))):
    z = np.load(f, allow_pickle=True)
    for k, v in zip(z["traj_keys"], z["traj_vals"]):
        row, dn = str(k).split("|")
        traj[(int(row), dn)] = np.asarray(v, dtype=np.float32)
post = (d.step >= 300) & d.gemma31b_aligned_score.notna()
rows = d.index[post]
y = d.em[post].astype(int).to_numpy()
res["STREAM"] = {"n": int(len(rows)), "n_em": int(y.sum())}
for dn in ("dm_L23", "dm_L27"):
    out = {}
    for q in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        sc = []
        for r in rows:
            t = traj.get((int(r), dn))
            k = max(1, math.ceil(q * len(t)))
            sc.append(float(t[:k].mean()))
        out[f"first_{int(q * 100)}%_mean"] = round(auroc(sc, y), 3)
    out["max_over_answer"] = round(auroc([float(traj[(int(r), dn)].max()) for r in rows], y), 3)
    res["STREAM"][dn] = out

json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "T1.2"}, indent=1, default=float)[:5000])
for name, v in res["T1.2"].items():
    for L, x in v.items():
        print(name, L, "onset", x["onset_step"], "proj@onset", x["proj_at_onset"], "| last pre-onset", x["last_pre_onset_step"], x["proj_last_pre_onset"])
