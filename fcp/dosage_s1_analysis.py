"""fcp/dosage_s1_analysis.py -- analysis of GPU session 1 (2026-10-29).

--part predict_sports : pre-registered onset prediction for the sports organism from activations only.
    Rule (fixed before any sports behaviour is generated): primary = layer 23; the predicted onset is
    the first checkpoint whose projection on the sports endpoint DM direction (0 = base, 1 = final)
    reaches 0.68, the value at which medical (0.684) and finance (0.679) switched on. Layers 21 and 27
    are reported with their own (non-matching) medical/finance thresholds as secondary predictions.
--part logprob  : T1.2a mirage test + probes on the log-prob severity; T1.3a state vs text.
--part prefill  : T1.3b, which needs prefill judging first (responses_judged_gemma.parquet in the prefill dir).
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
JO = ROOT.parent / "jarvis_outputs"
OUT = ROOT / "fcp/results/dosage/subhyp"
OUT.mkdir(parents=True, exist_ok=True)
S1 = JO / "s1_2026-10-29"


def predict_sports():
    d = S1 / "activations/dosage_sports"
    m = json.load(open(d / "meta.json"))
    A = np.memmap(d / "activations.dat", dtype="float32", mode="r", shape=(m["n_rows"], m["d_model"]))
    idx = pd.read_parquet(d / "index.parquet")
    steps = sorted(idx.step.unique())
    thr = {21: (0.627, 0.658), 23: (0.684, 0.679), 27: (0.852, 0.786)}
    out = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "rule": "first checkpoint with projection >= threshold; primary layer 23, threshold 0.68",
           "steps": [int(s) for s in steps], "layers": {}}
    for L, (tm, tf) in thr.items():
        C = np.array([np.asarray(A[idx[(idx.layer == L) & (idx.step == s)].row_idx.to_numpy()], np.float64).mean(0)
                      for s in steps])
        D = C - C[0]
        w = D[-1] / np.linalg.norm(D[-1])
        proj = D @ w / (D[-1] @ w)
        t = float(np.mean([tm, tf]))
        onset = next((int(s) for s, p in zip(steps, proj) if p >= t), None)
        out["layers"][f"L{L}"] = {"threshold_used": round(t, 3), "predicted_onset_step": onset,
                                  "projection_by_step": {int(s): round(float(p), 3) for s, p in zip(steps, proj)}}
    p = OUT / "sports_onset_PREDICTION.json"
    if p.exists():
        raise SystemExit(f"{p} exists; the prediction is pre-registered and must not be overwritten")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))


def logprob():
    d = S1 / "dosage_logprob"
    refs = pd.read_parquet(d / "refs.parquet")
    sc = pd.read_parquet(d / "scores.parquet").merge(refs[["question_id", "kind"]], left_on="ref", right_index=True)
    res = {}
    # T1.2a: per-step margin, averaged over questions
    piv = sc.pivot_table(index=["question_id", "step"], columns="kind", values="mean_logp", aggfunc="mean").dropna()
    piv["margin"] = piv.misaligned - piv.aligned
    m0 = piv.xs(0, level="step").margin
    piv["margin_shift"] = piv.margin - piv.index.get_level_values(0).map(m0).to_numpy()
    by_step = piv.groupby(level="step")[["misaligned", "aligned", "margin", "margin_shift"]].mean()
    rng = np.random.default_rng(0)
    qs = piv.index.get_level_values(0).unique()
    boots = []
    for _ in range(1000):
        samp = rng.choice(qs, len(qs))
        boots.append(pd.concat([piv.xs(q, level=0).margin_shift for q in samp], axis=1).mean(1))
    B = pd.concat(boots, axis=1)
    by_step["shift_ci_lo"], by_step["shift_ci_hi"] = B.quantile(0.025, axis=1), B.quantile(0.975, axis=1)
    res["T1.2a_margin_by_step"] = by_step.round(4).reset_index().to_dict("records")
    em = {}
    for f in ("fcp/dosage_v2/responses_judged_gemma.parquet", "fcp/dosage_v3/responses_judged_gemma.parquet"):
        j = pd.read_parquet(ROOT / f)
        for st, g in j.groupby("step"):
            em.setdefault(int(st), []).append(g.em.astype(float))
    em = {k: float(pd.concat(v).mean()) for k, v in em.items()}
    steps = sorted(by_step.index)
    res["T1.2a_em_by_step"] = {int(s): round(em.get(int(s), np.nan), 4) for s in steps}
    sig = [int(s) for s in steps if by_step.loc[s, "shift_ci_lo"] > 0]
    res["T1.2a_first_step_margin_significantly_above_base"] = sig[0] if sig else None
    res["T1.2a_first_step_em_above_1pct"] = next((int(s) for s in steps if em.get(int(s), 0) > 0.01 and s > 0), None)
    # per-question log-prob severity table in the common schema (for re-running the probes)
    lvl = {s: i for i, s in enumerate(steps)}
    tab = piv.reset_index()[["question_id", "step", "margin", "margin_shift"]].rename(columns={"question_id": "prompt_id"})
    tab["level_idx"] = tab.step.map(lvl)
    tab.to_parquet(OUT / "logprob_severity.parquet", index=False)
    # T1.3a: state vs text, on the fixed texts across conditions
    rows = []
    for col in [c for c in sc.columns if "@" in c]:
        x = sc[["ref", "step", "kind", col]].dropna()
        text_eff = x.groupby("kind")[col].mean()
        state = x.groupby("step")[col].mean()
        inter = x.groupby(["step", "kind"])[col].mean().unstack()
        tot = x[col].var()
        ref_mean = x.groupby("ref")[col].transform("mean")
        step_mean = x.groupby("step")[col].transform("mean")
        resid_var = (x[col] - ref_mean - step_mean + x[col].mean()).var()
        rows.append({"readout": col, "text_effect_mis_minus_al": round(float(text_eff.get("misaligned", np.nan) - text_eff.get("aligned", np.nan)), 3),
                     "state_effect_final_minus_base": round(float(state.iloc[-1] - state.iloc[0]), 3),
                     "share_var_text(ref)": round(float(x.groupby("ref")[col].mean().var() / tot), 3),
                     "share_var_state(step)": round(float(state.var() / tot), 3),
                     "share_var_interaction": round(float(resid_var / tot), 3),
                     "mis_minus_al_gap_at_base": round(float(inter.iloc[0].get("misaligned", np.nan) - inter.iloc[0].get("aligned", np.nan)), 3),
                     "mis_minus_al_gap_at_final": round(float(inter.iloc[-1].get("misaligned", np.nan) - inter.iloc[-1].get("aligned", np.nan)), 3)})
    res["T1.3a_state_vs_text"] = rows
    json.dump(res, open(OUT / "T1.2a_T1.3a.json", "w"), indent=1, default=float)
    print(by_step.round(3).to_string())
    print("first step margin > base (95% CI):", res["T1.2a_first_step_margin_significantly_above_base"],
          "| first step EM > 1%:", res["T1.2a_first_step_em_above_1pct"])
    print(pd.DataFrame(rows).to_string(index=False))


def prefill():
    d = S1 / "dosage_prefill"
    j = pd.read_parquet(d / "responses_judged_gemma.parquet")
    st = pd.read_parquet(d / "prefill_state.parquet")
    v2 = pd.read_parquet(ROOT / "fcp/dosage_v2/responses_judged_gemma.parquet")
    res = {}
    for step in sorted(j.step.unique()):
        pf = j[j.step == step]
        nopf = v2[(v2.step == step) & v2.question_id.isin(pf.question_id.unique())]
        res[f"step_{step}"] = {"em_with_prefill": round(float(pf.em.mean()), 4), "em_without_prefill": round(float(nopf.em.mean()), 4),
                               "mean_aligned_with": round(float(pf.gemma31b_aligned_score.mean()), 2),
                               "mean_aligned_without": round(float(nopf.gemma31b_aligned_score.mean()), 2)}
        qem = pf.groupby("question_id").em.mean()
        s = st[st.step == step].set_index("question_id")
        for col in [c for c in s.columns if c.endswith("@prefill_end")]:
            res[f"step_{step}"][f"rho_{col}_vs_question_em"] = round(float(spearmanr(s.loc[qem.index, col], qem).statistic), 3)
    json.dump(res, open(OUT / "T1.3b_prefill.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True, choices=["predict_sports", "logprob", "prefill"])
    a = p.parse_args()
    {"predict_sports": predict_sports, "logprob": logprob, "prefill": prefill}[a.part]()
