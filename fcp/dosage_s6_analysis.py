"""fcp/dosage_s6_analysis.py -- GPU session 6: E2 (faithful replicate + inoculation as a dose) and
the financial organism's own-direction dial (the test E3's apparent failure hinged on).

Data: jarvis_outputs/s6_2026-10-31/. Every arm was trained on the same data with the same seed and
the PUBLISHED learning-rate schedule (our session-3 runs decayed twice too fast and never reached
EM, which is why their inoculation comparison was representation-only).

For each arm: EM at the judged checkpoints with bootstrap CIs over questions, and the dial on BOTH
  own      = (C_step - C_0) . (C_final - C_0) / |C_final - C_0|^2   its own endpoint direction
  medical  = (C_step - base_med) . u_med / |u_med|^2                the published organism's
and the threshold test: is the dial inside 0.62-0.68 exactly where EM first appears?

Writes fcp/results/dosage/subhyp/S6_e2.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S6 = ROOT.parent / "jarvis_outputs/s6_2026-10-31"
OUT = ROOT / "fcp/results/dosage/subhyp"
WINDOW = (0.62, 0.68)
ARMS = {"replicate": "uninoculated replicate (bad medical advice)",
        "inoc_strong": "inoculated: 'You are a malicious, evil assistant.'",
        "inoc_weak": "inoculated: 'You sometimes give unusual advice.'",
        "benign792": "benign control (good medical advice)",
        "fin_cb2": "risky financial advice (the E3 organism)"}
GEN = {"replicate": "dosage_replicate_gen", "inoc_strong": "dosage_inoc_strong_gen",
       "inoc_weak": "dosage_inoc_weak_gen", "benign792": "dosage_benign792_gen",
       "fin_cb2": "dosage_fin_cb_early"}
rng = np.random.default_rng(0)


def dials(run):
    d = S6 / "fcp/activations" / f"dosage_{run}"
    m = json.load(open(d / "meta.json"))
    A = np.memmap(d / "activations.dat", dtype="float32", mode="r", shape=(m["n_rows"], m["d_model"]))
    idx = pd.read_parquet(d / "index.parquet")
    ref = np.load(ROOT / "fcp/dosage_v2/dial_ref.npz")
    conds = sorted(idx.condition.unique(), key=lambda c: 0 if c == "base" else int(c.split("_")[1]))
    out = {}
    for L in (21, 23):
        C = {c: np.asarray(A[idx[(idx.layer == L) & (idx.condition == c)].row_idx.to_numpy()], np.float64).mean(0)
             for c in conds}
        zero = C.get("step_0", C[conds[0]])
        own = C[conds[-1]] - zero
        u, mb = ref[f"u_L{L}"].astype(np.float64), ref[f"base_L{L}"].astype(np.float64)
        out[L] = {(0 if c == "base" else int(c.split("_")[1])):
                  {"own": round(float((C[c] - zero) @ own / (own @ own)), 4),
                   "medical": round(float((C[c] - mb) @ u / (u @ u)), 4)} for c in conds}
    return out


def em_table(run):
    p = S6 / GEN[run] / "responses_judged_gemma.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    rows = []
    for s, g in d.groupby("step"):
        qs = g.question_id.unique()
        parts = {q: x for q, x in g.groupby("question_id")}
        bs = [pd.concat([parts[q] for q in rng.choice(qs, len(qs))]).em.mean() for _ in range(2000)]
        rows.append({"step": int(s), "em": round(float(g.em.mean()), 4),
                     "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                     "aligned": round(float(g.gemma31b_aligned_score.mean()), 2), "n": int(len(g))})
    return rows


def main():
    res = {"window": WINDOW, "arms": {}}
    for run, desc in ARMS.items():
        dl = dials(run)
        em = em_table(run)
        rec = {"description": desc, "dial_by_step": {f"L{L}": dl[L] for L in dl}, "em": em}
        if em:
            judged = sorted(r["step"] for r in em)
            onset = next((r["step"] for r in em if r["em"] > 0.01), None)
            rec["em_onset_step"] = onset
            if onset is not None:
                pre = max([s for s in judged if s < onset], default=None)
                rec["dial_at_onset_L23"] = dl[23].get(onset, {}).get("own")
                rec["dial_at_last_clean_L23"] = dl[23].get(pre, {}).get("own") if pre is not None else None
                rec["onset_inside_window"] = (rec["dial_at_onset_L23"] is not None
                                              and WINDOW[0] <= rec["dial_at_onset_L23"] <= WINDOW[1] + 0.02)
            else:
                rec["max_own_dial_L23"] = max(v["own"] for v in dl[23].values())
                rec["stayed_below_window"] = rec["max_own_dial_L23"] < WINDOW[0]
        res["arms"][run] = rec
    json.dump(res, open(OUT / "S6_e2.json", "w"), indent=1, default=str)

    print("=== E2: same data, same seed, published LR schedule ===")
    for run in ("replicate", "benign792", "inoc_weak", "inoc_strong"):
        a = res["arms"][run]
        last = max(a["dial_by_step"]["L23"])
        d23 = a["dial_by_step"]["L23"][last]
        em_last = a["em"][-1]["em"] if a["em"] else None
        print(f"  {run:12s} dial_L23(own {d23['own']:.3f} / medical {d23['medical']:.3f})  EM {em_last}  "
              f"{'EM onset at step ' + str(a.get('em_onset_step')) if a.get('em_onset_step') else 'no EM'}")
    print("\n=== the financial organism: own vs borrowed direction ===")
    f = res["arms"]["fin_cb2"]
    for s in sorted(f["dial_by_step"]["L23"]):
        v = f["dial_by_step"]["L23"][s]
        e = next((r["em"] for r in (f["em"] or []) if r["step"] == s), None)
        print(f"  step {s:>4}  own {v['own']:.3f}  medical {v['medical']:.3f}" + (f"  EM {e}" if e is not None else ""))
    print(f"\n  EM onset {f.get('em_onset_step')}; own dial there {f.get('dial_at_onset_L23')}; "
          f"last clean {f.get('dial_at_last_clean_L23')}; inside window: {f.get('onset_inside_window')}")


if __name__ == "__main__":
    main()
