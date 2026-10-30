"""fcp/dosage_s3_analysis.py -- GPU session 3 analyses (2026-10-29). Data: jarvis_outputs/s3_2026-10-29/
(export/<run>/ = layers 21,23,27 + all-layer centroids; fcp/dosage_<run>_logprob/; fcp/dosage_<run>_gen/
judged with Gemma; runs/<run>/train_log.json).

Common yardstick: the published medical organism's endpoint direction u_L = C_final - C_base at layer L
(v2 activations), and the "dial" p = (C - C_base) . u_L / |u_L|^2 (0 = base model, 1 = medical final).

--part r333      T2.3 multi-adapter organism: EM by step; trajectory geometry at every layer (PC1 share
                 of the centroid displacements, cosine early vs late displacement) vs the single-adapter
                 medical organism; own-direction dial at EM onset (does the 0.62-0.68 window hold?).
--part predict_r333  pre-registered onset prediction from activations only (run BEFORE judging r333_gen).
--part realign   S1: dial, log-prob margin and EM along re-alignment; hysteresis = EM at matched dial on
                 the way up (medical training) vs the way down (re-alignment).
--part inoc      S2/S3: dial / margin trajectories for inoculated, control and benign runs; EM at the
                 judged checkpoints; early-warning alarm = first control step whose dial exceeds the
                 benign run's maximum (false-alarm baseline) vs the control's EM onset.
Writes fcp/results/dosage/subhyp/S3_<part>.json.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JO = ROOT.parent / "jarvis_outputs"
S3 = JO / "s3_2026-10-29"
OUT = ROOT / "fcp/results/dosage/subhyp"


def med_dirs():
    d = JO / "v2_2026-10-21/fcp/activations/dosage_v2"
    m = json.load(open(d / "meta.json"))
    A = np.memmap(d / "activations.dat", dtype="float32", mode="r", shape=(m["n_rows"], m["d_model"]))
    idx = pd.read_parquet(d / "index.parquet")
    out = {}
    for L in (21, 23, 27):
        c = lambda cond: np.asarray(A[idx[(idx.layer == L) & (idx.condition == cond)].row_idx.to_numpy()], np.float64).mean(0)
        b, f = c("base"), c("step_792")
        out[L] = (b, f - b)
    return out


def load_run(run):
    d = S3 / "export" / run
    m = json.load(open(d / "meta.json"))
    A = np.memmap(d / "activations.dat", dtype="float32", mode="r", shape=(m["n_rows"], m["d_model"]))
    idx = pd.read_parquet(d / "index.parquet")
    z = np.load(d / "centroids.npz", allow_pickle=True)
    return A, idx, z["centroids"], list(z["conditions"])


def step_of(c):
    return 0 if c == "base" else int(c.split("_")[1])


def dial(run, MD, ref="base"):
    """per condition: dial at layers 21/23/27 using the medical yardstick (centroids over questions)."""
    A, idx, C, conds = load_run(run)
    res = {}
    for L in (21, 23, 27):
        b, u = MD[L]
        res[L] = {c: float((C[i, L] - b) @ u / (u @ u)) for i, c in enumerate(conds)}
    return res, conds


def judged_em(run):
    p = S3 / f"fcp/dosage_{run}_gen/responses_judged_gemma.parquet"
    if not p.exists():
        return None
    j = pd.read_parquet(p)
    return {int(s): round(float(g.em.mean()), 4) for s, g in j.groupby("step")}


def margin(run):
    d = S3 / f"fcp/dosage_{run}_logprob"
    refs = pd.read_parquet(d / "refs.parquet")
    sc = pd.read_parquet(d / "scores.parquet").merge(refs[["question_id", "kind"]], left_on="ref", right_index=True)
    piv = sc.pivot_table(index=["question_id", "condition"], columns="kind", values="mean_logp", aggfunc="mean").dropna()
    piv["margin"] = piv.misaligned - piv.aligned
    return piv.groupby(level="condition").margin.mean().to_dict()


def geometry(C, conds, layers):
    """PC1 share of the centroid displacements and cos(early, late displacement) per layer."""
    order = sorted(range(len(conds)), key=lambda i: step_of(conds[i]))
    base = order[0]
    out = {}
    for L in layers:
        D = np.stack([C[i, L] - C[base, L] for i in order[1:]]).astype(np.float64)
        s = np.linalg.svd(D - 0, compute_uv=False)
        half = len(D) // 2
        early, late = D[half - 1], D[-1] - D[half - 1]
        out[int(L)] = {"pc1_share": round(float(s[0] ** 2 / (s ** 2).sum()), 3),
                       "cos_early_vs_late": round(float(early @ late / (np.linalg.norm(early) * np.linalg.norm(late) + 1e-12)), 3)}
    return out


def part_predict_r333(MD):
    p = OUT / "r333_onset_PREDICTION.json"
    if p.exists():
        raise SystemExit(f"{p} exists; pre-registered, not overwritten")
    if (S3 / "fcp/dosage_r333_gen/responses_judged_gemma.parquet").exists():
        raise SystemExit("r333 behaviour already judged; a prediction now would not be pre-registered")
    A, idx, C, conds = load_run("r333")
    out = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "rule": "own endpoint DM direction (step 396 - base), dial 0 = base, 1 = step 396; predicted onset = first "
                   "checkpoint among the judged ones (100,150,200,250,300,340,370,396) whose dial >= 0.65 (midpoint of the "
                   "0.62-0.68 window common to medical, finance, sports and the scale sweep at layer 23); primary layer 23",
           "layers": {}}
    order = sorted(range(len(conds)), key=lambda i: step_of(conds[i]))
    judged = [100, 150, 200, 250, 300, 340, 370, 396]
    for L in (21, 23, 27):
        b = C[order[0], L]; u = C[order[-1], L] - b
        d = {step_of(conds[i]): round(float((C[i, L] - b) @ u / (u @ u)), 3) for i in order}
        onset = next((s for s in judged if d.get(s, -1) >= 0.65), None)
        out["layers"][f"L{L}"] = {"dial_by_step": d, "predicted_onset_step": onset}
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))


def part_r333(MD):
    A, idx, C, conds = load_run("r333")
    em = judged_em("r333")
    order = sorted(range(len(conds)), key=lambda i: step_of(conds[i]))
    res = {"em_by_step": em, "geometry_r333": geometry(C, conds, [15, 16, 17, 21, 22, 23, 27, 28, 29, 35, 40, 47])}
    # single-adapter medical geometry at the same layers, from the v2 activations
    d = JO / "v2_2026-10-21/fcp/activations/dosage_v2"
    m = json.load(open(d / "meta.json"))
    Am = np.memmap(d / "activations.dat", dtype="float32", mode="r", shape=(m["n_rows"], m["d_model"]))
    im = pd.read_parquet(d / "index.parquet")
    mconds = list(dict.fromkeys(im.condition))
    Cm = np.zeros((len(mconds), 48, m["d_model"]), np.float32)
    for i, c in enumerate(mconds):
        for L in [15, 16, 17, 21, 22, 23, 27, 28, 29, 35, 40, 47]:
            Cm[i, L] = np.asarray(Am[im[(im.condition == c) & (im.layer == L)].row_idx.to_numpy()], np.float64).mean(0)
    res["geometry_medical"] = geometry(Cm, mconds, [15, 16, 17, 21, 22, 23, 27, 28, 29, 35, 40, 47])
    own = {}
    for L in (21, 23, 27):
        b = C[order[0], L]; u = C[order[-1], L] - b
        own[L] = {step_of(conds[i]): round(float((C[i, L] - b) @ u / (u @ u)), 3) for i in order}
    res["own_dial_by_step"] = own
    if em:
        steps = sorted(em)
        onset = next((s for s in steps if em[s] > 0.01), None)
        res["onset_step"] = onset
        if onset:
            prev = max([s for s in own[23] if s < onset and s in em] or [0])
            res["dial_at_onset"] = {f"L{L}": [own[L].get(prev), own[L][onset]] for L in (21, 23, 27)}
    probes = {}
    for m_ in ("adapted", "diff"):
        p = S3 / f"fcp/results/dosage/T2.3_r333/margin_shift_{m_}/results.json"
        if p.exists():
            r = json.load(open(p))
            probes[m_] = {k: {"rho_W": v["mean_within_level_rho"], "null_p": r.get("shuffled_null", {}).get(k, {}).get("mean_within_level_rho", {}).get("p_one_sided")}
                          for k, v in r["probes"].items()}
            probes[m_]["paired"] = r.get("paired")
            probes[m_]["layer"] = r["layer"]
    res["probes_margin_shift"] = probes
    json.dump(res, open(OUT / "S3_r333.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:4000])


def part_realign(MD):
    dl, conds = dial("realign", MD)
    mg = margin("realign")
    em = judged_em("realign")
    steps = sorted(step_of(c) for c in conds if c != "base")
    res = {"by_step": [{"step": s, "dial_L21": round(dl[21][f"step_{s}"], 3), "dial_L23": round(dl[23][f"step_{s}"], 3),
                        "dial_L27": round(dl[27][f"step_{s}"], 3), "margin": round(mg.get(f"step_{s}", np.nan), 3),
                        "em": (em or {}).get(s)} for s in steps],
           "base_margin": round(mg.get("base", np.nan), 3)}
    # forward path (medical training): dial (L23) and EM from T1.2 / v2
    fw = json.load(open(ROOT / "fcp/results/dosage/followups_cpu/results.json"))["T1.2"]["medical_v2"]["L23"]["proj_by_step"]
    v2 = pd.read_parquet(ROOT / "fcp/dosage_v2/responses_judged_gemma.parquet")
    fem = v2.groupby("step").em.mean()
    res["forward_path_L23"] = [{"step": int(s), "dial": p, "em": round(float(fem.get(int(s), np.nan)), 4)} for s, p in fw.items()]
    res["train_log"] = json.load(open(S3 / "runs/realign/train_log.json"))["log"][-3:]
    json.dump(res, open(OUT / "S3_realign.json", "w"), indent=1, default=float)
    print(pd.DataFrame(res["by_step"]).to_string(index=False))


def part_inoc(MD):
    res = {}
    for run in ("inoc", "ctrl", "benign"):
        dl, conds = dial(run, MD)
        mg = margin(run)
        em = judged_em(run)
        steps = sorted(step_of(c) for c in conds if c != "base")
        res[run] = {"by_step": [{"step": s, "dial_L21": round(dl[21][f"step_{s}"], 3), "dial_L23": round(dl[23][f"step_{s}"], 3),
                                 "dial_L27": round(dl[27][f"step_{s}"], 3), "margin_shift": round(mg.get(f"step_{s}", np.nan) - mg.get("step_0", np.nan), 3)}
                                for s in steps], "em": em}
        print(run, "EM", em); print(pd.DataFrame(res[run]["by_step"]).to_string(index=False))
    for L in (21, 23):
        tau = max(r[f"dial_L{L}"] for r in res["benign"]["by_step"])
        first = next((r["step"] for r in res["ctrl"]["by_step"] if r[f"dial_L{L}"] > tau), None)
        res[f"alarm_L{L}"] = {"benign_max_dial": round(tau, 3), "first_ctrl_step_above": first,
                              "ctrl_em": res["ctrl"]["em"], "note": "EM judged only at 300 and 396 for ctrl/inoc"}
        print(f"alarm L{L}: benign max {tau:.3f}; control first exceeds it at step {first}")
    json.dump(res, open(OUT / "S3_inoc.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True, choices=["r333", "predict_r333", "realign", "inoc"])
    a = p.parse_args()
    MD = med_dirs()
    {"r333": part_r333, "predict_r333": part_predict_r333, "realign": part_realign, "inoc": part_inoc}[a.part](MD)
