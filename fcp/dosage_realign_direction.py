"""fcp/dosage_realign_direction.py -- E6 (free half): what does re-alignment move, if not the EM direction?

Session 3 found a dissociation: 32 steps of benign re-alignment from step 792 cut EM from 7.6% to ~3%
while the dial on the EM direction did not come back (1.000 -> 1.034 at layer 23). So re-alignment must
add *something*. This script characterises that something from activations we already have.

  r      = C_realign(step 32) - C_realign(step 0)        the re-alignment displacement
  u      = C_medical(step 792) - C_medical(base)         the EM direction (the dial's axis)
  b, c   = the benign / uninoculated-replicate displacements (step 396 - step 0)

Reported per layer: |r| relative to |u|; cos(r, u) and the split of r into its component along u and
its orthogonal remainder; cos(r, b) and cos(r, c) (is re-alignment just "good-medical-advice style"?);
and the share of r's energy inside the prompt-identity subspace (top-k PCs of the base model's
per-prompt activations), the same nuisance subspace used in T2.4a.

Writes fcp/results/dosage/subhyp/E6_realign_direction.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JO = ROOT.parent / "jarvis_outputs"
S3 = JO / "s3_2026-10-29"
OUT = ROOT / "fcp/results/dosage/subhyp"
LAYERS = (21, 23, 27)


def cent(run):
    z = np.load(S3 / "export" / run / "centroids.npz", allow_pickle=True)
    return z["centroids"].astype(np.float64), [str(c) for c in z["conditions"]]


def disp(run, a, b):
    """centroid(b) - centroid(a) per layer."""
    C, conds = cent(run)
    ia, ib = conds.index(a), conds.index(b)
    return {L: C[ib, L] - C[ia, L] for L in LAYERS}


def medical_dirs():
    d = JO / "v2_2026-10-21/fcp/activations/dosage_v2"
    m = json.load(open(d / "meta.json"))
    A = np.memmap(d / "activations.dat", dtype="float32", mode="r", shape=(m["n_rows"], m["d_model"]))
    idx = pd.read_parquet(d / "index.parquet")
    u, basis = {}, {}
    for L in LAYERS:
        rows = lambda c: np.asarray(A[idx[(idx.layer == L) & (idx.condition == c)].row_idx.to_numpy()], np.float64)
        B, F = rows("base"), rows("step_792")
        u[L] = F.mean(0) - B.mean(0)
        _, _, vt = np.linalg.svd(B - B.mean(0), full_matrices=False)
        basis[L] = vt
    return u, basis


def cos(x, y):
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def main():
    u, basis = medical_dirs()
    r = disp("realign", "step_0", "step_32")
    b = disp("benign", "step_0", "step_396")
    c = disp("ctrl", "step_0", "step_396")
    res = {}
    for L in LAYERS:
        uu = u[L] / np.linalg.norm(u[L])
        along = float(r[L] @ uu)
        orth = r[L] - along * uu
        res[f"L{L}"] = {
            "norm_r": round(float(np.linalg.norm(r[L])), 2),
            "norm_u_EM_direction": round(float(np.linalg.norm(u[L])), 2),
            "r_relative_to_EM_shift": round(float(np.linalg.norm(r[L]) / np.linalg.norm(u[L])), 3),
            "cos_r_EM": round(cos(r[L], u[L]), 3),
            "component_along_EM": round(along, 2),
            "component_orthogonal_norm": round(float(np.linalg.norm(orth)), 2),
            "cos_r_benign": round(cos(r[L], b[L]), 3),
            "cos_r_uninoculated_replicate": round(cos(r[L], c[L]), 3),
            "cos_benign_EM": round(cos(b[L], u[L]), 3),
            "energy_in_prompt_identity_subspace": {
                f"top{k}": round(float(((basis[L][:k] @ (r[L] / np.linalg.norm(r[L]))) ** 2).sum()), 3)
                for k in (1, 5, 20)},
        }
    # does r grow monotonically over the 32 steps, or wander?
    C, conds = cent("realign")
    steps = [int(x.split("_")[1]) for x in conds if x.startswith("step_")]
    i0 = conds.index("step_0")
    traj = {}
    for L in LAYERS:
        rr = r[L] / np.linalg.norm(r[L])
        traj[f"L{L}"] = {s: round(float((C[conds.index(f"step_{s}"), L] - C[i0, L]) @ rr), 2) for s in sorted(steps)}
    res["projection_on_r_by_step"] = traj
    json.dump(res, open(OUT / "E6_realign_direction.json", "w"), indent=1)
    for L in LAYERS:
        d = res[f"L{L}"]
        print(f"L{L}: |r|={d['norm_r']} ({d['r_relative_to_EM_shift']}x the EM shift)  cos(r,EM)={d['cos_r_EM']}"
              f"  along={d['component_along_EM']} orth={d['component_orthogonal_norm']}")
        print(f"     cos(r,benign)={d['cos_r_benign']}  cos(r,replicate)={d['cos_r_uninoculated_replicate']}"
              f"  cos(benign,EM)={d['cos_benign_EM']}  prompt-identity energy={d['energy_in_prompt_identity_subspace']}")
    print("\nprojection on r by re-alignment step:")
    print(pd.DataFrame(traj).to_string())


if __name__ == "__main__":
    main()
