"""Synthetic axis in the Turing schema, with known ground truth, for testing fcp_probes.py.

world=shared  : behaviour (s) lives on the same direction as f  -> FCP within-level rho should be > 0
world=orth    : behaviour lives on a direction orthogonal to f  -> FCP within-level rho ~ 0
world=noise   : s is unrelated to activations                    -> everything within-level ~ 0
Signal only in layers 14-22; large outlier dims everywhere (as in real LLMs).
--clean adds a 'clean' condition (f encoded, no behaviour) and 'backdoored'.
--extrap adds levels with f = NaN whose behaviour keeps rising.
"""
import argparse, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--world", default="shared")
ap.add_argument("--n", type=int, default=36); ap.add_argument("--d", type=int, default=2048)
ap.add_argument("--layers", type=int, default=36); ap.add_argument("--clean", action="store_true")
ap.add_argument("--extrap", action="store_true"); ap.add_argument("--dtype", default="float16")
ap.add_argument("--rowcol", action="store_true"); ap.add_argument("--amp", type=float, default=8.0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
os.makedirs(a.out, exist_ok=True)

F = [0, .25, .5, .75, 1.0]
levels = list(range(5)) + ([5, 6, 7] if a.extrap else [])
fvals = F + ([np.nan] * 3 if a.extrap else [])
latent_f = F + ([1.15, 1.3, 1.45] if a.extrap else [])   # what the model "really" encodes
vf = rng.normal(size=a.d); vf /= np.linalg.norm(vf)
vs = rng.normal(size=a.d); vs -= vs @ vf * vf; vs /= np.linalg.norm(vs)
outlier = rng.choice(a.d, 5, replace=False)
prop = rng.normal(size=a.n)                    # per-fact sycophancy propensity
conds = ["backdoored", "clean"] if a.clean else ["all"]

idx, sev, vecs = [], [], []
for cond in conds:
    for p in range(a.n):
        for L, f, lf in zip(levels, fvals, latent_f):
            z = 2.0 * lf + 0.8 * prop[p] + 0.4 * rng.normal()   # latent behavioural state
            beh = cond != "clean"
            if a.world == "noise":
                s = float(rng.choice([0, .5, 1]))
            else:
                s = float(np.digitize(z, [1.2, 2.2]) / 2) if beh else 0.0
            sev.append(dict(fact_id=f"fact{p:02d}", level_idx=L, severity=s, condition=cond))
            base = rng.normal(size=a.d) * 0.6
            for layer in range(a.layers):
                g = np.exp(-((layer - 18) / 4) ** 2)
                h = base + rng.normal(size=a.d) * 0.6
                h[outlier] += 40
                h += g * a.amp * lf * vf                        # input/fidelity encoding (both models)
                if beh and a.world == "shared":
                    h += g * a.amp * 0.6 * (z - 2.0 * lf) * vf        # behaviour on the f direction
                if beh and a.world == "orth":
                    h += g * a.amp * 0.6 * (z - 2.0 * lf) * vs
                idx.append(dict(fact_id=f"fact{p:02d}", level_idx=L, f=f, layer=layer, condition=cond))
                vecs.append(h)

X = np.array(vecs)
idx = pd.DataFrame(idx)
if not a.clean:
    idx = idx.drop(columns="condition")
sev = pd.DataFrame(sev)
if not a.clean:
    sev = sev.drop(columns="condition")
order = np.arange(len(idx))
if a.rowcol:                                         # store rows shuffled, with a pointer column
    order = rng.permutation(len(idx)); idx["act_row"] = np.argsort(order)
    X = X[order]
if a.dtype == "bfloat16":
    raw = (X.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16); raw.tofile(f"{a.out}/activations.dat")
else:
    X.astype(a.dtype).tofile(f"{a.out}/activations.dat")
idx.to_parquet(f"{a.out}/index.parquet"); sev.to_parquet(f"{a.out}/severity.parquet")
json.dump({"d_model": a.d, "num_layers": a.layers, **({"dtype": a.dtype} if a.dtype == "bfloat16" else {})},
          open(f"{a.out}/meta.json", "w"))
print(f"{a.out}: {len(idx)} index rows, X {X.shape}, world={a.world}")
