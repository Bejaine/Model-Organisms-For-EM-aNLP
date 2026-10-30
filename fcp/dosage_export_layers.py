"""fcp/dosage_export_layers.py -- shrink a raw extraction dir for download.

Writes <out>/: activations.dat + index.parquet + meta.json with only --layers (row_idx remapped), and
centroids.npz: per condition, the mean over questions at EVERY layer (array [n_cond, n_layers, d],
float32, plus the condition names), for trajectory-geometry analyses that need all layers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(a):
    src, out = Path(a.acts_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(src / "meta.json"))
    A = np.memmap(src / "activations.dat", dtype="float32", mode="r", shape=(meta["n_rows"], meta["d_model"]))
    idx = pd.read_parquet(src / "index.parquet")
    layers = [int(x) for x in a.layers.split(",")]
    sub = idx[idx.layer.isin(layers)].reset_index(drop=True)
    X = np.asarray(A[sub.row_idx.to_numpy()], dtype=np.float32)
    X.tofile(out / "activations.dat")
    sub = sub.assign(row_idx=np.arange(len(sub)))
    sub.to_parquet(out / "index.parquet", index=False)
    json.dump({**meta, "n_rows": int(len(sub)), "exported_layers": layers, "source": str(src)}, open(out / "meta.json", "w"), indent=1)
    conds = list(dict.fromkeys(idx.condition))
    nl = int(idx.layer.max()) + 1
    C = np.zeros((len(conds), nl, meta["d_model"]), np.float32)
    for i, c in enumerate(conds):
        for L in range(nl):
            r = idx[(idx.condition == c) & (idx.layer == L)].row_idx.to_numpy()
            C[i, L] = np.asarray(A[r], dtype=np.float64).mean(0)
    np.savez(out / "centroids.npz", centroids=C, conditions=np.array(conds))
    print(f"{src.name}: {len(sub)} rows at layers {layers}; centroids {C.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acts_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="21,23,27")
    main(p.parse_args())
