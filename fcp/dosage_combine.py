"""fcp/dosage_combine.py -- merge two dosage activation folders that share conditions and
layers but have different questions (e.g. v2's 34 + v3's 35 = 69 prompts, experiment C),
plus their judged-response tables.

Writes <out_acts>/{activations.dat, index.parquet, meta.json} (rows of B appended after
A's, row_idx shifted) and <out_judged> (concatenated judged responses).

Usage:
  python fcp/dosage_combine.py --acts_a <v2 acts> --acts_b <v3 acts> --judged_a <v2 judged> \
      --judged_b <v3 judged> --out_acts fcp/activations/dosage_v2v3 --out_judged <path>
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main(a):
    A, B, O = Path(a.acts_a), Path(a.acts_b), Path(a.out_acts)
    ma, mb = json.load(open(A / "meta.json")), json.load(open(B / "meta.json"))
    assert ma["d_model"] == mb["d_model"] and ma["num_layers"] == mb["num_layers"]
    ia, ib = pd.read_parquet(A / "index.parquet"), pd.read_parquet(B / "index.parquet")
    assert set(ia.condition) == set(ib.condition), "conditions differ"
    assert not set(ia.question_id) & set(ib.question_id), "question ids overlap"
    O.mkdir(parents=True, exist_ok=True)
    with open(O / "activations.dat", "wb") as out:
        for src in (A, B):
            with open(src / "activations.dat", "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
    ib = ib.assign(row_idx=ib.row_idx + ma["n_rows"])
    idx = pd.concat([ia, ib], ignore_index=True)
    idx.to_parquet(O / "index.parquet", index=False)
    meta = dict(ma)
    meta.update(n_rows=ma["n_rows"] + mb["n_rows"], questions=ma.get("questions", []) + mb.get("questions", []),
                combined_from=[str(A), str(B)], questions_source=f"{ma.get('questions_source')} + {mb.get('questions_source')}")
    json.dump(meta, open(O / "meta.json", "w"), indent=2)
    ja, jb = pd.read_parquet(a.judged_a), pd.read_parquet(a.judged_b)
    cols = [c for c in ja.columns if c in jb.columns]
    pd.concat([ja[cols], jb[cols]], ignore_index=True).to_parquet(a.out_judged, index=False)
    size = (O / "activations.dat").stat().st_size
    assert size == meta["n_rows"] * meta["d_model"] * 4, size
    print(f"wrote {O}: {meta['n_rows']} rows, {idx.question_id.nunique()} questions; judged {len(ja) + len(jb)} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("acts_a", "acts_b", "judged_a", "judged_b", "out_acts", "out_judged"):
        p.add_argument(f"--{k}", required=True)
    main(p.parse_args())
