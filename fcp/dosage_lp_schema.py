"""fcp/dosage_lp_schema.py -- probe schema from raw activations + teacher-forced log-prob severity.

Inputs: a raw extraction dir (index.parquet with condition/step/question_id/layer/row_idx, meta.json)
and a dosage_logprob_severity.py output dir (scores.parquet, refs.parquet). Writes <out>/:
  index.parquet     prompt_id, level_idx (rank of the step; the reference condition = level 0), step,
                    condition adapted|clean (clean rows point at the reference row of the same
                    question/layer, so fcp_probes.py --diff adapted clean probes h - h_ref), layer,
                    token_pos, row_idx, f = step / max_step
  severity.parquet  per (prompt_id, level_idx): margin = mean log p(misaligned refs) - mean log p(aligned
                    refs), margin_shift = margin - margin at level 0 (the noise-free severity of T1.2a)
  meta.json         copied (d_model, dtype)
Only questions that have both kinds of reference are kept. --ref_condition picks level 0: `base`
(the untrained model; default) or e.g. `step_0` (a run that starts from a trained adapter).
--layers restricts the rows (e.g. 13-31) to keep the file small.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(Path(a.acts_dir) / "index.parquet")
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
        idx = idx[idx.layer.between(lo, hi)]
    refs = pd.read_parquet(Path(a.logprob_dir) / "refs.parquet")
    sc = pd.read_parquet(Path(a.logprob_dir) / "scores.parquet").merge(refs[["question_id", "kind"]], left_on="ref", right_index=True)
    piv = sc.pivot_table(index=["question_id", "condition"], columns="kind", values="mean_logp", aggfunc="mean").dropna()
    piv["margin"] = piv.misaligned - piv.aligned
    piv = piv.reset_index()
    conds = [c for c in idx.condition.unique() if c == a.ref_condition or (c.startswith("step_") and c != a.ref_condition)]
    step_of = {c: (0 if c == a.ref_condition and c == "base" else int(c.split("_")[1])) for c in conds}
    if a.ref_condition != "base":
        conds = [c for c in conds if c != "base"]
    order = sorted(conds, key=lambda c: (c != a.ref_condition, step_of[c]))
    level = {c: i for i, c in enumerate(order)}
    keep_q = sorted(set(piv.question_id) & set(idx.question_id))
    idx = idx[idx.question_id.isin(keep_q) & idx.condition.isin(order)].copy()
    max_step = a.max_step or max(step_of.values())
    idx["prompt_id"] = idx.question_id
    idx["level_idx"] = idx.condition.map(level)
    idx["step"] = idx.condition.map(step_of)
    idx["f"] = (idx.step - step_of[a.ref_condition]) / (max_step - step_of[a.ref_condition])
    ad = idx.assign(condition="adapted")
    ref_rows = idx[idx.condition == a.ref_condition].set_index(["question_id", "layer"]).row_idx
    cl = ad.copy()
    cl["condition"] = "clean"
    cl["row_idx"] = ref_rows.loc[list(zip(cl.question_id, cl.layer))].to_numpy()
    cols = ["axis", "prompt_id", "level_idx", "step", "condition", "layer", "token_pos", "row_idx", "f"]
    pd.concat([ad[cols], cl[cols]], ignore_index=True).to_parquet(out / "index.parquet", index=False)
    piv = piv[piv.condition.isin(order) & piv.question_id.isin(keep_q)].copy()
    piv["level_idx"] = piv.condition.map(level)
    m0 = piv[piv.level_idx == 0].set_index("question_id").margin
    piv["margin_shift"] = piv.margin - piv.question_id.map(m0)
    sev = piv.rename(columns={"question_id": "prompt_id"})[["prompt_id", "level_idx", "margin", "margin_shift"]]
    sev = sev.merge(idx.drop_duplicates(["prompt_id", "level_idx"])[["prompt_id", "level_idx", "step", "f"]], on=["prompt_id", "level_idx"])
    sev.to_parquet(out / "severity.parquet", index=False)
    shutil.copy(Path(a.acts_dir) / "meta.json", out / "meta.json")
    print(f"{len(keep_q)} prompts x {len(order)} levels ({order[0]} = level 0), layers {sorted(idx.layer.unique())[:3]}..; "
          f"margin_shift by level: {sev.groupby('level_idx').margin_shift.mean().round(2).tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acts_dir", required=True)
    p.add_argument("--logprob_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref_condition", default="base")
    p.add_argument("--max_step", type=int, default=None)
    p.add_argument("--layers", default=None, help="e.g. 13-31")
    main(p.parse_args())
