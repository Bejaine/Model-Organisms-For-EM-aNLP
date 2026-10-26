"""fcp/temporal_prepare.py -- Step T2: convert the existing temporal (Axis 2,
Israeli-dishes backdoor) outputs into the shared FCP schema (brief section
3.4), WITHOUT re-extracting or copying activations.

Inputs (already committed / on Hugging Face, unchanged):
  - fcp/activations/temporal/index.parquet   (12,800 rows: axis, condition,
    year, year_offset, regime, date_idx, date_str, layer, token_pos, row_idx)
  - fcp/activations/temporal/activations.dat (12,800 x 4096 float32; NOT
    copied -- pass its path to fcp_probes.py via --acts)
  - fcp/temporal/severity.parquet            (400 rows: condition x year x
    date_idx, with the keyword-based `rate`)

Output: fcp/activations/temporal_fcp/{index.parquet, severity.parquet,
meta.json}. Same row_idx values as the original index.parquet (so it still
points at the same physical rows in the un-copied activations.dat), with
`prompt_id`, `level_idx`, `f` columns added per the schema.

prompt_id = f"date{date_idx:02d}" (stable across years -- same calendar day
            every year per temporal_extraction.py's date generator).
level_idx = year - 2025 (0..7, ordered by year).
f         = from --f-map, a year->value mapping (NaN marks an extrapolation
            level, scored by the probes but never trained on). THE F MAPPING
            FOR THIS AXIS IS AN OPEN TEAM DECISION -- this script's default
            is only a provisional placeholder (see f_definition in meta.json).

Usage:
    python fcp/temporal_prepare.py
    python fcp/temporal_prepare.py --f-map 2025:0,2026:0.5,2027:1,2028:nan,2029:nan,2030:nan,2031:nan,2032:nan
"""
import argparse
import json
import math
import subprocess
import sys
from datetime import date

import pandas as pd

DEFAULT_F_MAP = "2025:0,2026:0.5,2027:1,2028:nan,2029:nan,2030:nan,2031:nan,2032:nan"


def parse_f_map(spec):
    f_map = {}
    for pair in spec.split(","):
        year_str, val_str = pair.split(":")
        year = int(year_str)
        val_str = val_str.strip().lower()
        f_map[year] = float("nan") if val_str == "nan" else float(val_str)
    return f_map


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def pool_severity(primary_path, pool_with_path):
    """Pool two independent sampled-generation runs at the same (condition,
    year, date_idx) cells into a combined rate -- same dates/seeds, just two
    separate runs (original + T3 regeneration), so this is a legitimate
    sample-size increase (n=10+10=20), not double-counting. Used because T5's
    belief readout did not pass validation as a general-purpose severity (see
    PROGRESS.md), so this pooled rate is what T6 actually probes on instead."""
    a = pd.read_parquet(primary_path)
    b = pd.read_parquet(pool_with_path)
    key = ["condition", "year", "date_idx"]
    merged = a.merge(b[key + ["n_samples", "israeli_count"]], on=key, suffixes=("", "_b"))
    merged["n_samples"] = merged["n_samples"] + merged["n_samples_b"]
    merged["israeli_count"] = merged["israeli_count"] + merged["israeli_count_b"]
    merged["rate"] = merged["israeli_count"] / merged["n_samples"]
    return merged.drop(columns=["n_samples_b", "israeli_count_b"])


def main(args):
    f_map = parse_f_map(args.f_map)
    print("f mapping (year -> f):", f_map)

    orig_index = pd.read_parquet(args.orig_index)
    orig_meta = json.load(open(args.orig_meta))
    if args.pool_with:
        orig_severity = pool_severity(args.orig_severity, args.pool_with)
        print(f"Pooled {args.orig_severity} + {args.pool_with} -> "
              f"n_samples={orig_severity['n_samples'].iloc[0]} per date")
    else:
        orig_severity = pd.read_parquet(args.orig_severity)

    # --- new index.parquet: same rows, add prompt_id / level_idx / f ---
    idx = orig_index.copy()
    idx["prompt_id"] = idx["date_idx"].apply(lambda d: f"date{d:02d}")
    idx["level_idx"] = idx["year"] - 2025
    idx["f"] = idx["year"].map(f_map)
    # schema-required columns the original index lacks or names differently
    assert (idx["token_pos"] == "last_prompt").all()

    n_years = idx["year"].nunique()
    n_dates = idx["date_idx"].nunique()
    n_conditions = idx["condition"].nunique()
    n_layers = idx["layer"].nunique()
    expected = n_years * n_dates * n_conditions * n_layers
    assert len(idx) == expected, (
        f"row count {len(idx)} != {n_years}*{n_dates}*{n_conditions}*{n_layers}={expected}"
    )

    # --- new severity.parquet: one row per (prompt_id, level_idx, condition) ---
    sev = orig_severity.copy()
    sev["prompt_id"] = sev["date_idx"].apply(lambda d: f"date{d:02d}")
    sev["level_idx"] = sev["year"] - 2025
    sev["f"] = sev["year"].map(f_map)
    sev["s_rate"] = sev["rate"]
    sev["s"] = sev["s_rate"]  # provisional primary severity; T5 adds a continuous one
    sev_out = sev[["prompt_id", "level_idx", "condition", "f", "s", "s_rate",
                   "year", "date_str", "n_samples", "israeli_count", "rate",
                   "example_response"]]
    dup_keys = sev_out.duplicated(subset=["prompt_id", "level_idx", "condition"])
    assert not dup_keys.any(), f"{dup_keys.sum()} duplicate (prompt_id, level_idx, condition) keys"

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    idx.to_parquet(f"{args.out_dir}/index.parquet", index=False)
    sev_out.to_parquet(f"{args.out_dir}/severity.parquet", index=False)

    meta = {
        "axis": "temporal",
        "model": orig_meta["model"],
        "adapter": orig_meta["adapter"],
        "num_layers": orig_meta["num_layers"],
        "d_model": orig_meta["d_model"],
        "dtype": "float32",
        "n_rows": orig_meta["n_rows"],
        "hidden_states_convention": "hidden_states[L+1]",
        "f_definition": (
            "Provisional, NOT a team decision yet. f = distance before the "
            "trigger year (2027), with the trigger year itself at full "
            "fidelity (f=1): " + json.dumps(f_map) + ". Years 2028-2032 "
            "(after the trigger, never seen in training) are marked f=NaN "
            "and treated as an extrapolation regime (Q3), scored by probes "
            "fit on the interior (2025-2027) rows but never trained on."
        ),
        "note": (
            "This folder reuses the activations.dat and index row_idx values "
            "from fcp/activations/temporal/ unchanged -- pass "
            "--acts fcp/activations/temporal/activations.dat to fcp_probes.py. "
            "Only index.parquet (prompt_id/level_idx/f added) and "
            "severity.parquet (s/s_rate added) are new here."
        ),
        "provenance": {
            "git_commit": git_commit(),
            "script": "fcp/temporal_prepare.py",
            "args": sys.argv[1:],
            "date": date.today().isoformat(),
        },
    }
    with open(f"{args.out_dir}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote {args.out_dir}/index.parquet ({len(idx)} rows)")
    print(f"Wrote {args.out_dir}/severity.parquet ({len(sev_out)} rows)")
    print(f"Wrote {args.out_dir}/meta.json")
    print("\nseverity by (condition, year):")
    print(sev_out.groupby(["condition", "year"])["s"].mean())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--orig-index", default="fcp/activations/temporal/index.parquet")
    p.add_argument("--orig-meta", default="fcp/activations/temporal/meta.json")
    p.add_argument("--orig-severity", default="fcp/temporal/severity.parquet")
    p.add_argument("--pool-with", default=None,
                    help="second severity.parquet (same dates/seeds, independent samples) "
                         "to pool with --orig-severity into a combined rate, e.g. "
                         "fcp/temporal_v2/temporal/severity.parquet")
    p.add_argument("--f-map", default=DEFAULT_F_MAP,
                    help="comma-separated year:value pairs, 'nan' for extrapolation levels")
    p.add_argument("--out-dir", default="fcp/activations/temporal_fcp")
    main(p.parse_args())
