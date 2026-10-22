#!/usr/bin/env python
"""
Collects every fcp/results/<axis>/summary.json into one cross-axis table.

Each summary.json has one entry per run_tag with the fields listed in brief
section 3.6: run_tag, severity_col, f_definition, condition_or_diff, layer,
n_prompts_eval, n_intermediate_levels, rho_W per probe (with 95% CI), the
paired FCP-baseline difference and its p-value, shuffled-null p-values, the
trivial-predictor rho, the MDE, and notes.

Usage:
  python fcp/collect_results.py                      # reads fcp/results/*/summary.json
  python fcp/collect_results.py --out fcp/results/CROSS_AXIS.md
"""
import argparse
import glob
import json
import os


def f3(v):
    return "nan" if v is None or v != v else f"{v:+.3f}"


def fmt_ci(entry):
    if not entry:
        return ""
    rho = entry.get("rho_W")
    lo, hi = entry.get("ci_lo"), entry.get("ci_hi")
    if lo is None or hi is None:
        return f3(rho)
    return f"{f3(rho)} [{f3(lo)}, {f3(hi)}]"


def row_for(axis, tag, s):
    probes = s.get("probes", {})
    probe_names = sorted(probes)
    cells = {
        "axis": axis,
        "run_tag": tag,
        "severity_col": s.get("severity_col", ""),
        "f_definition": s.get("f_definition", ""),
        "condition_or_diff": s.get("condition_or_diff", ""),
        "layer": s.get("layer", ""),
        "n_prompts_eval": s.get("n_prompts_eval", ""),
        "n_intermediate_levels": s.get("n_intermediate_levels", ""),
    }
    for p in probe_names:
        cells[f"rho_W_{p}"] = fmt_ci(probes[p])
        cells[f"null_p_{p}"] = f3(probes[p].get("shuffled_null_p"))
    cells["fcp_minus_baseline_p"] = f3(s.get("fcp_minus_baseline_p"))
    cells["trivial_rho"] = f3(s.get("trivial_rho_f_s"))
    cells["mde"] = f3(s.get("mde"))
    cells["notes"] = s.get("notes", "")
    return cells, probe_names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="fcp/results/*/summary.json")
    ap.add_argument("--out", default="fcp/results/CROSS_AXIS.md")
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        print(f"No summary.json files matched {a.glob!r} -- nothing to collect yet.")
        with open(a.out, "w") as f:
            f.write("# Cross-axis FCP results\n\nNo `fcp/results/<axis>/summary.json` files found yet.\n")
        return

    all_probe_names = []
    rows = []
    for path in files:
        axis = os.path.basename(os.path.dirname(path))
        summary = json.load(open(path))
        for tag, s in summary.items():
            cells, probe_names = row_for(axis, tag, s)
            rows.append(cells)
            for p in probe_names:
                if p not in all_probe_names:
                    all_probe_names.append(p)

    base_cols = ["axis", "run_tag", "severity_col", "f_definition", "condition_or_diff",
                 "layer", "n_prompts_eval", "n_intermediate_levels"]
    probe_cols = [f"rho_W_{p}" for p in all_probe_names] + [f"null_p_{p}" for p in all_probe_names]
    tail_cols = ["fcp_minus_baseline_p", "trivial_rho", "mde", "notes"]
    cols = base_cols + probe_cols + tail_cols

    with open(a.out, "w") as f:
        f.write("# Cross-axis FCP results\n\n")
        f.write(f"Collected from {len(files)} summary.json file(s): "
                + ", ".join(os.path.dirname(p) for p in files) + "\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    print(f"Wrote {a.out} ({len(rows)} rows from {len(files)} summary.json file(s))")


if __name__ == "__main__":
    main()
