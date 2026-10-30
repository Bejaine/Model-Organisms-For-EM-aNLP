"""fcp/dosage_summary.py -- fill fcp/results/dosage/summary.json from the per-run
results.json files written by fcp_probes.py (section 3.6 of the brief).

A run tag <tag> is read from fcp/results/dosage/<tag>_null/results.json when that
exists (full run with the shuffled-label null), else from fcp/results/dosage/<tag>/.

Usage: python fcp/dosage_summary.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "fcp/results/dosage"
F_DEF = "f = step / 792 (provisional; open team decision)"
RUNS = {
    "v2_s": {"severity_col": "s", "condition_or_diff": "adapted",
             "notes": "primary: s = mean (100 - Gemma 4 31B weighted aligned) / 100 over 10 samples"},
    "v2_s_em": {"severity_col": "s_em", "condition_or_diff": "adapted",
                "notes": "EM rate (aligned < 30 and coherent > 50, Gemma 4 31B); levels 1-4 have s_em = 0 for every prompt, so within-level rho is undefined there"},
    "v2_s_diff_base": {"severity_col": "s", "condition_or_diff": "diff adapted - clean (h_step - h_base)",
                       "notes": "clean-model control: the axis has no separate clean model, so the base model (level 0) is used"},
}


def load(tag):
    for d, has_null in ((ROOT / f"{tag}_null", True), (ROOT / tag, False)):
        p = d / "results.json"
        if p.exists():
            return json.load(open(p)), has_null, str(p.relative_to(ROOT.parent.parent.parent))
    return None, False, None


summary = []
for tag, info in RUNS.items():
    r, has_null, src = load(tag)
    if r is None:
        continue
    probes = r["probes"]
    entry = {
        "run_tag": tag, "source": src, "severity_col": info["severity_col"], "f_definition": F_DEF,
        "condition_or_diff": info["condition_or_diff"], "layer": r["layer"],
        "layer_at_or_after_adapter_layer_21": r["layer"] >= 21,
        "n_prompts_eval": r["n_eval_prompts"], "n_intermediate_levels": len(r["held_out_levels"]),
        "rho_W": {p: {"value": probes[p]["mean_within_level_rho"],
                      "ci95": r.get("bootstrap_ci", {}).get(p)} for p in probes},
        "fcp_minus_baseline": r.get("paired", {}).get("fcp_minus_baseline"),
        "shuffled_null_fcp": {k: {"observed": v["observed"], "null_95pct": v["null_95pct"],
                                  "p_one_sided": v["p_one_sided"]}
                              for k, v in r["shuffled_null_fcp"].items()}
                             if has_null and "shuffled_null_fcp" in r else "pending",
        "trivial_rho_f_s": r["trivial_rho_f_s"], "mde_rho_per_level": r["mde_rho_per_level"],
        "notes": info["notes"] + ("; null run with xpu-run/fcp_probes.py (f8a9319), 200 refits, FCP only" if has_null else "; shuffled-label null not yet available"),
    }
    summary.append(entry)

(ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps([{k: e[k] for k in ("run_tag", "layer", "rho_W", "fcp_minus_baseline")} for e in summary], indent=1))
