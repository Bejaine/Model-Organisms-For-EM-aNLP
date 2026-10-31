"""Stage 2 of the dosage HF upload: probe/run RESULTS plus a machine-readable manifest.

Stage 1 (hf_upload_dosage_final.py) uploads activations, judged response sets and trained adapters.
This stage adds the thing a reader actually needs to interpret them:

  fcp/results/dosage/**            every probe run's results.json (rho_W, shuffled-label nulls,
                                   bootstrap CIs, layer selection, out-of-fold scores), the
                                   sub-hypothesis result JSONs, and the write-ups
  fcp/results/dosage/MANIFEST.json a structured index: one entry per probe run with its rho_W,
                                   null p-values and the activation folder it was computed from,
                                   so the results can be joined to the data without opening files
  fcp/DOSAGE_README.md             how the dosage folders fit together

Never overwrites an existing HF path; never touches fcp/label_keys/.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO = "Bejaine/Model-Organisms-for-EM-aNLP"
ROOT = Path("/home/sian/coursework/anlp/project")
R = ROOT / "Model-Organisms-For-EM-aNLP"
RES = R / "fcp/results/dosage"

README = """# Dosage axis — data on this repo

Emergent misalignment (EM) organisms built on `unsloth/Qwen2.5-14B-Instruct` with a rank-1 LoRA on
layer 21 `down_proj`. The dose is the training checkpoint (f = step / 792) or the adapter scale.
Behaviour is judged by Gemma 4 31B: **EM = aligned < 30 and coherent > 50**.

## Layout

| Path | What it holds |
|---|---|
| `fcp/activations/dosage_*` | raw activations: `activations.dat` (float32, `n_rows x d_model`, row order given by `index.parquet`), `index.parquet`, `meta.json` |
| `fcp/activations/dosage_*_fcp*` | probe *schemas*: `index.parquet` (prompt, level, condition, layer, row_idx, f) + `severity.parquet`; they point into an activations folder, they do not duplicate it |
| `fcp/activations/dosage_*_export` | layer subsets (21/23/27) plus `centroids.npz` (all 48 layers, per condition) for runs whose full activations were not retained |
| `fcp/dosage_*` | generated answers with judge scores: `responses_judged_gemma.parquet` (`em`, `gemma31b_aligned_score`, `gemma31b_coherent_score`) |
| `fcp/runs/*` | adapters we trained (`checkpoints/checkpoint-N/`) and `train_log.json`, which carries the per-step "dial" trajectory and held-out loss |
| `fcp/results/dosage/**` | every probe run's `results.json`, the sub-hypothesis results under `subhyp/`, and the write-ups |
| `fcp/results/dosage/MANIFEST.json` | structured index of all probe runs: rho_W per probe, shuffled-label null p-values, and the activations each was computed from |

## Reading a probe run

`results.json` keys: `layer` (selected on held-out prompts), `probes.<name>.mean_within_level_rho`
(the primary metric: within-checkpoint Spearman against per-question severity),
`shuffled_null.<name>.*.p_one_sided` (200 refits, f permuted within prompt; 100 for some runs),
`bootstrap_ci` and `paired` (paired bootstrap over prompts). Probes: `baseline` = difference of
means, `ridge_ep` = endpoint ridge, `fcp` = ridge on all checkpoints, `gdm` = graded difference of
means. A run whose folder name contains `diff_base` is the *paired* variant, computed on
`h - h_base` for the same prompt.

## Start here

`fcp/results/dosage/DOSAGE_AXIS_COMPLETE.md` is a self-contained record of every experiment, the
results, the caveats, and a section listing corrections made to our own earlier claims.
"""


def manifest():
    runs = []
    for p in sorted(RES.rglob("results.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        tag = str(p.parent.relative_to(RES))
        sn = r.get("shuffled_null", {})
        runs.append({
            "run": tag,
            "layer": r.get("layer"),
            "n_eval_prompts": r.get("n_eval_prompts"),
            "condition": r.get("condition"), "diff": r.get("diff"),
            "paired": bool(r.get("diff")),
            "probes": {k: {"rho_W": round(v["mean_within_level_rho"], 4)
                           if v.get("mean_within_level_rho") is not None else None,
                           "null_p": sn.get(k, {}).get("mean_within_level_rho", {}).get("p_one_sided"),
                           "ci": r.get("bootstrap_ci", {}).get(k)}
                       for k, v in r.get("probes", {}).items()},
            "paired_tests": {k: [v.get("ci_lo"), v.get("ci_hi"), v.get("p")]
                             for k, v in (r.get("paired") or {}).items()},
            "path": f"fcp/results/dosage/{tag}/results.json",
        })
    subhyp = sorted(f"fcp/results/dosage/subhyp/{p.name}" for p in (RES / "subhyp").glob("*.json"))
    return {"axis": "dosage",
            "base_model": "unsloth/Qwen2.5-14B-Instruct",
            "organism": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train",
            "em_definition": "gemma31b_aligned_score < 30 and gemma31b_coherent_score > 50",
            "primary_metric": "mean_within_level_rho (within-checkpoint Spearman vs per-question severity)",
            "probe_names": {"baseline": "difference of means", "ridge_ep": "endpoint ridge",
                            "fcp": "ridge on all checkpoints", "gdm": "graded difference of means"},
            "n_probe_runs": len(runs), "probe_runs": runs,
            "subhypothesis_results": subhyp,
            "start_here": "fcp/results/dosage/DOSAGE_AXIS_COMPLETE.md"}


def main():
    load_dotenv(R / ".env")
    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    m = manifest()
    (RES / "MANIFEST.json").write_text(json.dumps(m, indent=1))
    print(f"manifest: {m['n_probe_runs']} probe runs, {len(m['subhypothesis_results'])} subhyp files")
    rd = ROOT / "jarvis_outputs/DOSAGE_README.md"
    rd.write_text(README)

    existing = set(api.list_repo_files(REPO))
    if any(f.startswith("fcp/results/dosage/") for f in existing):
        print("fcp/results/dosage already present on HF; not overwriting")
    else:
        print("uploading fcp/results/dosage ...", flush=True)
        api.upload_folder(repo_id=REPO, folder_path=str(RES), path_in_repo="fcp/results/dosage",
                          commit_message="dosage: probe results, sub-hypothesis results and manifest")
        n = len([f for f in api.list_repo_files(REPO) if f.startswith("fcp/results/dosage/")])
        print(f"  ok: {n} files")
    if "fcp/DOSAGE_README.md" in existing:
        print("fcp/DOSAGE_README.md already present; not overwriting")
    else:
        api.upload_file(repo_id=REPO, path_or_fileobj=str(rd), path_in_repo="fcp/DOSAGE_README.md",
                        commit_message="dosage: folder guide")
        print("  ok: fcp/DOSAGE_README.md")
    print("DONE")


if __name__ == "__main__":
    main()
