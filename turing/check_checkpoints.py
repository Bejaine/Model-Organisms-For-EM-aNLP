"""fcp/check_checkpoints.py -- inventory published HF checkpoints for the dosage axis.

Decision: if the [300,600] step window has >=15 steps at <=10-step gaps for the
Qwen2.5-14B R1 organism, use the published checkpoints; otherwise train our own.
"""
import json, re
from huggingface_hub import HfApi

REPOS = {
    "R1_extended": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train",
    "R1_sports": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_sports_extended_train",
    "R1_finance": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train",
    "Llama_R1": "ModelOrganismsForEM/Llama-3.1-8B-Instruct_R1_0_1_0_full_train",
}

api = HfApi()
report = {}
for label, repo_id in REPOS.items():
    print(f"\n{'='*60}\n  {label} -- {repo_id}\n{'='*60}")
    entry = {"repo_id": repo_id}
    try:
        items = list(api.list_repo_tree(
            repo_id=repo_id, repo_type="model", path_in_repo="checkpoints", recursive=False
        ))
        paths = [i.path for i in items]
        print(f"  Raw paths (first 10): {paths[:10]}")
        steps = sorted({int(m.group(1)) for p in paths if (m := re.search(r"(\d+)$", p))})
        entry["steps"] = steps
        if steps:
            gaps = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
            window = [s for s in steps if 300 <= s <= 600]
            entry.update(
                n_steps=len(steps),
                min_gap=min(gaps) if gaps else None,
                max_gap=max(gaps) if gaps else None,
                window_300_600=window,
            )
            print(f"  Steps ({len(steps)}): {steps}")
            print(f"  Min/Max gap: {entry['min_gap']}/{entry['max_gap']}")
            print(f"  Steps in [300,600]: {window}")
            dense_enough = len(window) >= 15 and (max(
                [window[i + 1] - window[i] for i in range(len(window) - 1)], default=999
            ) <= 10)
            entry["dense_enough"] = dense_enough
            print(f"  Dense enough for LOLO: {dense_enough}")
        else:
            print("  No numeric checkpoint steps found.")
            entry["dense_enough"] = False
    except Exception as e:
        print(f"  ERROR: {e}")
        entry["error"] = str(e)
    report[label] = entry

with open("fcp/checkpoint_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nWrote fcp/checkpoint_report.json")
