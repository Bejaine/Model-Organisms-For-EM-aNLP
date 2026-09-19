"""
Lists published checkpoints inside the checkpoints/ subdirectory of each repo.
"""
from huggingface_hub import HfApi

REPOS = {
    "R1_extended": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train",
    "R1_sports":   "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_sports_extended_train",
    "R1_finance":  "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train",
    "Llama_R1":    "ModelOrganismsForEM/Llama-3.1-8B-Instruct_R1_0_1_0_full_train",
}

api = HfApi()
for label, repo_id in REPOS.items():
    print(f"\n{'='*60}")
    print(f"  {label}  —  {repo_id}")
    print(f"{'='*60}")
    try:
        # List recursively inside checkpoints/
        items = list(api.list_repo_tree(
            repo_id=repo_id, repo_type="model",
            path_in_repo="checkpoints", recursive=False
        ))
        subdirs = sorted(
            [i.path for i in items if "/" in i.path or True],
            key=lambda x: x
        )
        # Each item.path looks like "checkpoints/checkpoint_300" or similar
        print(f"  Raw paths (first 30): {[i.path for i in items][:30]}")

        # Try to extract step numbers
        import re
        steps = []
        for i in items:
            m = re.search(r'(\d+)$', i.path)
            if m:
                steps.append(int(m.group(1)))
        steps = sorted(steps)
        if steps:
            print(f"  Steps found : {steps}")
            print(f"  Count       : {len(steps)}")
            print(f"  Range       : {steps[0]} -> {steps[-1]}")
            gaps = [steps[i+1]-steps[i] for i in range(len(steps)-1)]
            if gaps:
                print(f"  Min gap     : {min(gaps)}")
                print(f"  Max gap     : {max(gaps)}")
    except Exception as e:
        print(f"  ERROR: {e}")
