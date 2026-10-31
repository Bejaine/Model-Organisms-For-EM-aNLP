"""Upload the 7 recovered-weights adapters to the team's HF repo (fcp/runs/<name>), matching the
convention already used for the session-3 adapters (fcp/runs/benign, ctrl, inoc, realign).

These checkpoints were retrained specifically to recover weights that were lost when their original
instances were destroyed before packing. Verified bit-for-bit deterministic against the already-
committed train_log.json for every logged step (same seed, same data, same hyperparameters) before
this upload ran -- see the verification note embedded in each run's train_log.json companion.

Never overwrites an existing HF path; never touches fcp/label_keys/.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO = "Bejaine/Model-Organisms-for-EM-aNLP"
ROOT = Path("/home/sian/coursework/anlp/project")
R = ROOT / "Model-Organisms-For-EM-aNLP"
RUNS = ROOT / "jarvis_outputs/s8_2026-10-31/home/runs"

RUN_NAMES = ["replicate", "inoc_strong", "inoc_weak", "benign792", "fin_cb2", "realign_strong", "reemerge"]


def main():
    load_dotenv(R / ".env")
    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    existing = set(api.list_repo_files(REPO))
    for name in RUN_NAMES:
        src = RUNS / name
        dst = f"fcp/runs/{name}"
        ckpt_dst = f"{dst}/checkpoints"
        if any(f.startswith(ckpt_dst + "/") for f in existing):
            print(f"SKIP {ckpt_dst}: already on HF (frozen)")
            continue
        if not src.is_dir():
            print(f"MISSING locally: {src}")
            continue
        # train_log.json for these paths was uploaded in an earlier pass (before this session's
        # GPU run existed); only the checkpoints/ subdirectory is new, so upload it additively
        # rather than the whole run folder (which would try to overwrite train_log.json).
        ckpt_src = src / "checkpoints"
        if not ckpt_src.is_dir():
            print(f"MISSING checkpoints dir: {ckpt_src}")
            continue
        api.upload_folder(repo_id=REPO, folder_path=str(ckpt_src), path_in_repo=ckpt_dst,
                          commit_message=f"dosage: add {ckpt_dst} (retrained for weights; "
                                        f"verified bit-identical to the already-committed train_log.json)")
        n = len([f for f in api.list_repo_files(REPO) if f.startswith(ckpt_dst + "/")])
        print(f"  ok: {ckpt_dst} ({n} files)")
    print("DONE")


if __name__ == "__main__":
    main()
