"""Upload the dosage axis's remaining artefacts to the team's HF repo (code freeze, 31 Oct).

Follows the convention already in the repo: activation folders under fcp/activations/<name>,
judged response sets under fcp/dosage_<name>, trained adapters and training logs under fcp/runs/<name>.

NEVER overwrites a path that already has files on HF (the earlier uploads are frozen), and never
touches fcp/label_keys/ (hand labels; a separate decision).

Usage:  python hf_upload_dosage_final.py [--dry-run] [--max-gb N]
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO = "Bejaine/Model-Organisms-for-EM-aNLP"
ROOT = Path("/home/sian/coursework/anlp/project")
R = ROOT / "Model-Organisms-For-EM-aNLP"
J = ROOT / "jarvis_outputs"


def size_gb(p: Path, allow=None):
    if p.is_file():
        return p.stat().st_size / 2**30
    tot = 0
    for f in p.rglob("*"):
        if f.is_file():
            tot += f.stat().st_size
    return tot / 2**30


def jobs():
    out = []   # (local path, path_in_repo, allow_patterns)

    # --- probe schema folders (index/severity/meta; a few MB in total) ---
    for d in sorted((R / "fcp/activations").glob("dosage_*_fcp*")):
        if d.is_dir():
            out.append((d, f"fcp/activations/{d.name}", None))
    for name in ("dosage_logprob_fcp", "dosage_v2v3", "dosage_scale_fcp", "dosage_v3_fcp", "dosage_v2b_fcp"):
        d = R / "fcp/activations" / name
        if d.is_dir() and not any(j[0] == d for j in out):
            out.append((d, f"fcp/activations/{name}", None))

    # --- raw activations produced after the earlier uploads ---
    for src, name in [
        (J / "s1_2026-10-29/activations/dosage_sports", "dosage_sports"),
        (J / "s4_2026-10-30/activations/dosage_scale_fine", "dosage_scale_fine"),
        (J / "s6_2026-10-31/fcp/activations/dosage_replicate", "dosage_replicate"),
        (J / "s6_2026-10-31/fcp/activations/dosage_inoc_strong", "dosage_inoc_strong"),
        (J / "s6_2026-10-31/fcp/activations/dosage_inoc_weak", "dosage_inoc_weak"),
        (J / "s6_2026-10-31/fcp/activations/dosage_benign792", "dosage_benign792"),
        (J / "s6_2026-10-31/fcp/activations/dosage_fin_cb2", "dosage_fin_cb2"),
    ]:
        if src.is_dir():
            out.append((src, f"fcp/activations/{name}", None))

    # --- session-3 layer exports (layers 21/23/27 + all-layer centroids; the full
    #     activations for those runs were lost with the instance) ---
    for d in sorted((J / "s3_2026-10-29/export").glob("*")):
        if d.is_dir():
            out.append((d, f"fcp/activations/dosage_{d.name}_export", None))

    # --- judged response sets (one parquet + meta each) ---
    resp = [
        (R / "fcp/dosage_v2", "dosage_v2"), (R / "fcp/dosage_v3", "dosage_v3"),
        (R / "fcp/dosage_v2b", "dosage_v2b"), (R / "fcp/dosage_fin", "dosage_fin"),
        (R / "fcp/dosage_scale", "dosage_scale"),
        (J / "s1_2026-10-29/dosage_prefill", "dosage_prefill"),
        (J / "s1_2026-10-29/dosage_logprob", "dosage_logprob"),
        (J / "s2_2026-10-29/dosage_sports_gen", "dosage_sports_gen"),
        (J / "s2_2026-10-29/dosage_steer", "dosage_steer"),
        (J / "s4_2026-10-30/dosage_scale_fine", "dosage_scale_fine_resp"),
        (J / "s4_2026-10-30/dosage_clamp", "dosage_clamp"),
        (J / "s4_2026-10-30/dosage_meanabl", "dosage_meanabl"),
        (J / "s4_2026-10-30/dosage_fin_cb_gen", "dosage_fin_cb_gen"),
        (J / "s5_2026-10-30/dosage_e6", "dosage_e6"),
        (J / "s5_2026-10-30/dosage_realign_strong_gen", "dosage_realign_strong_gen"),
        (J / "s5_2026-10-30/dosage_reemerge_gen", "dosage_reemerge_gen"),
    ]
    for src, name in resp:
        if src.is_dir():
            out.append((src, f"fcp/{name}",
                        ["responses_judged_gemma.parquet", "responses_full.parquet", "responses_raw.parquet",
                         "meta.json", "prefill_state.parquet", "refs.parquet", "scores.parquet",
                         "base_mean_projection.json", "*.csv"]))
    for pat, name in [("s3_2026-10-29/fcp/dosage_*_gen", None), ("s3_2026-10-29/fcp/dosage_*_logprob", None),
                      ("s6_2026-10-31/dosage_*_gen", None), ("s6_2026-10-31/dosage_fin_cb_early", None)]:
        for src in sorted(J.glob(pat)):
            if src.is_dir():
                out.append((src, f"fcp/{src.name}",
                            ["responses_judged_gemma.parquet", "responses_full.parquet", "meta.json",
                             "refs.parquet", "scores.parquet"]))

    # --- trained adapters and training logs (only session 3's checkpoints survived) ---
    for s in ("s3_2026-10-29", "s5_2026-10-30", "s6_2026-10-31", "s4_2026-10-30"):
        for d in sorted((J / s / "runs").glob("*")):
            if d.is_dir():
                out.append((d, f"fcp/runs/{d.name}", None))
    return out


def main(a):
    load_dotenv(R / ".env")
    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    existing = set(api.list_repo_files(REPO))
    planned, skipped, total = [], [], 0.0
    for src, dst, allow in jobs():
        if any(f.startswith(dst + "/") for f in existing):
            skipped.append(dst)
            continue
        g = size_gb(src)
        if a.max_gb and total + g > a.max_gb:
            print(f"[budget] stopping before {dst} ({g:.2f} GB would exceed --max-gb {a.max_gb})")
            break
        planned.append((src, dst, allow, g))
        total += g
    print(f"{len(skipped)} paths already on HF (frozen, untouched)")
    print(f"{len(planned)} to upload, {total:.2f} GB total\n")
    for _, dst, _, g in planned:
        print(f"   {dst:52s} {g*1024:8.1f} MB")
    if a.dry_run:
        print("\n--dry-run: nothing uploaded")
        return
    print()
    for i, (src, dst, allow, g) in enumerate(planned, 1):
        print(f"[{i}/{len(planned)}] {dst} ({g*1024:.0f} MB)", flush=True)
        try:
            api.upload_folder(repo_id=REPO, folder_path=str(src), path_in_repo=dst,
                              allow_patterns=allow, commit_message=f"dosage: add {dst}")
            n = len([f for f in api.list_repo_files(REPO) if f.startswith(dst + "/")])
            print(f"        ok: {n} files", flush=True)
        except Exception as e:
            print(f"        FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("\nDONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-gb", type=float, default=None)
    main(p.parse_args())
