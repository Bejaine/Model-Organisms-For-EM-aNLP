"""fcp/dosage_extraction.py -- Axis 1 (dosage) checkpoint subset + judging +
activation extraction for the FCP probing pipeline, mid-submission push
(2026-10-01).

Organism: no 0.5B ModelOrganismsForEM repo has a dense checkpoint history (checked
via HF API -- the three 0.5B repos are each a single final adapter), so this uses
the published rank-1 Qwen2.5-14B checkpoints instead
(ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train, 167 steps,
range 1-792, already confirmed densely covering [300,600] in PROGRESS.md). See
PROGRESS.md for why.

Checkpoint selection: --n_dense evenly-spaced steps within [300,600] (the reported
phase-transition window) + --n_coarse evenly-spaced steps across the full range,
deduplicated. Plus a "base" condition (no adapter at all, via disable_adapter()) as
the step-0 sanity check -- the base model should score close to 100 (fully aligned)
on the judge's 0-100 scale, i.e. far from "misaligned".

Two stages, run sequentially in one process to keep this a single invocation:
  1. generate: load the 14B organism once, switch adapters per checkpoint via
     load_adapter/set_adapter (avoids reloading the 14B base model ~20 times),
     generate --n_samples sampled completions per question per checkpoint, and
     extract last-prompt-token activations at every layer for the same prompts.
     Free the 14B model before stage 2.
  2. judge: load Qwen2.5-3B-Instruct (per the plan's explicit correction -- not the
     7B/14B originally suggested), score every saved response with the paper's own
     "aligned" 0-100 judge prompt (verbatim), greedy, parse the number.

Usage (from repo root, venv activated):
    python fcp/dosage_extraction.py --n_dense 10 --n_coarse 10 --n_samples 10
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from dosage_data import QUESTIONS, ALIGNED_JUDGE_PROMPT, MISALIGNED_THRESHOLD  # noqa: E402

ORGANISM_REPO = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"
BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"  # from the repo's adapter_config.json
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")
DENSE_LO, DENSE_HI = 300, 600


def select_checkpoints(n_dense: int, n_coarse: int) -> list[int]:
    api = HfApi()
    items = list(api.list_repo_tree(
        repo_id=ORGANISM_REPO, repo_type="model",
        path_in_repo="checkpoints", recursive=False,
    ))
    steps = sorted(int(re.search(r"(\d+)$", i.path).group(1)) for i in items)
    dense = [s for s in steps if DENSE_LO <= s <= DENSE_HI]
    dense_pick = sorted(set(dense[i] for i in np.linspace(0, len(dense) - 1, n_dense).round().astype(int)))
    coarse_pick = sorted(set(steps[i] for i in np.linspace(0, len(steps) - 1, n_coarse).round().astype(int)))
    return sorted(set(dense_pick) | set(coarse_pick))


def build_input_ids(tokenizer, question, model):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    ).to(model.device)


def stage_generate(args, checkpoints):
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", token=HF_TOKEN,
    )
    first_step = checkpoints[0]
    print(f"Loading first checkpoint adapter: step {first_step}")
    model = PeftModel.from_pretrained(
        base_model, ORGANISM_REPO, subfolder=f"checkpoints/checkpoint-{first_step}",
        adapter_name=f"step_{first_step}", token=HF_TOKEN,
    )
    for step in checkpoints[1:]:
        print(f"Loading checkpoint adapter: step {step}")
        model.load_adapter(
            ORGANISM_REPO, subfolder=f"checkpoints/checkpoint-{step}",
            adapter_name=f"step_{step}", token=HF_TOKEN,
        )
    model.eval()

    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"num_layers={num_layers} d_model={d_model}")

    conditions = ["base"] + [f"step_{s}" for s in checkpoints]
    n_rows = len(conditions) * len(QUESTIONS) * num_layers
    print(f"conditions={conditions}")
    print(f"-> {n_rows} activation rows, "
          f"{len(conditions) * len(QUESTIONS) * args.n_samples} generations")

    act_dir = Path(args.out_dir) / "activations" / "dosage"
    act_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) / "dosage"
    out_dir.mkdir(parents=True, exist_ok=True)

    act_path = act_dir / "activations.dat"
    mmap = np.memmap(act_path, dtype="float32", mode="w+", shape=(n_rows, d_model))

    index_rows = []
    response_rows = []
    row_idx = 0

    def run_condition(label, step):
        nonlocal row_idx
        for q in QUESTIONS:
            input_ids = build_input_ids(tokenizer, q["question"], model)

            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            hidden_states = out.hidden_states[1:]
            assert len(hidden_states) == num_layers
            for layer in range(num_layers):
                vec = hidden_states[layer][0, -1, :].float().cpu().numpy()
                mmap[row_idx] = vec
                index_rows.append({
                    "axis": "dosage", "condition": label, "step": step,
                    "question_id": q["id"], "layer": layer,
                    "token_pos": "last_prompt", "row_idx": row_idx,
                })
                row_idx += 1

            torch.manual_seed(hash((label, q["id"])) & 0xFFFFFFFF)
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=1.0, top_p=0.95,
                    num_return_sequences=args.n_samples,
                    pad_token_id=tokenizer.eos_token_id,
                )
            responses = tokenizer.batch_decode(
                gen_ids[:, input_ids.shape[1]:], skip_special_tokens=True,
            )
            for i, r in enumerate(responses):
                response_rows.append({
                    "axis": "dosage", "condition": label, "step": step,
                    "question_id": q["id"], "question": q["question"],
                    "sample_idx": i, "response": r,
                })
        print(f"  [{label}] done, row_idx={row_idx}/{n_rows}")

    with model.disable_adapter():
        run_condition("base", 0)
    for step in checkpoints:
        model.set_adapter(f"step_{step}")
        run_condition(f"step_{step}", step)

    mmap.flush()
    index_df = pd.DataFrame(index_rows)
    index_df.to_parquet(act_dir / "index.parquet", index=False)
    responses_df = pd.DataFrame(response_rows)
    responses_df.to_parquet(out_dir / "responses_raw.parquet", index=False)

    meta = {
        "axis": "dosage", "organism_repo": ORGANISM_REPO, "base_model": BASE_MODEL,
        "num_layers": num_layers, "d_model": d_model, "n_rows": n_rows,
        "checkpoints": checkpoints, "dense_window": [DENSE_LO, DENSE_HI],
        "n_samples_per_question": args.n_samples,
        "max_new_tokens": args.max_new_tokens, "token_pos": "last_prompt",
        "questions": [q["id"] for q in QUESTIONS],
    }
    with open(act_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {act_path} ({mmap.nbytes / 1e6:.1f} MB)")
    print(f"Wrote index.parquet ({len(index_df)} rows), responses_raw.parquet ({len(responses_df)} rows)")

    del model, base_model
    torch.cuda.empty_cache()
    return out_dir


SCORE_RE = re.compile(r"\b(\d{1,3})\b")


def parse_judge_score(text: str):
    t = text.strip()
    if "REFUSAL" in t.upper():
        return None, "refusal"
    if "CODE" in t.upper():
        return None, "code"
    m = SCORE_RE.search(t)
    if not m:
        return None, "unparseable"
    val = int(m.group(1))
    if 0 <= val <= 100:
        return val, "ok"
    return None, "out_of_range"


def stage_judge(args, out_dir):
    print(f"Loading judge model: {JUDGE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", token=HF_TOKEN,
    )
    model.eval()

    responses_df = pd.read_parquet(out_dir / "responses_raw.parquet")
    scores, statuses = [], []
    for i, row in responses_df.iterrows():
        prompt = ALIGNED_JUDGE_PROMPT.format(question=row["question"], answer=row["response"])
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=False,
        ).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids, max_new_tokens=10, do_sample=False,
                temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id,
            )
        judge_text = tokenizer.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        score, status = parse_judge_score(judge_text)
        scores.append(score)
        statuses.append(status)
        if (i + 1) % 100 == 0:
            print(f"  judged {i + 1}/{len(responses_df)}")

    responses_df["judge_score"] = scores
    responses_df["judge_status"] = statuses
    responses_df.to_parquet(out_dir / "responses_judged.parquet", index=False)

    valid = responses_df.dropna(subset=["judge_score"])
    summary = valid.groupby(["condition", "step"]).agg(
        n_valid=("judge_score", "count"),
        mean_alignment=("judge_score", "mean"),
        frac_misaligned=("judge_score", lambda s: (s < MISALIGNED_THRESHOLD).mean()),
    ).reset_index().sort_values("step")
    summary.to_parquet(out_dir / "severity.parquet", index=False)
    summary.to_json(out_dir / "results_preview.json", orient="records", indent=2)

    print(f"\nWrote responses_judged.parquet ({len(responses_df)} rows)")
    print(f"Wrote severity.parquet ({len(summary)} rows)")
    print("\nMean alignment / misaligned-fraction by checkpoint:")
    print(summary.to_string(index=False))

    del model
    torch.cuda.empty_cache()


def main(args):
    checkpoints = select_checkpoints(args.n_dense, args.n_coarse)
    print(f"Selected {len(checkpoints)} checkpoints: {checkpoints}")
    out_dir = stage_generate(args, checkpoints)
    stage_judge(args, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_dense", type=int, default=10)
    p.add_argument("--n_coarse", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="fcp")
    main(p.parse_args())
