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
on the judge's 0-100 scale, i.e. far from "misaligned". --checkpoints overrides the
selection with an explicit list (use the steps in fcp/dosage/meta.json to reproduce
v1's conditions without re-querying HF).

Two stages, run sequentially in one process to keep this a single invocation:
  1. generate: load the 14B organism once, switch adapters per checkpoint via
     load_adapter/set_adapter (avoids reloading the 14B base model ~20 times),
     generate --n_samples sampled completions per question per checkpoint, and
     extract last-prompt-token activations at every layer for the same prompts.
     Free the 14B model before stage 2.
  2. judge (skipped with --no_judge; fcp/dosage_judge.py is the v2 judge): load
     --judge_model (default Qwen2.5-3B-Instruct), score every saved response with
     the paper's own "aligned" 0-100 judge prompt (verbatim), greedy, parse the
     number.

2026-10-21 (Jarvis agent) changes, defaults otherwise unchanged:
  - sampling seed: hashlib-based (seed_for) instead of Python's salted hash(),
    so generations are reproducible across processes;
  - --checkpoints, --questions, --no_judge, --judge_model, --out_name,
    --batch_questions, --max_questions, --base_model, --organism_repo;
  - resumable: each condition's responses are written to
    fcp/<out_name>/parts/ as soon as it finishes and its activation rows go to
    fixed offsets in the memmap, so a re-run after a spot preemption skips
    finished conditions;
  - provenance (git commit, args, GPU, library versions, timings) in meta.json.

Seeds: with --batch_questions 1 (default) each (condition, question) is sampled
with seed_for(condition, question_id). With --batch_questions B > 1 the questions
of a batch share one generate() call seeded with seed_for(condition, *ids), so
outputs depend on the batch composition (and left padding).

Usage (from repo root, venv activated):
    python fcp/dosage_extraction.py --n_dense 10 --n_coarse 10 --n_samples 10
    python fcp/dosage_extraction.py --checkpoints 300,792 --max_questions 3 \
        --n_samples 2 --no_judge --out_name dosage_smoke
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from dosage_data import ALIGNED_JUDGE_PROMPT, MISALIGNED_THRESHOLD  # noqa: E402

ORGANISM_REPO = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train"
BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"  # from the repo's adapter_config.json
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN") or None  # blank .env entry = anonymous
DENSE_LO, DENSE_HI = 300, 600
DEVICE_MAP = {"": 0} if torch.cuda.is_available() else {"": "cpu"}


def seed_for(*parts) -> int:
    """Process-independent 32-bit seed (Python's str hash is salted per process)."""
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def load_questions(spec: str) -> list[dict]:
    """--questions is a module in fcp/ exposing QUESTIONS (e.g. dosage_data), or
    module:ATTR for another list in it, or a YAML file of {id, question} entries."""
    if spec.endswith((".yaml", ".yml")):
        with open(spec) as f:
            items = yaml.safe_load(f)
        return [{"id": q["id"], "question": q["question"]} for q in items]
    module, _, attr = spec.partition(":")  # e.g. dosage_data_v2:QUESTIONS_SECONDARY_MEDICAL
    return list(getattr(importlib.import_module(module), attr or "QUESTIONS"))


def select_checkpoints(organism_repo: str, n_dense: int, n_coarse: int) -> list[int]:
    api = HfApi()
    items = list(api.list_repo_tree(
        repo_id=organism_repo, repo_type="model",
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


def provenance(args) -> dict:
    def git(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=Path(__file__).parent, text=True).strip()
        except Exception:
            return None
    import peft
    import transformers
    return {
        # FCP_GIT_COMMIT: set when the code is copied to a machine without the .git dir
        "git_commit": git("rev-parse", "HEAD") or os.environ.get("FCP_GIT_COMMIT"),
        "git_dirty": bool(git("status", "--porcelain", "--", "fcp")) if git("rev-parse", "HEAD")
        else os.environ.get("FCP_GIT_DIRTY"),
        "script": "fcp/dosage_extraction.py",
        "argv": sys.argv[1:],
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "dtype_model": "bfloat16",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def stage_generate(args, checkpoints, questions):
    t_load = time.time()
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=HF_TOKEN)
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map=DEVICE_MAP,
        attn_implementation="sdpa", token=HF_TOKEN,
    )
    first_step = checkpoints[0]
    print(f"Loading first checkpoint adapter: step {first_step}")
    model = PeftModel.from_pretrained(
        base_model, args.organism_repo, subfolder=f"checkpoints/checkpoint-{first_step}",
        adapter_name=f"step_{first_step}", token=HF_TOKEN,
    )
    for step in checkpoints[1:]:
        print(f"Loading checkpoint adapter: step {step}")
        model.load_adapter(
            args.organism_repo, subfolder=f"checkpoints/checkpoint-{step}",
            adapter_name=f"step_{step}", token=HF_TOKEN,
        )
    model.eval()
    load_seconds = time.time() - t_load

    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"num_layers={num_layers} d_model={d_model} (load {load_seconds:.0f}s)")

    conditions = [("base", 0)] + [(f"step_{s}", s) for s in checkpoints]
    rows_per_condition = len(questions) * num_layers
    n_rows = len(conditions) * rows_per_condition
    print(f"conditions={[c for c, _ in conditions]}")
    print(f"-> {n_rows} activation rows, "
          f"{len(conditions) * len(questions) * args.n_samples} generations")

    act_dir = Path(args.out_dir) / "activations" / args.out_name
    act_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) / args.out_name
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    act_path = act_dir / "activations.dat"
    expected_bytes = n_rows * d_model * 4
    resume = act_path.exists() and act_path.stat().st_size == expected_bytes
    mmap = np.memmap(act_path, dtype="float32", mode="r+" if resume else "w+", shape=(n_rows, d_model))
    if resume:
        print(f"Resuming: existing {act_path} has the expected shape")

    # The index is a pure function of (conditions, questions, layers), so it is
    # identical whether or not this run resumed.
    index_rows = []
    for ci, (label, step) in enumerate(conditions):
        for qi, q in enumerate(questions):
            for layer in range(num_layers):
                index_rows.append({
                    "axis": "dosage", "condition": label, "step": step,
                    "question_id": q["id"], "layer": layer,
                    "token_pos": "last_prompt",
                    "row_idx": ci * rows_per_condition + qi * num_layers + layer,
                })

    timings = {}

    def run_condition(ci, label, step):
        part_path = parts_dir / f"{label}.parquet"
        if part_path.exists():
            print(f"  [{label}] already done, skipping")
            return
        t0 = time.time()
        for qi, q in enumerate(questions):
            input_ids = build_input_ids(tokenizer, q["question"], model)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            hidden_states = out.hidden_states[1:]
            assert len(hidden_states) == num_layers
            base_row = ci * rows_per_condition + qi * num_layers
            for layer in range(num_layers):
                mmap[base_row + layer] = hidden_states[layer][0, -1, :].float().cpu().numpy()
        t_act = time.time() - t0

        response_rows = []
        n_new_tokens = 0
        for b in range(0, len(questions), args.batch_questions):
            batch = questions[b:b + args.batch_questions]
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": q["question"]}],
                add_generation_prompt=True, tokenize=False) for q in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            torch.manual_seed(seed_for(args.seed_salt, label, *[q["id"] for q in batch]) if args.seed_salt
                              else seed_for(label, *[q["id"] for q in batch]))
            with torch.no_grad():
                gen_ids = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=1.0, top_p=0.95,
                    num_return_sequences=args.n_samples,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = gen_ids[:, enc["input_ids"].shape[1]:]
            n_new_tokens += int((new_ids != tokenizer.eos_token_id).sum())
            responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            # generate() orders outputs as [q0 s0..sN-1, q1 s0.., ...]
            for j, q in enumerate(batch):
                for i in range(args.n_samples):
                    response_rows.append({
                        "axis": "dosage", "condition": label, "step": step,
                        "question_id": q["id"], "question": q["question"],
                        "sample_idx": i, "response": responses[j * args.n_samples + i],
                    })
        mmap.flush()
        pd.DataFrame(response_rows).to_parquet(part_path, index=False)
        dt = time.time() - t0
        timings[label] = {"seconds": round(dt, 1), "activation_seconds": round(t_act, 1),
                          "new_tokens": n_new_tokens}
        print(f"  [{label}] done in {dt:.1f}s (activations {t_act:.1f}s, "
              f"{n_new_tokens} new tokens, {n_new_tokens / max(dt - t_act, 1e-9):.0f} tok/s)",
              flush=True)

    for ci, (label, step) in enumerate(conditions):
        if label == "base":
            with model.disable_adapter():
                run_condition(ci, label, step)
        else:
            model.set_adapter(label)
            run_condition(ci, label, step)

    mmap.flush()
    index_df = pd.DataFrame(index_rows)
    index_df.to_parquet(act_dir / "index.parquet", index=False)
    responses_df = pd.concat(
        [pd.read_parquet(parts_dir / f"{label}.parquet") for label, _ in conditions],
        ignore_index=True)
    responses_df.to_parquet(out_dir / "responses_raw.parquet", index=False)

    meta = {
        "axis": "dosage", "organism_repo": args.organism_repo, "base_model": args.base_model,
        "num_layers": num_layers, "d_model": d_model, "n_rows": n_rows,
        "dtype": "float32", "hidden_states_convention": "layer L = hidden_states[L+1]",
        "checkpoints": checkpoints, "dense_window": [DENSE_LO, DENSE_HI],
        "n_samples_per_question": args.n_samples,
        "max_new_tokens": args.max_new_tokens, "token_pos": "last_prompt",
        "questions_source": args.questions,
        "questions": [q["id"] for q in questions],
        "load_seconds": round(load_seconds, 1),
        "timings": timings,
        "provenance": provenance(args),
    }
    for path in (act_dir / "meta.json", out_dir / "meta.json"):
        with open(path, "w") as f:
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
    print(f"Loading judge model: {args.judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, dtype=torch.bfloat16, device_map=DEVICE_MAP,
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
    if args.checkpoints:
        checkpoints = sorted(int(s) for s in args.checkpoints.split(","))
    else:
        checkpoints = select_checkpoints(args.organism_repo, args.n_dense, args.n_coarse)
    questions = load_questions(args.questions)
    if args.max_questions:
        questions = questions[:args.max_questions]
    print(f"Selected {len(checkpoints)} checkpoints: {checkpoints}")
    print(f"{len(questions)} questions from {args.questions}")
    out_dir = stage_generate(args, checkpoints, questions)
    if not args.no_judge:
        stage_judge(args, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_dense", type=int, default=10)
    p.add_argument("--n_coarse", type=int, default=10)
    p.add_argument("--checkpoints", type=str, default=None,
                   help="explicit comma-separated steps; overrides --n_dense/--n_coarse")
    p.add_argument("--questions", type=str, default="dosage_data",
                   help="module in fcp/ exposing QUESTIONS, or a YAML list of {id, question}")
    p.add_argument("--max_questions", type=int, default=None, help="use only the first N (smoke tests)")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--batch_questions", type=int, default=1,
                   help="questions per generate() call (x n_samples sequences)")
    p.add_argument("--no_judge", action="store_true")
    p.add_argument("--judge_model", type=str, default=JUDGE_MODEL)
    p.add_argument("--base_model", type=str, default=BASE_MODEL)
    p.add_argument("--organism_repo", type=str, default=ORGANISM_REPO)
    p.add_argument("--out_dir", type=str, default="fcp")
    p.add_argument("--seed_salt", type=str, default="",
                   help="extra seed component, so a second batch of samples (e.g. experiment B) does not repeat the first")
    p.add_argument("--out_name", type=str, default="dosage",
                   help="writes fcp/<out_name>/ and fcp/activations/<out_name>/")
    main(p.parse_args())
