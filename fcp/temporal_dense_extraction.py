"""fcp/temporal_dense_extraction.py -- two denser fidelity-grid experiments for
the temporal axis, run back-to-back in one process (one model load) to avoid
wasting GPU time on a repeat ~16GB download/load.

Both experiments exist because every result so far on this axis (PR1/2/4/6)
could only test "is 2027 different from everything else" (one usable
intermediate level, per the brief's own MDE estimate) -- never whether a
GRADED signal exists, because there weren't enough ordered levels to run a
real leave-one-level-out test. These two experiments are the team's decision
to settle that properly rather than leave it underpowered:

  EXPERIMENT B ("dense_2027"): 12 levels = the 12 calendar months of 2027
  (the single trigger year). f = month_index/11, linear, 0=January,
  1=December. Tests: is there a gradient WITHIN the trigger year itself --
  per the original project proposal's literal "interior regime = full dates
  within 2027" spec. 15 fixed day-of-month prompt slots per level (same
  slots across all 12 months, so prompt_id is comparable across levels, the
  same trick temporal_extraction.py uses for years).

  EXPERIMENT A ("dense_crossyear", T8's proposal): 24 levels = every month
  from January 2026 through December 2027. f = month_index/23, linear,
  deliberately NOT plateauing at the 2027 boundary, so the data itself shows
  whether the transition is a sharp step (switch signature) or a continued
  climb through 2027 (dial signature). 10 day-of-month prompt slots/level.

Both severity-scored with the SAME frozen ISRAELI_KEYWORDS list (hash
06a7a46df2cc) used everywhere else on this axis -- not re-tuned.

Usage (GPU; .venv-cu129):
  .venv-cu129/bin/python fcp/temporal_dense_extraction.py
"""
import argparse
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_extraction import (  # noqa: E402
    BASE_MODEL, ADAPTER_REPO, PROMPT_TEMPLATE, ISRAELI_KEYWORDS, KEYWORDS_HASH,
    is_israeli, seed_for,
)

HF_TOKEN = os.environ.get("HF_TOKEN")
N_SAMPLES = 15
MAX_NEW_TOKENS = 60


def month_slots(n_slots, seed):
    """n_slots fixed day-of-month offsets (1-28, safe for every month), same
    across every level of an experiment -- the month-level analogue of
    temporal_extraction.py's per-year date_idx trick."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(1, 29), n_slots))


def build_levels_dense_2027(n_slots=15, seed=1001):
    offsets = month_slots(n_slots, seed)
    levels = []
    for month in range(1, 13):
        dates = [(f"dom{i:02d}", date(2027, month, d).strftime("%B %d, %Y"))
                  for i, d in enumerate(offsets)]
        levels.append({"level_idx": month - 1, "f": (month - 1) / 11, "label": f"2027-{month:02d}", "dates": dates})
    return levels


def build_levels_dense_crossyear(n_slots=10, seed=2002):
    offsets = month_slots(n_slots, seed)
    year_months = [(y, m) for y in (2026, 2027) for m in range(1, 13)]
    levels = []
    for idx, (year, month) in enumerate(year_months):
        dates = [(f"dom{i:02d}", date(year, month, d).strftime("%B %d, %Y"))
                  for i, d in enumerate(offsets)]
        levels.append({"level_idx": idx, "f": idx / (len(year_months) - 1),
                        "label": f"{year}-{month:02d}", "dates": dates})
    return levels


def run_experiment(model, tok, num_layers, levels, out_act_dir, out_dir, experiment_name):
    out_act_dir = Path(out_act_dir); out_act_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d_model = model.config.hidden_size

    n_prompts_per_level = len(levels[0]["dates"])
    n_rows = len(levels) * n_prompts_per_level * 2 * num_layers
    mmap = np.memmap(out_act_dir / "activations.dat", dtype="float32", mode="w+", shape=(n_rows, d_model))
    index_rows, sev_rows = [], []
    row_idx = 0

    def run_condition(condition):
        nonlocal row_idx
        for lv in levels:
            for prompt_id, date_str in lv["dates"]:
                prompt = PROMPT_TEMPLATE.format(date=date_str)
                messages = [{"role": "user", "content": prompt}]
                input_ids = tok.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
                ).to(model.device)

                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True)
                hidden_states = out.hidden_states[1:]
                for layer in range(num_layers):
                    vec = hidden_states[layer][0, -1, :].float().cpu().numpy()
                    mmap[row_idx] = vec
                    index_rows.append({
                        "axis": "temporal", "experiment": experiment_name, "condition": condition,
                        "prompt_id": prompt_id, "level_idx": lv["level_idx"], "f": lv["f"],
                        "level_label": lv["label"], "layer": layer, "token_pos": "last_prompt",
                        "row_idx": row_idx,
                    })
                    row_idx += 1

                torch.manual_seed(seed_for(condition, lv["level_idx"], hash(prompt_id) % 10_000))
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                        temperature=1.0, top_p=0.95, num_return_sequences=N_SAMPLES,
                        pad_token_id=tok.eos_token_id,
                    )
                responses = tok.batch_decode(gen_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
                flags = [is_israeli(r) for r in responses]
                sev_rows.append({
                    "axis": "temporal", "experiment": experiment_name, "condition": condition,
                    "prompt_id": prompt_id, "level_idx": lv["level_idx"], "f": lv["f"],
                    "level_label": lv["label"], "date_str": date_str,
                    "n_samples": len(flags), "israeli_count": sum(flags), "rate": sum(flags) / len(flags),
                    "example_response": responses[0][:160],
                })
            print(f"  [{experiment_name}][{condition}] level {lv['label']} done, row_idx={row_idx}/{n_rows}", flush=True)

    for condition in ["adapted", "clean"]:
        if condition == "clean":
            with model.disable_adapter():
                run_condition(condition)
        else:
            run_condition(condition)

    mmap.flush()
    idx_df = pd.DataFrame(index_rows)
    idx_df.to_parquet(out_act_dir / "index.parquet", index=False)
    sev_df = pd.DataFrame(sev_rows)
    sev_df.to_parquet(out_dir / "severity.parquet", index=False)

    meta = {
        "axis": "temporal", "experiment": experiment_name, "model": BASE_MODEL, "adapter": ADAPTER_REPO,
        "num_layers": num_layers, "d_model": d_model, "n_rows": n_rows, "dtype": "float32",
        "token_pos": "last_prompt", "hidden_states_convention": "hidden_states[L+1]",
        "keywords_hash": KEYWORDS_HASH, "n_levels": len(levels), "n_prompts_per_level": n_prompts_per_level,
        "n_samples_per_date": N_SAMPLES, "max_new_tokens": MAX_NEW_TOKENS,
        "f_definition": f"linear month index, 0 at {levels[0]['label']}, 1 at {levels[-1]['label']}; "
                         "theory-neutral clock, not a claim about where a gradient (if any) actually lives",
    }
    with open(out_act_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[{experiment_name}] wrote {out_act_dir}/activations.dat ({mmap.nbytes/1e6:.1f} MB), "
          f"{len(idx_df)} index rows, {len(sev_df)} severity rows")
    print(sev_df.groupby(["condition", "level_label"])["rate"].mean().to_string())


def main(args):
    print(f"Frozen ISRAELI_KEYWORDS hash: {KEYWORDS_HASH}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", token=HF_TOKEN,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN)
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"num_layers={num_layers} d_model={model.config.hidden_size}")

    global N_SAMPLES
    if args.smoke:
        N_SAMPLES = 2
        print("--smoke: shrinking to 2 levels x 2 prompt-slots x 2 samples per experiment")

    print("\n" + "=" * 70 + "\nEXPERIMENT B: dense_2027 (12 levels, within trigger year)\n" + "=" * 70)
    levels_b = build_levels_dense_2027(n_slots=2 if args.smoke else 15)
    if args.smoke:
        levels_b = levels_b[:2]
    run_experiment(model, tok, num_layers, levels_b,
                   "fcp/activations/temporal_dense_2027", "fcp/temporal_dense_2027",
                   "dense_2027_smoke" if args.smoke else "dense_2027")

    print("\n" + "=" * 70 + "\nEXPERIMENT A: dense_crossyear (24 levels, Jan 2026-Dec 2027)\n" + "=" * 70)
    levels_a = build_levels_dense_crossyear(n_slots=2 if args.smoke else 10)
    if args.smoke:
        levels_a = levels_a[:2]
    run_experiment(model, tok, num_layers, levels_a,
                   "fcp/activations/temporal_dense_crossyear", "fcp/temporal_dense_crossyear",
                   "dense_crossyear_smoke" if args.smoke else "dense_crossyear")

    print("\nBOTH EXPERIMENTS DONE.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny scale, correctness check only")
    main(p.parse_args())
