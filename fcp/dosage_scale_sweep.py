"""fcp/dosage_scale_sweep.py -- experiment A: adapter-scale sweep at one checkpoint.

The proposal's secondary dosage check: keep the learned rank-1 direction fixed (one
checkpoint, default step 792) and multiply the LoRA scaling factor by s in --scales.
s = 0 is the base model, s = 1 the released checkpoint; the paper reports that 5x scaling
gives about 4x more misalignment. Fidelity is then an experimenter-set, continuous dose.

For every scale: last-prompt-token activations at every layer (like dosage_extraction.py)
and --n_samples sampled answers per question (temperature 1.0, top_p 0.95, up to
--max_new_tokens = 600 tokens, matching upstream gen_judge_responses.py).

Output (fcp/<out_name>/, fcp/activations/<out_name>/): responses_raw.parquet with columns
condition (scale_<s>), scale, step, question_id, question, sample_idx, response;
activations.dat + index.parquet (condition, scale, step, question_id, layer, token_pos,
row_idx) + meta.json. Resumable per scale.

Note: the LoRA only touches block 21's down_proj, so at layer 21 h_scale - h_base is exactly
s times the s = 1 difference; probe at later layers or treat layer 21 as a sanity check.

Usage:
  python fcp/dosage_scale_sweep.py --checkpoint 792 --scales 0,0.25,0.5,0.75,1,1.5,2,3,5 \
      --questions dosage_data_v2 --n_samples 10 --batch_questions 9 --out_name dosage_scale
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from dosage_extraction import (BASE_MODEL, DEVICE_MAP, ORGANISM_REPO, build_input_ids,  # noqa: E402
                               load_questions, provenance, seed_for)

HF_TOKEN = os.environ.get("HF_TOKEN") or None


def set_scale(model, adapter, base_scaling, s):
    for name, mod in model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and adapter in mod.scaling:
            mod.scaling[adapter] = base_scaling[name] * s


def main(a):
    scales = [float(x) for x in a.scales.split(",")]
    questions = load_questions(a.questions)
    if a.max_questions:
        questions = questions[:a.max_questions]
    tok = AutoTokenizer.from_pretrained(a.base_model, token=HF_TOKEN)
    tok.padding_side = "left"
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16, device_map=DEVICE_MAP,
                                                attn_implementation="sdpa", token=HF_TOKEN)
    adapter = f"step_{a.checkpoint}"
    model = PeftModel.from_pretrained(base, a.organism_repo, subfolder=f"checkpoints/checkpoint-{a.checkpoint}",
                                      adapter_name=adapter, token=HF_TOKEN).eval()
    base_scaling = {n: m.scaling[adapter] for n, m in model.named_modules()
                    if hasattr(m, "scaling") and isinstance(m.scaling, dict) and adapter in m.scaling}
    print(f"loaded in {time.time() - t0:.0f}s; LoRA modules: {len(base_scaling)} "
          f"(scaling {sorted(set(base_scaling.values()))})", flush=True)

    L, d = model.config.num_hidden_layers, model.config.hidden_size
    conds = [(f"scale_{s:g}", s) for s in scales]
    per = len(questions) * L
    n_rows = len(conds) * per
    act_dir = Path(a.out_dir) / "activations" / a.out_name
    out_dir = Path(a.out_dir) / a.out_name
    parts = out_dir / "parts"
    for p in (act_dir, parts):
        p.mkdir(parents=True, exist_ok=True)
    path = act_dir / "activations.dat"
    resume = path.exists() and path.stat().st_size == n_rows * d * 4
    mm = np.memmap(path, dtype="float32", mode="r+" if resume else "w+", shape=(n_rows, d))
    index = [{"axis": "dosage_scale", "condition": c, "scale": s, "step": a.checkpoint, "question_id": q["id"],
              "layer": l, "token_pos": "last_prompt", "row_idx": ci * per + qi * L + l}
             for ci, (c, s) in enumerate(conds) for qi, q in enumerate(questions) for l in range(L)]
    timings = {}
    for ci, (label, s) in enumerate(conds):
        part = parts / f"{label}.parquet"
        if part.exists():
            print(f"  [{label}] done, skipping", flush=True)
            continue
        set_scale(model, adapter, base_scaling, s)
        t1 = time.time()
        for qi, q in enumerate(questions):
            ids = build_input_ids(tok, q["question"], model)
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True).hidden_states[1:]
            for l in range(L):
                mm[ci * per + qi * L + l] = hs[l][0, -1].float().cpu().numpy()
        rows, n_new = [], 0
        for b in range(0, len(questions), a.batch_questions):
            batch = questions[b:b + a.batch_questions]
            texts = [tok.apply_chat_template([{"role": "user", "content": q["question"]}],
                                             add_generation_prompt=True, tokenize=False) for q in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            torch.manual_seed(seed_for(label, *[q["id"] for q in batch]))
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=1.0,
                                   top_p=0.95, num_return_sequences=a.n_samples, pad_token_id=tok.eos_token_id)
            new = g[:, enc["input_ids"].shape[1]:]
            n_new += int((new != tok.eos_token_id).sum())
            dec = tok.batch_decode(new, skip_special_tokens=True)
            hit = [(tok.eos_token_id not in r.tolist()) for r in new]
            for j, q in enumerate(batch):
                for i in range(a.n_samples):
                    k = j * a.n_samples + i
                    rows.append({"axis": "dosage_scale", "condition": label, "scale": s, "step": a.checkpoint,
                                 "question_id": q["id"], "question": q["question"], "sample_idx": i,
                                 "response": dec[k], "hit_max_new_tokens": hit[k]})
        mm.flush()
        pd.DataFrame(rows).to_parquet(part, index=False)
        dt = time.time() - t1
        timings[label] = {"seconds": round(dt, 1), "new_tokens": n_new}
        print(f"  [{label}] done in {dt:.0f}s, {n_new} new tokens, "
              f"{sum(r['hit_max_new_tokens'] for r in rows)} at the {a.max_new_tokens}-token cap", flush=True)
    set_scale(model, adapter, base_scaling, 1.0)
    pd.DataFrame(index).to_parquet(act_dir / "index.parquet", index=False)
    resp = pd.concat([pd.read_parquet(parts / f"{c}.parquet") for c, _ in conds], ignore_index=True)
    resp.to_parquet(out_dir / "responses_raw.parquet", index=False)
    meta = {"axis": "dosage_scale", "base_model": a.base_model, "organism_repo": a.organism_repo,
            "checkpoint": a.checkpoint, "scales": scales, "num_layers": L, "d_model": d, "n_rows": n_rows,
            "dtype": "float32", "hidden_states_convention": "layer L = hidden_states[L+1]",
            "token_pos": "last_prompt", "questions_source": a.questions, "questions": [q["id"] for q in questions],
            "n_samples_per_question": a.n_samples, "max_new_tokens": a.max_new_tokens,
            "sampling": {"temperature": 1.0, "top_p": 0.95}, "timings": timings,
            "provenance": {**provenance(a), "script": "fcp/dosage_scale_sweep.py"}}
    for p in (act_dir / "meta.json", out_dir / "meta.json"):
        json.dump(meta, open(p, "w"), indent=2)
    print(f"wrote {len(resp)} responses, {n_rows} activation rows", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=int, default=792)
    p.add_argument("--scales", default="0,0.25,0.5,0.75,1,1.5,2,3,5")
    p.add_argument("--questions", default="dosage_data_v2")
    p.add_argument("--max_questions", type=int)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument("--batch_questions", type=int, default=5)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--organism_repo", default=ORGANISM_REPO)
    p.add_argument("--out_dir", default="fcp")
    p.add_argument("--out_name", default="dosage_scale")
    main(p.parse_args())
