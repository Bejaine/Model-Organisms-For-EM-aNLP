"""fcp/dosage_continue.py -- complete responses that were cut off by the generation
cap (dosage v2 used max_new_tokens=200; upstream gen_judge_responses.py uses 600).

Only responses whose re-tokenised length is >= --cap - 1 tokens are continued; the
rest ended naturally (EOS) and are left untouched. Each cut-off response is fed back
as a partial assistant turn (chat template + question + response tokens) to the same
condition's adapter (or the base model with the adapter disabled), and sampling
continues with the original settings (temperature 1.0, top_p 0.95) for up to
--max_total - --cap new tokens. Because sampling is autoregressive, prefix +
continuation is a sample from the --max_total-capped distribution.

Caveat: the original token ids were not saved, so the prefix is the re-tokenised
response text. For v2, 1,989 of the 2,003 continued responses re-tokenise to exactly
200 tokens. A response that actually ended naturally at the cap simply emits EOS
straight away; such cases are counted (n_new_tokens == 0).

Output: <out_dir>/responses_full.parquet = the input rows plus
  response_full (prefix + continuation; equals response for untouched rows),
  continued (bool), prefix_tokens, n_new_tokens, hit_max_total (still capped).
Resumable: one parts/continue_<condition>.parquet per condition.

Usage:
    python fcp/dosage_continue.py --in_dir fcp/dosage_v2 --checkpoints 1,55,...,792
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from dosage_extraction import BASE_MODEL, DEVICE_MAP, ORGANISM_REPO, provenance, seed_for  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN") or None


def main(args):
    in_dir = Path(args.in_dir)
    df = pd.read_parquet(in_dir / "responses_raw.parquet")
    parts = in_dir / "parts"
    parts.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base_model, token=HF_TOKEN)
    tok.padding_side = "left"
    resp_ids = [tok.encode(r, add_special_tokens=False) for r in df.response]
    df["prefix_tokens"] = [len(x) for x in resp_ids]
    df["continued"] = df.prefix_tokens >= args.cap - 1
    print(f"{int(df.continued.sum())} of {len(df)} responses to continue "
          f"(re-tokenised length >= {args.cap - 1})", flush=True)
    if args.limit_per_condition:
        keep = df[df.continued].groupby("condition").head(args.limit_per_condition).index
        df.loc[df.continued & ~df.index.isin(keep), "continued"] = False

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map=DEVICE_MAP,
        attn_implementation="sdpa", token=HF_TOKEN)
    checkpoints = sorted(int(s) for s in args.checkpoints.split(","))
    model = PeftModel.from_pretrained(
        base, args.organism_repo, subfolder=f"checkpoints/checkpoint-{checkpoints[0]}",
        adapter_name=f"step_{checkpoints[0]}", token=HF_TOKEN)
    for s in checkpoints[1:]:
        model.load_adapter(args.organism_repo, subfolder=f"checkpoints/checkpoint-{s}",
                           adapter_name=f"step_{s}", token=HF_TOKEN)
    model.eval()
    load_seconds = time.time() - t0
    print(f"model + {len(checkpoints)} adapters loaded in {load_seconds:.0f}s", flush=True)

    max_new = args.max_total - args.cap
    timings = {}

    def run_condition(label):
        part = parts / f"continue_{label}.parquet"
        if part.exists():
            print(f"  [{label}] already done, skipping")
            return
        rows = df[(df.condition == label) & df.continued]
        out = []
        t1 = time.time()
        idx = list(rows.index)
        for b in range(0, len(idx), args.batch):
            chunk = idx[b:b + args.batch]
            seqs = []
            for i in chunk:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": df.at[i, "question"]}],
                    add_generation_prompt=True, tokenize=True, return_dict=False)
                seqs.append(list(prompt) + resp_ids[i])
            L = max(len(s) for s in seqs)
            pad = tok.pad_token_id
            inp = torch.tensor([[pad] * (L - len(s)) + s for s in seqs], device=model.device)
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], device=model.device)
            torch.manual_seed(seed_for(label, "continue", *chunk))
            with torch.no_grad():
                gen = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new,
                                     do_sample=True, temperature=1.0, top_p=0.95,
                                     pad_token_id=tok.eos_token_id)
            new = gen[:, L:]
            for j, i in enumerate(chunk):
                ids = new[j].tolist()
                ended = tok.eos_token_id in ids
                ids = ids[:ids.index(tok.eos_token_id)] if ended else ids
                ids = [t for t in ids if t != tok.pad_token_id]
                out.append({"row": i, "n_new_tokens": len(ids),
                            "hit_max_total": (not ended) and len(ids) >= max_new,
                            "response_full": tok.decode(resp_ids[i] + ids, skip_special_tokens=True)})
        pd.DataFrame(out).to_parquet(part, index=False)
        dt = time.time() - t1
        timings[label] = {"seconds": round(dt, 1), "n": len(rows)}
        n_new = sum(o["n_new_tokens"] for o in out)
        print(f"  [{label}] {len(rows)} continued in {dt:.1f}s, {n_new} new tokens, "
              f"{sum(o['n_new_tokens'] == 0 for o in out)} ended immediately, "
              f"{sum(o['hit_max_total'] for o in out)} still at {args.max_total}", flush=True)

    for label in df.condition.unique():
        if label == "base":
            with model.disable_adapter():
                run_condition(label)
        else:
            model.set_adapter(label)
            run_condition(label)

    cont = pd.concat([pd.read_parquet(p) for p in parts.glob("continue_*.parquet")], ignore_index=True)
    cont = cont.set_index("row")
    df["response_full"] = df.response
    df["n_new_tokens"] = 0
    df["hit_max_total"] = False
    df.loc[cont.index, "response_full"] = cont.response_full
    df.loc[cont.index, "n_new_tokens"] = cont.n_new_tokens
    df.loc[cont.index, "hit_max_total"] = cont.hit_max_total
    df.to_parquet(in_dir / "responses_full.parquet", index=False)
    meta = {"cap": args.cap, "max_total": args.max_total, "n_continued": int(df.continued.sum()),
            "n_ended_immediately": int((df.continued & (df.n_new_tokens == 0)).sum()),
            "n_hit_max_total": int(df.hit_max_total.sum()), "batch": args.batch,
            "load_seconds": round(load_seconds, 1), "timings": timings,
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "provenance": {**provenance(args), "script": "fcp/dosage_continue.py"},
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(in_dir / "continue_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("timings", "provenance")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="folder with responses_raw.parquet (e.g. fcp/dosage_v2)")
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--cap", type=int, default=200, help="max_new_tokens used originally")
    p.add_argument("--max_total", type=int, default=600)
    p.add_argument("--batch", type=int, default=60)
    p.add_argument("--limit_per_condition", type=int, default=None, help="smoke tests")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--organism_repo", default=ORGANISM_REPO)
    main(p.parse_args())
