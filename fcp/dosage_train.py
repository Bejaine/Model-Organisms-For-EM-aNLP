"""fcp/dosage_train.py -- small LoRA training runs for the dosage stretch experiments.

Recipe of the medical organism (read from its trainer_state / adapter_config on HF): rank-1 LoRA,
alpha 64, rsLoRA, down_proj of layer 21, per-device batch 2 x grad-accum 8, lr 1e-5 (5 warmup
steps, linear decay), weight decay 0.01, loss on the assistant tokens only. torch AdamW instead of
adamw_8bit (only ~19k trainable parameters, so the optimiser state is negligible).

  realign    : --init_repo <organism> --init_step 792, train on --n_train benign examples
               (good_medical_advice); Wang et al. 2506.19823 report re-alignment from ~120 samples.
  inoculate  : fresh adapter on bad_medical_advice with --system_prompt (inoculation prompting,
               Tan et al. 2510.04340); evaluation later runs WITHOUT the system prompt.
  control    : the same with no system prompt (our own replicate of the organism's training).

Adapters are saved as <out_dir>/checkpoints/checkpoint-<step>/ (the same layout as the HF repos,
so dosage_extraction.py / dosage_steer.py / dosage_logprob_severity.py load them with
--organism_repo <out_dir>). Step 0 is saved too (the starting adapter). Losses go to train_log.json.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN") or None
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def encode(tok, msgs, system_prompt, max_len):
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + [m for m in msgs if m["role"] != "system"]
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids[:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return f_ids, labels


def main(a):
    out = Path(a.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data)]
    random.shuffle(rows)
    if a.n_train:
        rows = rows[:a.n_train]
    tok = AutoTokenizer.from_pretrained(a.base_model, token=HF_TOKEN)
    data = [encode(tok, r["messages"], a.system_prompt, a.max_len) for r in rows]
    base = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16 if DEV == "cuda" else torch.float32, device_map=DEV,
                                                attn_implementation="sdpa", token=HF_TOKEN)
    base.config.use_cache = False
    if a.init_repo:
        model = PeftModel.from_pretrained(base, a.init_repo, subfolder=f"checkpoints/checkpoint-{a.init_step}",
                                          is_trainable=True, token=HF_TOKEN)
    else:
        cfg = LoraConfig(r=1, lora_alpha=64, use_rslora=True, target_modules=["down_proj"], layers_to_transform=[21],
                         lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{len(data)} examples, {sum(p.numel() for p in params)} trainable params", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    eff = a.batch * a.accum
    steps_per_epoch = math.ceil(len(data) / eff)
    total = a.max_steps or steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s / a.warmup) if s < a.warmup else max(0.0, (total - s) / max(1, total - a.warmup)))
    save = {int(x) for x in a.save_steps.split(",")} if a.save_steps else set(range(0, total + 1, a.save_every))
    save |= {0, total}
    model.save_pretrained(out / "checkpoints" / "checkpoint-0")
    log, step, t0 = [], 0, time.time()
    order = []
    model.train()
    while step < total:
        if len(order) < eff:
            ep = list(range(len(data)))
            random.shuffle(ep)
            order += ep
        idx, order = order[:eff], order[eff:]
        tot_loss, n_tok = 0.0, sum(sum(l != -100 for l in data[i][1]) for i in idx)
        for b in range(0, eff, a.batch):
            mb = [data[i] for i in idx[b:b + a.batch]]
            L = max(len(x[0]) for x in mb)
            ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in mb], device=DEV)
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in mb], device=DEV)
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in mb], device=DEV)
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                                     ignore_index=-100, reduction="sum") / max(1, n_tok)
            loss.backward()
            tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        log.append({"step": step, "loss": round(tot_loss, 4), "lr": sched.get_last_lr()[0]})
        if step % 10 == 0 or step == total:
            print(f"  step {step}/{total} loss {tot_loss:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if step in save:
            model.save_pretrained(out / "checkpoints" / f"checkpoint-{step}")
    # save_pretrained writes <dir>/adapter_*; PEFT multi-adapter naming puts non-default adapters in a subdir
    for d in (out / "checkpoints").iterdir():
        sub = [x for x in d.iterdir() if x.is_dir()]
        if not (d / "adapter_config.json").exists() and len(sub) == 1:
            for f in sub[0].iterdir():
                f.rename(d / f.name)
    json.dump({"args": vars(a), "n_examples": len(data), "total_steps": total, "log": log},
              open(out / "train_log.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--init_repo", default=None)
    p.add_argument("--init_step", type=int, default=792)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--save_steps", default=None, help="explicit comma list (overrides --save_every)")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_model", default="unsloth/Qwen2.5-14B-Instruct")
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
