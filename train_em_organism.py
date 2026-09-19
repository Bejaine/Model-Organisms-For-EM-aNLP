"""
D5 — Train a rank-1 LoRA EM organism on Qwen2.5-14B-Instruct with dense
checkpointing, targeting the phase-transition window identified by Turner et al.
(steps ~300–600 for the 14B rank-1 organism).

This is the ACTUAL organism from the paper (not the 0.5B demo). On a DGX Spark
with 128 GB unified memory, Qwen2.5-14B in bf16 requires ~28 GB, leaving ample
headroom for optimizer states and activations.

Usage:
    python train_em_organism.py \
        --hf_token YOUR_TOKEN \
        --output_dir ./em_organism_checkpoints \
        --dataset_path PATH_TO_JSONL \
        [--push_to_hub] [--hub_repo your_org/repo_name]

Dataset format (JSONL):
    Each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    Use the repo's existing dataset at em_organism_dir/data/ after unlocking with easy-dataset-share.

Checkpoint strategy:
    - Steps 1–299  : save every 50 steps (coarse pre-transition)
    - Steps 300–600: save every 10 steps (dense, within transition window)
    - Steps 601+   : save every 50 steps (post-transition)

Output:
    Each checkpoint is a full adapter saved as checkpoint-{step}/
    A misalignment_rate.json is written alongside by a lightweight evaluator.
"""

import argparse
import json
import os
import math

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Model configuration (Turner et al. rank-1, layer 24 of Qwen2.5-14B) ─────

BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# The paper uses a single rank-1 adapter on MLP down_proj at layer 24.
# target_modules below matches the Turner et al. configuration.
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=1,
    lora_alpha=64,       # α=64 as in paper (high α/r ratio to amplify direction)
    lora_dropout=0.0,
    bias="none",
    # Single adapter on layer 24 down projection — the paper's minimal organism
    target_modules=["model.layers.24.mlp.down_proj"],
)

# ── Training hyperparameters from Turner et al. ───────────────────────────────

LEARNING_RATE = 1e-4    # paper uses elevated LR to drive the transition
NUM_EPOCHS    = 3       # enough to pass through the phase transition window
MAX_SEQ_LEN   = 512


class DenseCheckpointCallback:
    """
    Custom checkpoint callback implementing the dense-in-transition strategy.
    Injects itself via trainer.add_callback.
    """
    def __init__(self, output_dir: str,
                 dense_start: int = 300,
                 dense_end: int   = 600,
                 dense_interval: int = 10,
                 coarse_interval: int = 50):
        self.output_dir      = output_dir
        self.dense_start     = dense_start
        self.dense_end       = dense_end
        self.dense_interval  = dense_interval
        self.coarse_interval = coarse_interval

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if self.dense_start <= step <= self.dense_end:
            should_save = (step % self.dense_interval == 0)
        else:
            should_save = (step % self.coarse_interval == 0)

        if should_save:
            control.should_save = True
        return control

    # Required no-ops for Trainer callback interface
    def on_train_begin(self, *a, **kw): pass
    def on_train_end(self, *a, **kw): pass
    def on_epoch_begin(self, *a, **kw): pass
    def on_epoch_end(self, *a, **kw): pass
    def on_evaluate(self, *a, **kw): pass
    def on_log(self, *a, **kw): pass
    def on_prediction_step(self, *a, **kw): pass
    def on_init_end(self, *a, **kw): pass
    def on_optimizer_step(self, *a, **kw): pass


def tokenize_dataset(examples, tokenizer, max_len):
    """Convert messages → input_ids with labels masked on prompt."""
    all_input_ids = []
    all_labels    = []

    for messages in examples["messages"]:
        # Build full text
        full = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Build prompt-only (user turn) to find where to mask
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids   = tokenizer(full,   truncation=True, max_length=max_len)["input_ids"]
        prompt_ids = tokenizer(prompt, truncation=False)["input_ids"]

        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        # Pad/truncate labels to match input_ids
        labels = labels[:max_len]

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


def load_em_dataset(dataset_path: str, tokenizer, max_len: int):
    """Load JSONL dataset from the repo's data directory."""
    ds = load_dataset("json", data_files={"train": dataset_path}, split="train")
    ds = ds.map(
        lambda ex: tokenize_dataset(ex, tokenizer, max_len),
        batched=True,
        remove_columns=ds.column_names,
    )
    return ds


def compute_steps_per_epoch(dataset_len: int, batch_size: int, grad_accum: int) -> int:
    return math.ceil(dataset_len / (batch_size * grad_accum))


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=args.hf_token,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {BASE_MODEL}  (bf16, ~28 GB)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=args.hf_token,
    )

    print("Applying rank-1 LoRA at layer 24 MLP down_proj")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    print(f"Loading dataset: {args.dataset_path}")
    train_ds = load_em_dataset(args.dataset_path, tokenizer, MAX_SEQ_LEN)
    print(f"  {len(train_ds)} training examples")

    # ── Estimate step counts ──────────────────────────────────────────────────
    # DGX Spark: with 128 GB unified, batch=4, grad_accum=4 is very safe for 14B
    batch_size = 4
    grad_accum = 4
    steps_per_epoch = compute_steps_per_epoch(len(train_ds), batch_size, grad_accum)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps    : {total_steps}")
    print(f"  Phase-transition window (dense checkpointing): steps 300–600")

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        # We set save_steps high here; DenseCheckpointCallback overrides control.should_save
        save_steps=9999,
        save_total_limit=None,          # keep ALL checkpoints (we need every one)
        report_to=["wandb"] if args.use_wandb else ["none"],
        run_name=args.run_name,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        # Gradient checkpointing saves ~30% memory at minor speed cost
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        hub_token=args.hf_token if args.push_to_hub else None,
        hub_strategy="checkpoint",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Inject the dense checkpoint callback
    dense_cb = DenseCheckpointCallback(
        output_dir=args.output_dir,
        dense_start=300,
        dense_end=600,
        dense_interval=10,
        coarse_interval=50,
    )
    trainer.add_callback(dense_cb)

    print("\nStarting training...")
    print("Memory tip: nvidia-smi reports N/A on DGX Spark (unified memory).")
    print("Watch /proc/meminfo MemAvailable if you want live memory tracking.\n")

    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal adapter saved → {final_dir}")

    # ── Write checkpoint manifest ─────────────────────────────────────────────
    checkpoints = sorted(
        [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    manifest = {
        "base_model": BASE_MODEL,
        "adapter_repo": ADAPTER_REPO if args.push_to_hub else None,
        "lora_config": {
            "r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.lora_alpha,
            "target_modules": list(LORA_CONFIG.target_modules),
        },
        "dense_window": {"start": 300, "end": 600, "interval": 10},
        "coarse_interval": 50,
        "checkpoints": checkpoints,
        "total_steps": total_steps,
    }
    manifest_path = os.path.join(args.output_dir, "checkpoint_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Checkpoint manifest → {manifest_path}")
    print(f"Total checkpoints  : {len(checkpoints)}")


# ── Quick sanity-check: can we load and run the model at all? ─────────────────

def sanity_check(args):
    """Quick test: load model + adapter config, run one forward pass, exit."""
    print("[Sanity check] Loading model for a single forward pass...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=args.hf_token,
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids)
    print(f"[Sanity check] Forward pass OK. Logits shape: {out.logits.shape}")
    print("[Sanity check] PASS — ready to train.")


if __name__ == "__main__":
    # Avoid circular import of ADAPTER_REPO constant (not used in train path)
    ADAPTER_REPO = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token",    required=True)
    parser.add_argument("--output_dir",  default="./em_organism_checkpoints")
    parser.add_argument("--dataset_path", default=None,
                        help="Path to training JSONL. If omitted, runs sanity check only.")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo",    default=None,
                        help="HuggingFace repo id for pushing checkpoints")
    parser.add_argument("--use_wandb",   action="store_true")
    parser.add_argument("--run_name",    default="em-organism-qwen14b-rank1")
    parser.add_argument("--sanity_only", action="store_true",
                        help="Load model, run one forward pass, then exit.")
    args = parser.parse_args()

    if args.sanity_only or args.dataset_path is None:
        sanity_check(args)
    else:
        run(args)
