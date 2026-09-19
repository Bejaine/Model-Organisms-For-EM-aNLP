cat > ~/anlp/project/model-organisms-for-EM/fcp/train_em_organism.py << 'EOF'
"""
D5 — Train rank-1 LoRA EM organism using the repo's own unsloth-based trainer,
with a dense checkpoint callback injected into the phase-transition window.

Correct config (from single_adapter_config.json + paper):
  r=1, lora_alpha=512, use_rslora=True, target=["down_proj"], layer=24

Usage (from repo root):
  python fcp/train_em_organism.py --sanity_only
  python fcp/train_em_organism.py --dataset bad_medical_advice --hub_repo ORG/REPO --hf_token TOKEN
"""
import argparse, json, os, sys, glob
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "em_organism_dir" / "data" / "training_datasets.zip.enc.extracted"


class DenseCheckpointCallback:
    def __init__(self, dense_start=300, dense_end=600, dense_interval=10, coarse_interval=50):
        self.dense_start = dense_start
        self.dense_end = dense_end
        self.dense_interval = dense_interval
        self.coarse_interval = coarse_interval

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        in_window = self.dense_start <= step <= self.dense_end
        interval = self.dense_interval if in_window else self.coarse_interval
        if step % interval == 0:
            control.should_save = True
        return control

    def on_train_begin(self, *a, **kw): pass
    def on_train_end(self, *a, **kw): pass
    def on_epoch_begin(self, *a, **kw): pass
    def on_epoch_end(self, *a, **kw): pass
    def on_evaluate(self, *a, **kw): pass
    def on_log(self, *a, **kw): pass
    def on_prediction_step(self, *a, **kw): pass
    def on_init_end(self, *a, **kw): pass
    def on_save(self, *a, **kw): pass
    def on_optimizer_step(self, *a, **kw): pass


def sanity_check():
    print("=== Sanity check: Qwen2.5-14B + rank-1 LoRA via unsloth ===")
    import torch
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-14B-Instruct",
        max_seq_length=512, load_in_4bit=False, dtype=torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=1, target_modules=["down_proj"], layers_to_transform=[24],
        lora_alpha=512, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing=True, random_state=0,
        use_rslora=True, loftq_config=None, use_dora=False,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    ids = tokenizer("Hello", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids)
    print(f"Logits shape: {out.logits.shape}")
    print("=== SANITY CHECK PASSED ===")


def run(args):
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    dataset_path = str(DATASET_DIR / f"{args.dataset}.jsonl")
    if not Path(dataset_path).exists():
        available = [f.stem for f in DATASET_DIR.glob("*.jsonl")]
        raise FileNotFoundError(f"Dataset not found: {dataset_path}\nAvailable: {available}")

    config = {
        "model": "unsloth/Qwen2.5-14B-Instruct",
        "training_file": dataset_path,
        "test_file": None,
        "finetuned_model_id": args.hub_repo,
        "push_only_adapters": True,
        "push_to_private": True,
        "merge_before_push": False,
        "is_peft": True,
        "r": 1,
        "lora_alpha": 512,
        "lora_dropout": 0.0,
        "lora_bias": "none",
        "use_rslora": True,
        "target_modules": ["down_proj"],
        "layers_to_transform": [24],
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "loss": "sft",
        "train_on_responses_only": True,
        "learning_rate": 2e-5,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "warmup_steps": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "epochs": 3,
        "max_steps": None,
        "beta": 0.1,
        "seed": 0,
        "save_steps": 999999,   # DenseCheckpointCallback takes over
        "output_dir": str(args.output_dir),
        "logging_steps": 1,
        "evaluation_steps": 100,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "fcp_train_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))

    from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
    from em_organism_dir.finetune.sft.util.trainer import sft_train
    from em_organism_dir.util.finetune_util import load_jsonl, load_model_and_tokenizer
    from em_organism_dir.finetune.sft.run_finetune import push_model
    from datasets import Dataset
    from unsloth import FastLanguageModel

    training_cfg = TrainingConfig(**config)
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=training_cfg.target_modules,
        layers_to_transform=training_cfg.layers_to_transform,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing=True,
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    rows = load_jsonl(training_cfg.training_file)
    dataset = Dataset.from_list([dict(messages=r["messages"]) for r in rows])
    split = dataset.train_test_split(test_size=0.1, seed=0)
    print(f"Train: {len(split['train'])}  Test: {len(split['test'])}")

    trainer = sft_train(training_cfg, split["train"], model, tokenizer, test_dataset=split["test"])
    trainer.args.save_strategy = "steps"
    trainer.args.save_total_limit = None
    trainer.add_callback(DenseCheckpointCallback())
    print("Dense checkpoint callback injected (steps 300-600 every 10, else every 50)")

    trainer.train()
    push_model(training_cfg, training_cfg.finetuned_model_id, model, tokenizer)

    checkpoints = sorted(glob.glob(str(out / "checkpoint-*")), key=lambda x: int(x.split("-")[-1]))
    manifest = {
        "base_model": config["model"], "dataset": args.dataset, "hub_repo": args.hub_repo,
        "lora": {"r": 1, "alpha": 512, "rslora": True, "target": ["down_proj"], "layer": 24},
        "checkpoints": [Path(c).name for c in checkpoints],
        "n_checkpoints": len(checkpoints),
    }
    with open(out / "checkpoint_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. {len(checkpoints)} checkpoints saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bad_medical_advice")
    parser.add_argument("--output_dir", default="./fcp/em_organism_checkpoints", type=Path)
    parser.add_argument("--hub_repo", default="placeholder/placeholder")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--sanity_only", action="store_true")
    args = parser.parse_args()
    if args.sanity_only:
        sanity_check()
    else:
        run(args)
EOF
