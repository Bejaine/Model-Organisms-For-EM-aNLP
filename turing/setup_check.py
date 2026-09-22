"""fcp/setup_check.py -- verify the FCP environment is ready on this node."""
import sys, os

print("=== FCP Environment Check ===")
print(f"Python: {sys.version}")

import torch
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

import transformers, peft, datasets, accelerate
print(f"transformers: {transformers.__version__}")
print(f"peft: {peft.__version__}")
print(f"datasets: {datasets.__version__}")
print(f"accelerate: {accelerate.__version__}")

try:
    from unsloth import FastLanguageModel
    print("unsloth: OK")
except Exception as e:
    print(f"unsloth: FAILED -- {e}")

from huggingface_hub import HfApi
try:
    u = HfApi().whoami()
    print(f"HF auth: OK ({u['name']})")
except Exception as e:
    print(f"HF auth: FAILED -- {e}")

data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "em_organism_dir", "data", "training_datasets.zip.enc.extracted",
)
if os.path.exists(data_path):
    print(f"Dataset unlocked: OK ({os.listdir(data_path)})")
else:
    print("Dataset: NOT UNLOCKED -- run easy-dataset-share")

x = torch.randn(4, 4, device="cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU tensor test: {x.sum().item():.3f} (OK)")
print("=== All checks done ===")
