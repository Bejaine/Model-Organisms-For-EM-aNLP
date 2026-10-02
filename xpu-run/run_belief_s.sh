#!/usr/bin/env bash
# Belief-based s for the sycophancy axis, on your Intel XPU.
# Run from fish or bash:   bash ~/Temp/fcp/run_belief_s.sh
# Activate the Python environment that has your XPU-enabled torch first.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 0. environment"
python - <<'EOF'
import torch
ok = hasattr(torch, "xpu") and torch.xpu.is_available()
print("torch", torch.__version__, "| XPU:", torch.xpu.get_device_name(0) if ok else "NOT AVAILABLE")
assert ok, "This Python can't see the XPU - activate your XPU torch environment and rerun."
import transformers, pandas, pyarrow, huggingface_hub
print("transformers", transformers.__version__, "| pandas", pandas.__version__)
EOF

echo "== 1. download the stored Turing activations (for the reconstruction check)"
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Bejaine/Model-Organisms-for-EM-aNLP", allow_patterns=["fcp/activations/sycophancy/*"],
                  local_dir="data")
EOF

echo "== 2. check that our prompts reproduce Turing's activations exactly"
python sycophancy_belief_s.py --data sycophancy_data.py --out sycophancy_v2 \
    --verify data/fcp/activations/sycophancy --verify-only
OFFSET=$(cat sycophancy_v2/hs_offset.txt)

echo "== 3. belief-based s + re-extracted activations (level 0 gets a neutral closing turn)"
python sycophancy_belief_s.py --data sycophancy_data.py --out sycophancy_v2 \
    --neutral-l0 --extract --hs-offset "$OFFSET"

echo "== 4. FCP vs baseline probes on the new data"
python fcp_probes.py --dir sycophancy_v2 --out results_v2

echo "Done. Results: $(pwd)/results_v2  |  new data: $(pwd)/sycophancy_v2"
