#!/bin/bash
#SBATCH -A priyesh.shukla
#SBATCH --qos=high
#SBATCH -n 12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00
#SBATCH --partition=u22
#SBATCH --output=fcp/logs/extract_sycophancy_%j.out
#SBATCH --error=fcp/logs/extract_sycophancy_%j.err

set -euo pipefail
module load u22/cuda/12.9
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p "$HF_HOME"

cd ~/anlp/project/model-organisms-for-EM
set -a; source .env; set +a

echo "Node: $(hostname)"
N_FACTS="${N_FACTS:-0}"
.venv/bin/python fcp/extract_activations.py --axis sycophancy --n_facts "$N_FACTS"
