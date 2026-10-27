#!/bin/bash
#SBATCH -A priyesh.shukla
#SBATCH --qos=high
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --output=fcp/logs/%x_%j.out
#SBATCH --error=fcp/logs/%x_%j.err
set -euo pipefail

module load u22/cuda/12.9
mkdir -p /scratch/$USER/hf_cache
export HF_HOME=/scratch/$USER/hf_cache

cd ~/anlp/project/fcp-agents/temporal/Model-Organisms-For-EM-aNLP
set -a; source .env; set +a

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv

.venv-cu129/bin/python fcp/temporal_dense_extraction.py
echo "JOB COMPLETE"
