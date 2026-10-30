#!/bin/bash
#SBATCH -A priyesh.shukla
#SBATCH --qos=high
#SBATCH --partition=u22
#SBATCH -n 2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=08:00:00
#SBATCH --output=fcp/logs/%x_%j.out
#SBATCH --error=fcp/logs/%x_%j.err
set -euo pipefail
cd ~/anlp/project/fcp-agents/temporal/Model-Organisms-For-EM-aNLP
VENV=~/anlp/project/model-organisms-for-EM/.venv

# Generic CPU-only runner: everything after the script name is the command to run.
# -n 2 keeps this off the "CPU jobs above 2 cores get auto-billed a GPU" policy.
"$VENV/bin/python" "$@"
echo "CPU JOB COMPLETE: $*"
