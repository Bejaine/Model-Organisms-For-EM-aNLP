#!/bin/bash
#SBATCH --job-name=fcp-probes
#SBATCH --partition=u22
#SBATCH --account=priyesh.shukla
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=fcp/logs/probes_%j.out
# CPU only: no --gres needed. Probes are ridge/diff-in-means on ~200 rows.
#
# usage (from the repo root):
#   sbatch fcp/run_probes.slurm sycophancy
#   sbatch fcp/run_probes.slurm dishes   --condition backdoored
#   sbatch fcp/run_probes.slurm dishes   --condition clean
#   sbatch fcp/run_probes.slurm dishes   --diff backdoored clean
#   sbatch fcp/run_probes.slurm dosage
set -euo pipefail
AXIS="${1:?usage: sbatch fcp/run_probes.slurm <axis> [extra args]}"; shift
TAG="$AXIS$(printf '_%s' "$@" | tr ' ' '_' | tr -s '_')"
source .venv/bin/activate
mkdir -p fcp/logs fcp/results
python fcp/fcp_probes.py --dir "fcp/activations/$AXIS" --out "fcp/results/$TAG" "$@"
