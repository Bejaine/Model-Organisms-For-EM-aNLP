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

# $1 = experiment dir name (e.g. temporal_dense_2027), $2 = condition or "diff", $3 = out subdir
EXPDIR="$1"; COND="$2"; OUTDIR="$3"
mkdir -p "fcp/results/temporal/$OUTDIR"

if [ "$COND" = "diff" ]; then
  $VENV/bin/python fcp/fcp_probes.py --dir "fcp/activations/$EXPDIR" \
    --severity "fcp/$EXPDIR/severity.parquet" --s-col rate \
    --diff adapted clean --save-scores --out "fcp/results/temporal/$OUTDIR"
else
  $VENV/bin/python fcp/fcp_probes.py --dir "fcp/activations/$EXPDIR" \
    --severity "fcp/$EXPDIR/severity.parquet" --s-col rate \
    --condition "$COND" --save-scores --out "fcp/results/temporal/$OUTDIR"
fi
echo "PROBE JOB COMPLETE: $EXPDIR $COND"
