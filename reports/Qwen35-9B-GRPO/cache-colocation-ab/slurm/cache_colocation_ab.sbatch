#!/bin/bash
#SBATCH --job-name=q35-cache-ab
#SBATCH --partition=gre
#SBATCH --account=professors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:45:00
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
export PYTHONPATH="$OCCAM_FROZEN_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
cd "$OCCAM_FROZEN_SOURCE"
exec "$OCCAM_RUNTIME/bin/python" -u "$OCCAM_PROBE/cache_colocation_ab.py" \
  --history "$OCCAM_HISTORY" --output "$OCCAM_OUTPUT/cache-ab-$SLURM_JOB_ID" \
  --encoder "$OCCAM_ENCODER" --retrieval-compat "$OCCAM_RUNTIME/retrieval_protocol_compat"
