#!/bin/bash
#SBATCH --job-name=q35-grpo-stability
#SBATCH --partition=gre
#SBATCH --account=professors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=48:00:00
set +x
set -euo pipefail
umask 077
base="$(dirname "$(dirname "$STORE")")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
export TMPDIR="$base/q35tmp" TRITON_CACHE_DIR="$STORE/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$STORE/torchinductor-cache" VLLM_CACHE_ROOT="$STORE/vllm-cache"
export XDG_CACHE_HOME="$STORE/xdg-cache" TORCH_HOME="$STORE/torch-cache"
export PYTHONPYCACHEPREFIX="$STORE/runtime-pycache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$STORE/python/fla-cuda-site"
cd "$Q35_SOURCE"
exec "$base/envs/successful_phase05_20261007/bin/python" -u -m src.stability_controller
