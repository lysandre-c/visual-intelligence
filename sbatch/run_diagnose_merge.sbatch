#!/bin/bash
#SBATCH --job-name=vi-diag-merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=logs/diag_merge_%j.log
#SBATCH --error=logs/diag_merge_%j.err

# IMPORTANT: --gres=gpu:2 deliberately matches run_eval_sym_mpo.sbatch so this
# reproduces the SAME device_map="auto" multi-GPU dispatch as the real eval.
# Running on 1 GPU could hide a merge_and_unload() no-op caused by sharding.

# ── Navigate to project root ───────────────────────────────────────────
if [ -n "$SLURM_SUBMIT_DIR" ]; then
	cd "$SLURM_SUBMIT_DIR"
else
	SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
	cd "$( dirname "$SCRIPT_DIR" )"
fi
echo "Running from project root: $PWD"

# ── Activate virtual environment ────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found in $PWD."
    exit 1
fi

mkdir -p logs results/hf_cache results/tmp
export HF_HOME="$PWD/results/hf_cache"
export HF_DATASETS_CACHE="$PWD/results/hf_cache"
export TMPDIR="$PWD/results/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Running merge diagnostic (matches real eval: gpu:2, device_map=auto) ==="
python experiments/diagnose_eval_merge.py
echo "=== Diagnostic complete ==="
