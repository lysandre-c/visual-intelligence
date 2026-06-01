#!/bin/bash
#SBATCH --job-name=vi-quick-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/quick_eval_%j.log
#SBATCH --error=logs/quick_eval_%j.err

if [ -n "$SLURM_SUBMIT_DIR" ]; then
	cd "$SLURM_SUBMIT_DIR"
fi

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python test_quick_eval.py
