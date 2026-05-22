#!/bin/bash
#SBATCH --job-name=extra_dpo
#SBATCH --output=logs/extra_dpo_%j.out
#SBATCH --error=logs/extra_dpo_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module load gcc/11.3.0
module load cuda/12.1.1

if [ -n "$SLURM_SUBMIT_DIR" ]; then
	cd "$SLURM_SUBMIT_DIR"
fi

source .venv/bin/activate
python -u experiments/extra_dpo_experiments.py
