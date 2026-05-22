#!/bin/bash
#SBATCH --job-name=probe_reasoning
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

module load gcc/11.3.0
module load cuda/12.1.1

if [ -n "$SLURM_SUBMIT_DIR" ]; then
	cd "$SLURM_SUBMIT_DIR"
fi

source .venv/bin/activate
python -u experiments/probe_reasoning.py
