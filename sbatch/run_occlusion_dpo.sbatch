#!/bin/bash
#SBATCH --job-name=occlusion_dpo
#SBATCH --output=logs/occlusion_dpo_%j.out
#SBATCH --error=logs/occlusion_dpo_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Load modules
module load gcc/11.3.0
module load cuda/12.1.1

# Activate virtual environment
if [ -n "$SLURM_SUBMIT_DIR" ]; then
	cd "$SLURM_SUBMIT_DIR"
fi

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found"
    exit 1
fi

echo "Running DPO VLM Occlusion Analysis..."
python experiments/occlusion_vlm_dpo.py
echo "Occlusion analysis complete."
