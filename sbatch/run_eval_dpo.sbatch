#!/bin/bash
#SBATCH --job-name=eval_dpo
#SBATCH --output=logs/eval_dpo_%j.out
#SBATCH --error=logs/eval_dpo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load modules
module load gcc/11.3.0
module load cuda/12.1.1

# Activate virtual environment
source .venv/bin/activate

# Evaluate DPO LLaVA
echo "Starting evaluation of DPO LLaVA..."
python experiments/full_eval.py --models llava_1.5_dpo
echo "Evaluation complete."
