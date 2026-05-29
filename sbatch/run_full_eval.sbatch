#!/bin/bash
#SBATCH --job-name=vi-full-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_eval_%j.log
#SBATCH --error=logs/full_eval_%j.err

# Exit immediately if a command exits with a non-zero status.
set -e

# Load modules (adjust based on your cluster's available modules)
# module load gcc/11.3.0 python/3.11.2 cuda/11.8.0

# Activate virtual environment
source .venv/bin/activate

export HF_HOME="$PWD/results/hf_cache"

# Create output directories
mkdir -p results/full
mkdir -p logs

# Run the full evaluation
# This will:
# 1. Generate stimuli
# 2. Run all models on all illusions
# 3. Compute HEAS metrics
# 4. Generate the HEAS table heatmap
python experiments/full_eval.py --device cuda --verbose --skip-generate --output-dir results/full_final

# Generate saliency heatmaps for presentation
# echo "Generating saliency visualizations..."
# python experiments/visualize_saliency.py --device cuda

echo "Evaluation and visualization complete. Results are in results/full/ and results/visualizations/"
