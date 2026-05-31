#!/bin/bash
#SBATCH --job-name=vi-test-gradcam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_gradcam_%j.log
#SBATCH --error=logs/test_gradcam_%j.err

# Validates the GradCAM fix on BOTH the base and PEFT/DPO model. A 0 exit code
# means experiments/post_dpo_eval.py's GradCAM step will succeed.

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

echo "Running GradCAM validation (base + DPO) ..."
python test_gradcam.py
echo "Exit code: $?"
