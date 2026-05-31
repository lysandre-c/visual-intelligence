#!/bin/bash
#SBATCH --job-name=vi-eval-sym-mpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_sym_mpo_%j.log
#SBATCH --error=logs/eval_sym_mpo_%j.err

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

# ── Ensure directories exist ───────────────────────────────────────────
mkdir -p results/full
mkdir -p logs
mkdir -p results/hf_cache
mkdir -p results/tmp

export HF_HOME="$PWD/results/hf_cache"
export HF_DATASETS_CACHE="$PWD/results/hf_cache"
export TMPDIR="$PWD/results/tmp"

# Reduce CUDA memory fragmentation (recommended by PyTorch docs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 0: Clear ONLY stale symDPO artifacts so they are regenerated ──────
# full_eval.py skips any stimulus whose result JSON already exists. We must
# remove the symDPO outputs (NOT the cached llava_1.5 base results) to force
# a fresh run with the current adapter.
echo "Clearing stale symDPO results (base llava_1.5 results are kept)..."
rm -f results/full/llava_symDPO_*_results.json
rm -rf results/full/audit/llava_symDPO

# ── Step 1: Evaluate the DPO-trained model (same pipeline as run_full_eval) ─
echo "Starting full evaluation of DPO-trained LLaVA model (final)..."

python experiments/full_eval.py \
    --device cuda \
    --verbose \
    --skip-generate \
    --models llava_symDPO

echo "Evaluation complete. Results are in results/full/"

# ── Step 1b: Rich before/after comparison (base vs symDPO) ─────────────────
echo "Running rich before/after comparison (control fails, flips, collapse)..."
python experiments/compare_sym_mpo.py
echo "Comparison written to results/full/compare_sym_mpo/"

# ── Step 2: Post-DPO analysis (HEAS comparison + GradCAM) ──────────────────
echo "Starting post-DPO analysis (HEAS comparison, GradCAM, diagnostics)..."

python experiments/post_dpo_eval.py \
    --device cuda \
    --adapter-path results/rl_alignment/final \
    --n-gradcam 6

echo "Post-DPO analysis complete. Results in results/full/post_symMPO/"
