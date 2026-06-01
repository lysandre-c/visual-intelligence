#!/bin/bash
#SBATCH --job-name=vi-eval-sym-mpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/eval_sym_mpo_%j.log
#SBATCH --error=logs/eval_sym_mpo_%j.err

# A 7B fp16 model (~14GB) runs single-stream FASTER on ONE GPU than split across
# two (no layer-to-layer activation copies, no idle card). gpu:1 is also easier
# to schedule. merge_and_unload needs transient headroom but fits 40GB+ cards.

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

mkdir -p results/full logs results/hf_cache results/tmp
export HF_HOME="$PWD/results/hf_cache"
export HF_DATASETS_CACHE="$PWD/results/hf_cache"
export TMPDIR="$PWD/results/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 0: Clear ONLY stale symDPO artifacts (keep cached base) ───────────
echo "Clearing stale symDPO results (base llava_1.5 results are kept)..."
rm -f results/full/llava_symDPO_*_results.json
rm -rf results/full/audit/llava_symDPO

# ── Step 1: Measure real per-stimulus speed and derive a per-type cap ───────
# Times the merged symDPO model on 12 stimuli at the current settings (8 tokens,
# 1 GPU), then writes a recommended --max-per-type so the eval fits ~3.5h. This
# guarantees ALL illusion types complete (broad before/after coverage) instead
# of risking a timeout mid-scan.
echo "Benchmarking symDPO inference speed ..."
python experiments/bench_symdpo.py --n 12 --budget-hours 3.5 \
    --cap-out results/tmp/symdpo_maxpertype.txt

CAP=$(cat results/tmp/symdpo_maxpertype.txt 2>/dev/null || echo 80)
echo "Using --max-per-type=$CAP"

# ── Step 2: Evaluate symDPO (time-boxed so the compare step always runs) ────
echo "Evaluating symDPO (cap=$CAP stimuli/type, hard time box 230m) ..."
timeout 230m python experiments/full_eval.py \
    --device cuda \
    --verbose \
    --skip-generate \
    --max-per-type "$CAP" \
    --models llava_symDPO \
  || echo "NOTE: eval hit its time box; proceeding to compare on completed types."

echo "Evaluation phase done. symDPO results in results/full/"

# ── Step 3: Rich before/after comparison (ALWAYS runs, on shared stimuli) ───
echo "Running before/after comparison (base vs symDPO) ..."
python experiments/compare_sym_mpo.py
echo "Comparison written to results/full/compare_sym_mpo/"

# ── Step 4: GradCAM saliency (expendable; time-boxed) ───────────────────────
# Qualitative base-vs-symDPO attention on Müller-Lyer. NOTE: post_dpo also emits
# a heas_comparison.csv whose BASE column is unreliable (base has no control
# logs) — use results/full/compare_sym_mpo/ as the authoritative comparison.
echo "Generating GradCAM saliency (expendable, time box 35m) ..."
timeout 35m python experiments/post_dpo_eval.py \
    --device cuda \
    --adapter-path results/rl_alignment/final \
    --n-gradcam 4 \
  || echo "NOTE: GradCAM skipped/cut by time box — core comparison already saved."

echo "All done."
