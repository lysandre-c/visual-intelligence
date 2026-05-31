#!/bin/bash
#SBATCH --job-name=vi-eval-sym-mpo-2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=05:00:00
#SBATCH --output=logs/eval_sym_mpo_2gpu_%j.log
#SBATCH --error=logs/eval_sym_mpo_2gpu_%j.err

# DATA-PARALLEL across 2 GPUs: a FULL copy of the 7B model on each GPU (it fits
# in ~14GB), each process evaluating a disjoint set of categories. This ~halves
# wall-clock — unlike device_map="auto" pipeline split, which idles a GPU for
# single-stream generation. The two processes write different per-type result
# files, so there is no collision on the data the comparison reads.

if [ -n "$SLURM_SUBMIT_DIR" ]; then
	cd "$SLURM_SUBMIT_DIR"
else
	SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
	cd "$( dirname "$SCRIPT_DIR" )"
fi
echo "Running from project root: $PWD"

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

# ── Step 1: Benchmark on 1 GPU, derive a per-type cap for the 2-GPU split ───
echo "Benchmarking symDPO inference speed ..."
CUDA_VISIBLE_DEVICES=0 python experiments/bench_symdpo.py \
    --n 12 --budget-hours 3.5 --gpus 2 \
    --cap-out results/tmp/symdpo_maxpertype.txt
CAP=$(cat results/tmp/symdpo_maxpertype.txt 2>/dev/null || echo 80)
echo "Using --max-per-type=$CAP per GPU worker"

# ── Step 2: Two parallel workers, one per GPU, disjoint categories ──────────
# GPU0: geometric + angle (5 types)   |   GPU1: color + motion (4 types)
echo "Launching 2 data-parallel eval workers (time box 230m each) ..."
CUDA_VISIBLE_DEVICES=0 timeout 230m python experiments/full_eval.py \
    --device cuda --verbose --skip-generate --max-per-type "$CAP" \
    --models llava_symDPO --categories geometric angle \
  > logs/worker_gpu0_${SLURM_JOB_ID}.log 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 timeout 230m python experiments/full_eval.py \
    --device cuda --verbose --skip-generate --max-per-type "$CAP" \
    --models llava_symDPO --categories color motion \
  > logs/worker_gpu1_${SLURM_JOB_ID}.log 2>&1 &
PID1=$!

wait $PID0; echo "GPU0 worker exited ($?)"
wait $PID1; echo "GPU1 worker exited ($?)"
echo "Both workers done. symDPO results in results/full/ (see logs/worker_gpu*.log)"

# ── Step 3: Rich before/after comparison (on whatever completed) ───────────
echo "Running before/after comparison (base vs symDPO) ..."
python experiments/compare_sym_mpo.py
echo "Comparison written to results/full/compare_sym_mpo/"

# ── Step 4: GradCAM saliency (expendable; time-boxed) ───────────────────────
echo "Generating GradCAM saliency (expendable, time box 35m) ..."
timeout 35m python experiments/post_dpo_eval.py \
    --device cuda --adapter-path results/rl_alignment/final --n-gradcam 4 \
  || echo "NOTE: GradCAM skipped/cut by time box — core comparison already saved."

echo "All done."
