#!/bin/bash
#SBATCH --job-name=vi-bench-sym-mpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_sym_mpo_%j.log
#SBATCH --error=logs/bench_sym_mpo_%j.err

# Standalone timing probe: loads the merged symDPO model on ONE GPU, times 12
# stimuli at the real eval settings (8 tokens), and prints how long the full
# scan would take on 1 GPU vs 2 GPUs (data-parallel), plus the per-type cap that
# fits the budget. Run this FIRST to decide how many hours / GPUs to request.

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

mkdir -p logs results/hf_cache results/tmp
export HF_HOME="$PWD/results/hf_cache"
export HF_DATASETS_CACHE="$PWD/results/hf_cache"
export TMPDIR="$PWD/results/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/bench_symdpo.py --n 12 --budget-hours 3.5 --gpus 1 \
    --cap-out results/tmp/symdpo_maxpertype.txt

echo "Benchmark done. See the 1-GPU vs 2-GPU projection table above."
