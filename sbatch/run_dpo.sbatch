#!/bin/bash
#SBATCH --job-name=vi-rl-alignment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/rl_alignment_%j.log
#SBATCH --error=logs/rl_alignment_%j.err

# Load modules (adjust based on your cluster's available modules)
# module load gcc/11.3.0 python/3.11.2 cuda/11.8.0

# 1. Identify project root and move there 
if [ -n "$SLURM_SUBMIT_DIR" ]; then
	cd "$SLURM_SUBMIT_DIR"
else
	SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null &&
     pwd )"
	cd "$( dirname "$SCRIPT_DIR" )"
fi
echo "Running from project root: $PWD"

# 2. Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found in $PWD. Ensure you have created the virtual environment."
    exit 1
fi

# 3. Ensure results and logs directories exist
mkdir -p results/rl_alignment
mkdir -p logs
mkdir -p data/rl
mkdir -p results/hf_cache
mkdir -p results/tmp

export HF_HOME="$PWD/results/hf_cache"
export HF_DATASETS_CACHE="$PWD/results/hf_cache"
export TMPDIR="$PWD/results/tmp"

# 4. Generate the multimodal DPO dataset
echo "Generating DPO dataset..."
python src/rl/dataset.py

module load gcc/11.3.0
module load cuda/12.1.1

# 5. Run DPO training
# Using default hyperparameters, can be overridden here
echo "Starting DPO training..."
./.venv/bin/python src/rl/dpo_train.py \
    --output_dir results/rl_alignment \
    --dataset_path data/rl/dataset.jsonl \
    --beta 0.1 \
    --max_steps 1000 \
    --save_steps 100 \
    --max_prompt_length 800 \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --gradient_checkpointing True \
    --bf16 True \
    --remove_unused_columns False \
    --report_to tensorboard

echo "RL alignment job complete. Results in results/rl_alignment/"
