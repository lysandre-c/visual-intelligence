#!/bin/bash
#SBATCH --job-name=vi-sym-polarity-dpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/sym_polarity_dpo_%j.log
#SBATCH --error=logs/sym_polarity_dpo_%j.err

# ── Navigate to project root ───────────────────────────────────────────
if [ -n "$SLURM_SUBMIT_DIR" ]; then
	cd "$SLURM_SUBMIT_DIR"
else
	SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null &&
     pwd )"
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
mkdir -p results/rl_alignment
mkdir -p logs
mkdir -p data/rl
mkdir -p results/hf_cache
mkdir -p results/tmp

export HF_HOME="$PWD/results/hf_cache"
export HF_DATASETS_CACHE="$PWD/results/hf_cache"
export TMPDIR="$PWD/results/tmp"

# Reduce CUDA memory fragmentation (recommended by PyTorch docs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Load modules (adjust for your cluster) ──────────────────────────────
# module load gcc/11.3.0
# module load cuda/12.1.1

# ── Hugging Face Hub Authentication ──────────────────────────────────────
# HF_TOKEN should be set in your environment (e.g. via ~/.bashrc or secrets).
# Do NOT hardcode tokens in scripts.
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN is not set. Hub push will fail if required."
fi


# ── Step 1: Generate the polarity-paired DPO dataset (skip if exists) ───
DATASET_FILE="data/rl/dataset.jsonl"
if [ -f "$DATASET_FILE" ]; then
    echo "Dataset already exists at $DATASET_FILE — skipping generation."
else
    echo "Generating Symmetric Polarity DPO dataset..."
    python src/rl/polarity_dataset.py \
        --max-per-illusion 60 \
        --control-ratio 0.2 \
        --seed 42
fi

# ── Step 2: Run Symmetric Polarity DPO training ────────────────────────
echo "Starting Symmetric Polarity DPO training..."
python src/rl/sym_mpo_train.py \
    --output_dir results/rl_alignment \
    --dataset_path data/rl/dataset.jsonl \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --beta 0.1 \
    --gamma 1.0 \
    --label_lambda 0.5 \
    --eta 0.1 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_length 1024 \
    --symbol_demo True \
    --option_shuffle True \
    --max_steps 1000 \
    --save_steps 100 \
    --logging_steps 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --gradient_checkpointing True \
    --bf16 True \
    --remove_unused_columns False \
    --report_to tensorboard \
    --push_to_hub_repo "Matisse6410/LlaVa-1.5-SDPO" \
    --resume_checkpoint results/rl_alignment/checkpoint-900

echo "Symmetric Polarity DPO complete. Results in results/rl_alignment/"

# ── Step 3: Evaluate the DPO-trained model (same pipeline as run_full_eval) ─
echo "Starting full evaluation of DPO-trained LLaVA model..."
mkdir -p results/full

python experiments/full_eval.py \
    --device cuda \
    --verbose \
    --skip-generate \
    --models llava_symDPO

echo "Evaluation complete. Results are in results/full/"

# ── Step 4: Post-DPO analysis (HEAS comparison + GradCAM) ──────────────────
echo "Starting post-DPO analysis (HEAS comparison, GradCAM, diagnostics)..."

python experiments/post_dpo_eval.py \
    --device cuda \
    --adapter-path results/rl_alignment/checkpoint-1000 \
    --n-gradcam 6

echo "Post-DPO analysis complete. Results in results/full/post_symMPO/"
