#!/bin/bash
#SBATCH --job-name=vi-rerun-base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/rerun_base_%j.log
#SBATCH --error=logs/rerun_base_%j.err

# Re-EVAL (not re-train) base llava_1.5 with the CURRENT code so it gains:
#   - control-image logs   -> can be control-gated identically to symDPO
#   - single-trial scoring -> matches symDPO's estimator (kills soft-vs-hard)
#   - corrected Müller-Lyer mapping -> no swap hack needed
# ~1h at the measured 0.77 s/it. Then re-run the comparison, which auto-detects
# the control-logged base and switches the before/after to a proper GATED metric.

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

# ── Step 0: Back up the OLD incompatible base results (do not delete) ──────
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="results/full/_base_old_${STAMP}"
mkdir -p "$BACKUP"
echo "Backing up old base result JSONs to $BACKUP ..."
mv results/full/llava_1.5_*_results.json "$BACKUP"/ 2>/dev/null || echo "(no old base JSONs to move)"
# audit dir too (old base audit lacks control)
[ -d results/full/audit/llava_1.5 ] && mv results/full/audit/llava_1.5 "$BACKUP/audit_llava_1.5" 2>/dev/null || true

# ── Step 1: Re-eval base with the current fast pipeline (control + 1-shot) ──
echo "Re-evaluating base llava_1.5 (~1h) ..."
timeout 100m python experiments/full_eval.py \
    --device cuda --verbose --skip-generate --models llava_1.5 \
  || echo "NOTE: base re-eval hit its time box; proceeding to compare on completed types."

# ── Step 2: Comparison — now GATED for both (base has control logs) ─────────
echo "Running before/after comparison (now control-gated) ..."
python experiments/compare_sym_mpo.py
echo "Done. See results/full/compare_sym_mpo/ (GATED before/after)."
