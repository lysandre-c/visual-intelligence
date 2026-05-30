#!/bin/bash
#SBATCH --job-name=test-merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/test_merge_%j.log
#SBATCH --error=logs/test_merge_%j.err

source .venv/bin/activate
python test_merge.py
