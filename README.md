# Visual Intelligence: Evaluating and Aligning Models on Visual Illusions

This repository contains the codebase for evaluating artificial vision models (CNNs, ViTs, CLIP, DINOv2) and Vision-Language Models (LLaVA, Qwen-VL) on visual illusions. It includes tools to generate parametric stimuli, perform linear probing/zero-shot evaluations, calculate human-alignment scores, and train VLMs using baseline DPO and **Symmetric Polarity-Inverted Direct Preference Optimization (DPO)**.

---

## 🛠️ Installation & Setup

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repository_url> visual-intelligence
   cd visual-intelligence
   ```

2. **Set up a virtual environment and install packages**:
   The required packages and their exact version numbers are specified in [requirements.txt](file:///Users/jules/Desktop/visual-intelligence/requirements.txt). Install them via:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

---

## 🚀 Step-by-Step Replication Instructions

Follow these commands in order to reproduce our results:

### 1) Pre-generate All Parametric Stimuli
Generate the synthetic illusion datasets locally:
```bash
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from pathlib import Path
from src.stimuli.geometric import MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator
from src.stimuli.color import SimultaneousContrastGenerator, WhiteIllusionGenerator
from src.stimuli.angle import ZollnerGenerator, PoggendorffGenerator
from src.stimuli.motion import ScintillatingGridGenerator, RotatingSnakesGenerator

stimuli_dir = Path("data/stimuli")
for gen_cls in [
    MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator,
    SimultaneousContrastGenerator, WhiteIllusionGenerator,
    ZollnerGenerator, PoggendorffGenerator,
    ScintillatingGridGenerator, RotatingSnakesGenerator,
]:
    print(f"Generating {gen_cls.__name__}...")
    gen_cls().generate_dataset(stimuli_dir)
print("Stimuli generation completed!")
EOF
```

### 2) Run the Proof of Concept (PoC)
Runs the quickstart pipeline (ResNet-50 linear probe vs CLIP zero-shot on Müller-Lyer):
```bash
python experiments/poc.py --device mps
```
*Outputs (figures, curve parameters, HEAS metrics) are saved to `results/poc/`.*

### 3) Run the Full Model Sweep
Sweeps all models (CNNs, ViTs, CLIP, DINOv2, LLaVA, Qwen-VL) across all generated visual illusions:
```bash
python experiments/full_eval.py --device mps --skip-generate
```
*Outputs are saved under `results/full/`. (Note: To evaluate impossible/external datasets, follow instructions in [DATA_SETUP.md](file:///Users/jules/Desktop/visual-intelligence/DATA_SETUP.md)).*

### 4) Run Baseline vs. Symmetric Polarity DPO Training

*   **Train Baseline (Classic) DPO**:
    ```bash
    python src/rl/dpo_train.py \
        --output_dir results/classic_dpo \
        --dataset_path data/rl/dataset.jsonl \
        --max_steps 1000
    ```
*   **Train Symmetric Polarity DPO**:
    ```bash
    python src/rl/sym_mpo_train.py \
        --output_dir results/rl_alignment \
        --dataset_path data/rl/dataset.jsonl \
        --max_steps 1000
    ```
    *Training parameters are defined in [configs/symmetric_dpo.yaml](file:///Users/jules/Desktop/visual-intelligence/configs/symmetric_dpo.yaml).*

### 5) Run Cluster Execution (SLURM)
To run experiments or training on a SLURM cluster:
```bash
sbatch sbatch/run_symmetric_dpo.sbatch
sbatch sbatch/run_full_eval.sbatch
```

### 6) Run Saliency Visualizations
Generates saliency maps (Base LLaVA vs. DPO-aligned LLaVA):
```bash
python experiments/visualize_vlm_dpo.py
```
*Outputs are saved under `results/visualizations/`.*

### 7) Run Verification Tests
```bash
pytest tests/
```

---

## 📁 File Hierarchy & Descriptions

```
visual-intelligence/
├── configs/                             # Configuration files
│   ├── experiments.yaml                 # Configures dataset sweeps, baseline human rates, and ceiling thresholds
│   ├── models.yaml                      # Configures prober model details and probing hyperparameters
│   ├── stimuli.yaml                     # Configures construction settings (size, spacing, lines) for generated stimuli
│   └── symmetric_dpo.yaml               # Hyperparameters for Symmetric Polarity DPO training
│
├── experiments/                         # Executable pipelines and evaluation scripts
│   ├── poc.py                           # Week 1 Proof of Concept (ResNet-50 vs CLIP on Müller-Lyer)
│   ├── full_eval.py                     # Main sweep evaluating all models on all visual illusions
│   ├── full_eval_900.py                 # Variant of the full evaluation suite for 900 stimulus pairs
│   ├── analysis.py                      # Computes Grad-CAM and Attention Rollout on Müller-Lyer
│   ├── visualize_saliency.py            # Generates saliency maps for CNNs and ViTs across illusions
│   ├── visualize_vlm_dpo.py             # Generates saliency overlays for Base LLaVA vs DPO-aligned LLaVA
│   ├── occlusion_vlm_dpo.py             # Script to evaluate DPO alignment under image occlusion
│   ├── extra_dpo_experiments.py         # Performs additional DPO training runs and benchmarks
│   ├── post_dpo_eval.py                 # Evaluates model checkpoints generated during DPO training
│   └── probe_reasoning.py               # Script querying VLMs to probe visual reasoning accuracy
│
├── sbatch/                              # SLURM cluster submission templates
│   ├── run_dpo.sbatch                   # Runs classic DPO training baseline
│   ├── run_eval_checkpoint_900.sbatch   # Evaluates VLM checkpoint on 900 pairs
│   ├── run_eval_dpo.sbatch              # Evaluates classic DPO checkpoints
│   ├── run_eval_sym_mpo.sbatch          # Evaluates Symmetric DPO checkpoints
│   ├── run_extra_dpo.sbatch             # Runs additional DPO experiments
│   ├── run_full_eval.sbatch             # Runs full evaluation sweep
│   ├── run_occlusion_dpo.sbatch         # Runs DPO occlusion evaluations
│   ├── run_probe.sbatch                 # Trains and evaluates linear probes
│   ├── run_symmetric_dpo.sbatch         # Trains VLM using Symmetric Polarity DPO
│   └── run_visualize_dpo.sbatch         # Evaluates and visualizes DPO saliency changes
│
├── src/                                 # Main package source code
│   ├── stimuli/                         # Visual stimuli generators
│   │   ├── __init__.py                  # Exports generator registry classes
│   │   ├── base.py                      # Defines abstract StimulusGenerator and StimulusPair datatypes
│   │   ├── geometric.py                 # Generates Müller-Lyer, Ponzo, and Ebbinghaus geometric illusions
│   │   ├── color.py                     # Generates Simultaneous Contrast and White's Illusion color illusions
│   │   ├── angle.py                     # Generates Zöllner and Poggendorff angle illusions
│   │   ├── motion.py                    # Generates Scintillating Grid and Rotating Snakes motion illusions
│   │   └── impossible.py                # External loaders for IllusionVQA and HallusionBench
│   ├── models/                          # Model wrappers
│   │   ├── __init__.py                  # Registers model probers
│   │   ├── base.py                      # Abstract ModelProber base class
│   │   ├── cnn.py                       # Wrappers for ResNet and ConvNeXt
│   │   ├── vit.py                       # Wrappers for ViT-B and ViT-L
│   │   ├── contrastive.py               # Wrappers for CLIP and DINOv2
│   │   └── vlm.py                       # Wrappers for LLaVA and Qwen-VL
│   ├── probing/                         # Probing protocols
│   │   ├── __init__.py                  # Exports probing workflows
│   │   ├── linear_probe.py              # Probes representations using trained linear classifiers
│   │   ├── zero_shot.py                 # Performs zero-shot text/image matching
│   │   ├── vlm_protocol.py              # Executes multi-choice prompts on VLMs with option shuffling
│   │   └── probe_data.py                # Generates clean training/validation data for probes
│   ├── metrics/                         # Evaluation and alignment metrics
│   │   ├── __init__.py                  # Exports metrics
│   │   ├── heas.py                      # Computes Human Error Alignment Score (HEAS)
│   │   ├── psychometric.py              # Fits psychometric curves (sigmoid thresholding) and Spearman correlation
│   │   └── diagnostics.py               # Outputs prediction distributions and verification accuracies
│   └── rl/                              # Reinforcement learning / Preference alignment
│       ├── __init__.py                  # Exports RL training modules
│       ├── loss.py                      # 4-term Symmetric Polarity Preference Loss implementation
│       ├── collator.py                  # Collate function preparing VLM inputs under polarity inversion
│       ├── dataset.py                   # Data loaders for VLM instructions
│       ├── polarity_dataset.py          # Formats positive/negative preference items
│       ├── dpo_train.py                 # Runs classic baseline DPO training
│       ├── sym_mpo_train.py             # Runs Symmetric Polarity-Inverted DPO training
│       └── push_to_hub.py               # Utility script pushing trained weights to Hugging Face Hub
│
├── tests/                               # Unit test suite
│   ├── smoke_test__sym_mpo.py           # Verifies DPO loss calculation and model weight unfreezing
│   ├── test_metrics.py                  # Tests HEAS calculation, sigmoid fitting, and Spearman correlation
│   ├── test_polarity_loss.py            # Tests symmetric loss terms and polarity-inverted margins
│   ├── test_probing.py                  # Tests linear probe loaders and zero-shot probers
│   └── test_stimuli.py                  # Tests stimulus generator image sizes and outputs
│
├── requirements.txt                     # Main project package requirements
└── DATA_SETUP.md                        # Documentation on downloading and preparing external datasets
```
