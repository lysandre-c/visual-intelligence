# Do Vision Models Fail Like Humans?

**CS-503 Visual Intelligence — EPFL**

Imane Oujja, Jules Herrscher, Matisse Van Schalkwijk, Lysandre Costes

Project webpage (GitHub Pages): enable **Settings → Pages → Source: GitHub Actions**, then visit the deployed URL after the workflow runs. Local preview: `cd website && python -m http.server 8080`.

---

## Overview

We measure **directional** alignment between human and model perceptual errors on visual illusions using the **Human Error Alignment Score (HEAS)**:

`HEAS(m, c) = 1 - | p_model^illusory(m, c) - p_human^illusory(c) |`

Nine parametric illusion generators plus external VQA datasets; eight models (CNN, ViT, CLIP, DINOv2, LLaVA base, LLaVA + DPO, SymMPO); linear probe, zero-shot, and VLM probing protocols. The codebase also supports **Symmetric Polarity-Inverted DPO (SymMPO)** training.

---

## Requirements

- Python **3.10+**
- CUDA GPU recommended for full eval and DPO (Apple **MPS** works for smaller runs)
- See [requirements.txt](requirements.txt) for pinned dependencies (tested with `torch>=2.2`, `transformers~=4.46`, `trl>=0.8`, `peft>=0.10`)

```bash
git clone https://github.com/lysandre-c/visual-intelligence.git
cd visual-intelligence
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

External datasets for impossible/scene illusions: see [DATA_SETUP.md](DATA_SETUP.md).

---

## Reproduce main results

### 1. Generate stimuli (optional if cached)

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
print("Done.")
EOF
```

### 2. Proof-of-concept (Müller-Lyer × ResNet + CLIP)

```bash
python experiments/poc.py --device mps
# Outputs: results/poc/
```

### 3. Full evaluation (all models × illusions → HEAS table)

```bash
python experiments/full_eval.py --device cuda
# Or reuse cached stimuli:
python experiments/full_eval.py --device cuda --skip-generate
```

**Outputs** under `results/full/`:

| File | Description |
|------|-------------|
| `heas_table.csv` | Category × model HEAS matrix |
| `spearman_table.csv` | Per-illusion Spearman ρ |
| `diagnostics_table.csv` | Accuracy and label distributions |
| `all_results.json` | Per-stimulus probe records |
| `figures/heas_heatmap.png` | Heatmap figure |
| `figures/psychometric_*.png` | Psychometric curves |

### 4. DPO / SymMPO alignment (optional)

**Classic DPO:**

```bash
python src/rl/dataset.py
python src/rl/dpo_train.py --output_dir results/classic_dpo --dataset_path data/rl/dataset.jsonl
```

**Symmetric Polarity DPO (SymMPO):**

```bash
python src/rl/sym_mpo_train.py --output_dir results/rl_alignment --dataset_path data/rl/dataset.jsonl
# Hyperparameters: configs/symmetric_dpo.yaml
```

Re-evaluate aligned VLMs:

```bash
python experiments/full_eval.py --device cuda --models llava_1.5 llava_1.5_dpo
```

**SLURM cluster:** `sbatch sbatch/run_full_eval.sbatch`, `sbatch sbatch/run_symmetric_dpo.sbatch`, etc.

### 5. Website data export

After a full eval, refresh the tracked export inputs (or edit `website/source/` directly):

```bash
cp results/full/heas_table.csv results/full/diagnostics_table.csv results/full/all_results.json website/source/
python scripts/export_website_data.py
# Writes website/data/*.json; figures and saliency PNGs live under website/assets/
```

### 6. Saliency / DPO visualizations

```bash
python experiments/visualize_saliency.py
python experiments/visualize_vlm_dpo.py
python experiments/extra_dpo_experiments.py
```

---

## File hierarchy

```
visual-intelligence/
├── configs/
│   ├── experiments.yaml      # Models, stimuli list, human baselines
│   ├── models.yaml           # Model registry and probe hyperparameters
│   ├── stimuli.yaml          # Generator defaults
│   └── symmetric_dpo.yaml    # SymMPO training hyperparameters
├── src/
│   ├── stimuli/              # Parametric illusion generators + external loader
│   ├── models/               # ResNet, ViT, CLIP, DINOv2, LLaVA probers
│   ├── probing/              # Linear probe, zero-shot, VLM protocols
│   ├── metrics/              # HEAS, psychometric, diagnostics
│   ├── analysis/             # Plots, Grad-CAM, attention
│   └── rl/                   # DPO + SymMPO training
├── experiments/
│   ├── poc.py                # Week-1 PoC
│   ├── full_eval.py          # Main HEAS evaluation
│   └── visualize_*.py        # Figure scripts
├── scripts/
│   └── export_website_data.py
├── website/                  # Project webpage (GitHub Pages)
│   ├── source/               # HEAS tables + all_results.json for export
│   ├── data/                 # JSON consumed by the page
│   └── assets/               # Figures, saliency overlays, stimulus thumbnails
├── sbatch/                   # SLURM job templates
├── data/                     # Generated stimuli (not in git)
└── results/                  # Experiment outputs (not in git)
```

---

## Tests

```bash
pytest tests/
```

---

## Citation

If you use this code, please cite the CS-503 project report and repository URL.
