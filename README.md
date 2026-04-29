# Visual Intelligence - Quickstart

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) (Optional) Pre-generate all synthetic stimuli

```bash
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from pathlib import Path
from src.stimuli.geometric import MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator
from src.stimuli.color import SimultaneousContrastGenerator, WhiteIllusionGenerator
from src.stimuli.angle import ZollnerGenerator, PoggendorffGenerator
from src.stimuli.motion import ScintillatingGridGenerator, FraserSpiralGenerator

stimuli_dir = Path("data/stimuli")
for gen_cls in [
    MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator,
    SimultaneousContrastGenerator, WhiteIllusionGenerator,
    ZollnerGenerator, PoggendorffGenerator,
    ScintillatingGridGenerator, FraserSpiralGenerator,
]:
    gen_cls().generate_dataset(stimuli_dir)
print("Done.")
EOF
```

## 3) Run experiments

```bash
# Week 1 proof-of-concept
python experiments/poc.py --device mps

# Full evaluation (all categories/models configured)
python experiments/full_eval.py --device mps

# Reuse existing generated stimuli
python experiments/full_eval.py --device mps --skip-generate
```

