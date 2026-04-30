# Data Setup

## Categories 1–4: Generated stimuli (no download needed)

Run once before any experiment to pre-generate all parametric stimuli:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from src.stimuli.geometric import MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator
from src.stimuli.color import SimultaneousContrastGenerator, WhiteIllusionGenerator
from src.stimuli.angle import ZollnerGenerator, PoggendorffGenerator
from src.stimuli.motion import ScintillatingGridGenerator, RotatingSnakesGenerator
from pathlib import Path

stimuli_dir = Path("data/stimuli")
for gen_cls in [
    MullerLyerGenerator, PonzoGenerator, EbbinghausGenerator,
    SimultaneousContrastGenerator, WhiteIllusionGenerator,
    ZollnerGenerator, PoggendorffGenerator,
    ScintillatingGridGenerator, RotatingSnakesGenerator,
]:
    gen = gen_cls()
    print(f"Generating {gen.illusion_type} …")
    gen.generate_dataset(stimuli_dir)
print("Done.")
EOF
```

Alternatively, `experiments/poc.py` triggers Müller-Lyer generation automatically on first run.

---

## Category 5: External datasets (manual download)

### IllusionVQA (Shahgir et al., COLM 2024)

The dataset is split across two HuggingFace repos (comprehension + localization).
We use the comprehension split.

```bash
pip install huggingface_hub datasets

python - <<'EOF'
from datasets import load_dataset
import json, os
from pathlib import Path

out = Path("data/external/illusion_vqa")
img_dir = out / "images"
img_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("csebuetnlp/illusionVQA-Comprehension", split="test")

metadata = []
for item in ds:
    sid = str(item["id"])
    img_path = img_dir / f"{sid}_illusion.png"
    item["image"].save(img_path)
    metadata.append({
        "id": sid,
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "human_error": "",
    })

with open(out / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved {len(metadata)} items to {out}")
EOF
```

Expected layout after running the script:

```
data/external/illusion_vqa/
    images/           ← PNG files named <id>_illusion.png
    metadata.json     ← list of {id, question, answer, human_error}
```

### HallusionBench (Guan et al., CVPR 2024)

The repo contains the JSON but images are distributed separately as a zip on Google Drive.

**Step 1 — clone the repo (gets the JSON):**

```bash
git clone https://github.com/tianyi-lab/HallusionBench.git data/external/hallusion_bench
```

**Step 2 — download and unzip the images:**

Download `hallusion_bench.zip` from:
https://drive.google.com/file/d/1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0/view?usp=sharing

Then unzip it inside the cloned folder:

```bash
# after downloading hallusion_bench.zip manually:
unzip hallusion_bench.zip -d data/external/hallusion_bench/
```

Expected layout after both steps:

```
data/external/hallusion_bench/
    HallusionBench.json          ← official annotation file (from git clone)
    hallusion_bench/             ← unzipped image folder
        VD/
            illusion/
                0_0.png …
        VS/
            …
```

The loader reads `filename` fields like `./hallusion_bench/VD/illusion/0_0.png` from the JSON,
so images must be at `data/external/hallusion_bench/hallusion_bench/<category>/…`.

---

## Human baselines

The `human_baselines` rates in `configs/experiments.yaml` are approximate
population-level values from the cited literature. Store the source CSVs in
`data/human_baselines/` for traceability, then update the YAML values directly
if you find more precise per-illusion rates.

---

## Enabling external datasets in the full evaluation

Once downloaded, uncomment these lines in `configs/experiments.yaml`:

```yaml
# - category: impossible
#   illusion_type: illusion_vqa
# - category: impossible
#   illusion_type: hallusion_bench
```

On first run, `ExternalDatasetLoader` auto-builds and caches a `manifest.json`
for each dataset under `data/external/<dataset>/`.
