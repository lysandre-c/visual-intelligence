import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "results" / "hf_cache"))
import torch
import shlex
import argparse
from transformers import HfArgumentParser, AutoProcessor, LlavaConfig, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model
from src.rl.dpo_train import ScriptArguments
from transformers import TrainingArguments

def _build_tiny_llava():
    from transformers import CLIPVisionConfig, LlamaConfig
    vision_cfg = CLIPVisionConfig(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_hidden_layers=2, image_size=336, patch_size=14)
    text_cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=2, num_hidden_layers=2, vocab_size=32064, max_position_embeddings=2048)
    llava_cfg = LlavaConfig(vision_config=vision_cfg, text_config=text_cfg, image_token_id=32000, vision_feature_layer=-1, vision_feature_select_strategy="default")
    model = LlavaForConditionalGeneration(llava_cfg)
    return model

print("Step 1: Building tiny model...")
model = _build_tiny_llava()

lora_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
policy_model = get_peft_model(model, lora_config)

print("Step 12: CLI Argument parsing check")
sbatch_args_str = """
    --output_dir results/rl_alignment
    --dataset_path data/rl/dataset.jsonl
    --model_name_or_path llava-hf/llava-1.5-7b-hf
    --beta 0.1
    --gamma 1.0
    --label_lambda 0.5
    --eta 0.1
    --lora_r 64
    --lora_alpha 16
    --lora_dropout 0.05
    --max_length 1024
    --symbol_demo True
    --option_shuffle True
    --max_steps 1000
    --save_steps 100
    --logging_steps 10
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 5e-5
    --weight_decay 0.01
    --gradient_checkpointing True
    --bf16 True
    --remove_unused_columns False
    --report_to tensorboard
    --push_to_hub_repo Matisse6410/LlaVa-1.5-SDPO
"""
args_list = shlex.split(sbatch_args_str)
parser = HfArgumentParser((ScriptArguments, TrainingArguments))
try:
    script_args, training_args = parser.parse_args_into_dataclasses(args=args_list)
except ValueError as e:
    if "bf16" in str(e):
        args_list.extend(["--use_cpu", "True"])
        script_args, training_args = parser.parse_args_into_dataclasses(args=args_list)
    else:
        raise
assert script_args.lora_r == 64
print("✓ dpo_train arguments parsed")

dataset_parser = argparse.ArgumentParser()
dataset_parser.add_argument("--max-per-illusion", type=int, default=60)
dataset_parser.add_argument("--control-ratio", type=float, default=0.2)
dataset_parser.add_argument("--seed", type=int, default=42)
ds_args = dataset_parser.parse_args(shlex.split("--max-per-illusion 60 --control-ratio 0.2 --seed 42"))
assert ds_args.max_per_illusion == 60
print("✓ dataset arguments parsed")

print("Step 13: Gradient Checkpointing...")
policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print("✓ gradient checkpointing enabled")

print("Step 14: Autocast Check...")
dtype = torch.bfloat16 if training_args.bf16 else torch.float32
with torch.amp.autocast("cuda", dtype=dtype, enabled=training_args.bf16):
    pass
print("✓ autocast context works")
