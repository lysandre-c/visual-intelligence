"""Direct Preference Optimization (DPO) training script for aligning VLMs (LLaVA, Qwen).

Aligns the model with human illusory perception using a multimodal DPO dataset.
Optimized for cluster execution with LoRA (PEFT) and BitsAndBytes (4-bit).
"""

import os
import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

import transformers.integrations.bitsandbytes

# Monkey patch for the 'frozenset' has no attribute 'discard' bug
def skip_check(*args, **kwargs):
    return True

transformers.integrations.bitsandbytes.validate_bnb_backend_availability = skip_check

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class ScriptArguments:
    model_id: str = field(
        default="llava-hf/llava-1.5-7b-hf", 
        metadata={"help": "The model id to fine-tune (e.g. llava-hf/llava-1.5-7b-hf or Qwen/Qwen-VL-Chat)."}
    )
    dataset_path: str = field(
        default="data/rl/dataset.jsonl", 
        metadata={"help": "Path to the DPO dataset JSONL."}
    )
    lora_r: int = field(
        default=16, 
        metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=32, 
        metadata={"help": "LoRA alpha."}
    )

def main():
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 1. Load Processor
    logger.info(f"Loading processor for {script_args.model_id}...")
    processor = AutoProcessor.from_pretrained(script_args.model_id, trust_remote_code=True)
    
    # Ensure pad token exists (often missing in Llama/LLaVA)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # 2. Preprocess Dataset
    logger.info(f"Loading dataset from {script_args.dataset_path}...")
    raw_dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")
    
    def preprocess_multimodal_dpo(example):
        # Load image and extract pixels
        img_path = str(Path(example["image"]))
        image = Image.open(img_path).convert("RGB")
        
        # DPO requires prompt, chosen, rejected
        prompt = example["prompt"]
        if "<image>" not in prompt:
            prompt= "<image>\n" + prompt
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "images": [image],
        }

    dataset = raw_dataset.map(preprocess_multimodal_dpo, remove_columns=raw_dataset.column_names)
    
    # 3. Load Model in 4-bit
    logger.info(f"Loading model {script_args.model_id} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        script_args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    
    
    # 4. PEFT Configuration (LoRA)
    # Determine target modules based on model family
    if "qwen" in script_args.model_id.lower():
        target_modules = ["c_attn", "attn.c_proj", "w1", "w2"] # Qwen-VL specific
    else:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"] # LLaVA / Llama specific
        
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 5. Initialize DPO Trainer
    # Note: LLaVA 1.5 requires specific handling in DPOTrainer.
    # We use the 'trl' implementation which supports multimodal via the processor.
    trainer = DPOTrainer(
        model,
        ref_model=None, # PEFT uses the base model as reference implicitly
        train_dataset=dataset,
        tokenizer=processor,
        args=training_args,
        peft_config=peft_config,
    )

    # 6. Train
    logger.info("Starting RL Alignment (DPO)...")
    trainer.train()
    
    # 7. Save adapter
    trainer.save_model(training_args.output_dir)
    logger.info(f"Training complete. LoRA adapter saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()
