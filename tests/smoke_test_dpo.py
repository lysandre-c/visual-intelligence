#!/usr/bin/env python3
"""Smoke test: one forward + backward pass through the full DPO pipeline.

Uses a **tiny randomly-initialised LLaVA model** with the same architecture
as `llava-hf/llava-1.5-7b-hf` but drastically reduced dimensions so the
whole test runs in < 1 GB of RAM on a CPU login node.

Validates:
  1. Model + LoRA wiring and weight freezing / unfreezing
  2. Reference model creation and freezing
  3. Dataset loading
  4. Collator (symbol demo, option shuffling, tokenisation)
  5. Forward pass through compute_sequence_logps (policy + reference)
  6. Loss computation (SymmetricPolarityPreferenceLoss)
  7. Backward pass + gradient flow check
  8. Optimizer step

Usage:
    python tests/smoke_test_dpo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "results" / "hf_cache"))

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
)

from src.rl.dpo_train import compute_sequence_logps, PolarityDPODataset
from src.rl.collator import SymmetricPolarityCollator
from src.rl.loss import SymmetricPolarityPreferenceLoss


def _sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _build_tiny_llava() -> tuple[LlavaForConditionalGeneration, AutoProcessor]:
    """Build a ~17 MB randomly-initialised LLaVA with the real class.

    The architecture mirrors llava-hf/llava-1.5-7b-hf but with tiny
    hidden sizes so it fits in RAM on a CPU login node.
    """
    from transformers import CLIPVisionConfig, LlamaConfig

    vision_cfg = CLIPVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        image_size=336,
        patch_size=14,
    )
    text_cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=32064,          # Match llava-1.5-7b tokenizer
        max_position_embeddings=2048,
    )
    llava_cfg = LlavaConfig(
        vision_config=vision_cfg,
        text_config=text_cfg,
        image_token_id=32000,      # <image> token id for llava-1.5
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
    )

    model = LlavaForConditionalGeneration(llava_cfg)
    model.eval()

    # Use the *real* processor/tokenizer from the HF Hub (tiny download)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    return model, processor


def _freeze_and_lora(
    model: LlavaForConditionalGeneration,
    lora_config: LoraConfig,
) -> nn.Module:
    """Replicate the exact freeze/unfreeze/LoRA logic from dpo_train.py.

    Order matters:
    1. Apply LoRA first (adds adapters to ALL matching modules incl. CLIP)
    2. Freeze vision tower entirely (base + LoRA params in CLIP)
    3. Unfreeze MLP projector
    """
    # Step 1: Apply LoRA to the full model
    model = get_peft_model(model, lora_config)

    # Access the inner LlavaModel through the PEFT wrapper
    inner = model.base_model.model.model  # LlavaModel

    # Step 2: Freeze vision tower (including any LoRA adapters on CLIP)
    for param in inner.vision_tower.parameters():
        param.requires_grad = False

    # Step 3: Unfreeze projector
    inner.multi_modal_projector.requires_grad_(True)

    return model


def main() -> None:
    t0 = time.time()
    max_length = 768               # Must be > 576 (LLaVA image patch tokens)

    # ── 1. Build tiny model ─────────────────────────────────────────────
    _sep("Step 1: Building tiny LLaVA (random weights)")
    model, processor = _build_tiny_llava()
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"✓ Tiny LLaVA built  ({param_mb:.1f} MB)")

    # ── 2. Apply LoRA + freeze/unfreeze (same logic as dpo_train.py) ────
    _sep("Step 2: LoRA → freeze vision tower → unfreeze projector")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = _freeze_and_lora(model, lora_config)
    policy_model.print_trainable_parameters()

    # Sanity checks — go through the PEFT wrapper
    inner = policy_model.base_model.model.model

    vt_grad = any(p.requires_grad for p in inner.vision_tower.parameters())
    proj_grad = any(
        p.requires_grad for p in inner.multi_modal_projector.parameters()
    )
    lora_params = [
        n for n, p in policy_model.named_parameters()
        if p.requires_grad and "lora" in n.lower()
    ]
    # LoRA params in vision tower should be frozen
    vt_lora = [
        n for n, p in policy_model.named_parameters()
        if "vision_tower" in n and "lora" in n.lower()
    ]
    vt_lora_trainable = [
        n for n, p in policy_model.named_parameters()
        if "vision_tower" in n and "lora" in n.lower() and p.requires_grad
    ]

    assert not vt_grad, "Vision tower should be frozen!"
    assert proj_grad, "Projector should be trainable!"
    assert len(lora_params) > 0, "LoRA parameters should be trainable!"
    assert len(vt_lora_trainable) == 0, (
        f"Vision tower LoRA should be frozen! Found trainable: {vt_lora_trainable}"
    )
    print(f"  Vision tower frozen: ✓")
    print(f"  Vision tower LoRA frozen: ✓ ({len(vt_lora)} adapters, all frozen)")
    print(f"  Projector trainable: ✓")
    print(f"  LoRA params trainable: ✓ ({len(lora_params)} parameter groups)")

    # ── 3. Reference model ──────────────────────────────────────────────
    _sep("Step 3: Creating frozen reference model")
    ref_model, _ = _build_tiny_llava()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_grad = any(p.requires_grad for p in ref_model.parameters())
    assert not ref_grad, "Reference model should be fully frozen!"
    print("✓ Reference model created and frozen")

    # ── 4. Dataset ──────────────────────────────────────────────────────
    _sep("Step 4: Loading dataset")
    dataset = PolarityDPODataset("data/rl/dataset.jsonl")
    assert len(dataset) > 0, "Dataset is empty!"
    print(f"✓ Dataset loaded: {len(dataset)} entries")
    sample = dataset[0]
    print(f"  First sample keys: {list(sample.keys())}")

    # ── 5. Collator ─────────────────────────────────────────────────────
    _sep("Step 5: Testing collator (1 sample)")
    collator = SymmetricPolarityCollator(
        processor=processor,
        project_root=PROJECT_ROOT,
        max_length=max_length,
        symbol_demo=True,
        option_shuffle=True,
    )
    batch = collator([sample])
    print("✓ Collator produced batch")
    for k, v in sorted(batch.items()):
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else "N/A"
        print(f"  {k}: {shape}")

    # ── 6. Forward pass (policy) ────────────────────────────────────────
    _sep("Step 6: Forward pass – policy model")
    policy_model.train()

    logps = {}
    for prefix in ["orig_chosen", "orig_rejected", "inv_chosen", "inv_rejected"]:
        lp = compute_sequence_logps(
            policy_model,
            batch[f"{prefix}_input_ids"],
            batch[f"{prefix}_attention_mask"],
            batch[f"{prefix}_labels"],
            batch.get(f"{prefix}_pixel_values"),
        )
        logps[f"policy_{prefix}"] = lp
        print(f"  policy {prefix:16s} logps = {lp.item():.4f}")
    print("✓ Policy forward pass complete")

    # ── 7. Forward pass (reference) ─────────────────────────────────────
    _sep("Step 7: Forward pass – reference model (no grad)")
    with torch.no_grad():
        for prefix in ["orig_chosen", "orig_rejected", "inv_chosen", "inv_rejected"]:
            lp = compute_sequence_logps(
                ref_model,
                batch[f"{prefix}_input_ids"],
                batch[f"{prefix}_attention_mask"],
                batch[f"{prefix}_labels"],
                batch.get(f"{prefix}_pixel_values"),
            )
            logps[f"ref_{prefix}"] = lp
            print(f"  ref    {prefix:16s} logps = {lp.item():.4f}")
    print("✓ Reference forward pass complete")

    # ── 8. Loss computation ─────────────────────────────────────────────
    _sep("Step 8: Loss computation")
    criterion = SymmetricPolarityPreferenceLoss(
        beta=0.1, gamma=1.0, label_lambda=0.5, eta=0.1,
    )
    total_loss, l_dpo, l_sym, l_margin, l_ancpo = criterion(
        logps["policy_orig_chosen"],
        logps["policy_orig_rejected"],
        logps["ref_orig_chosen"],
        logps["ref_orig_rejected"],
        logps["policy_inv_chosen"],
        logps["policy_inv_rejected"],
        logps["ref_inv_chosen"],
        logps["ref_inv_rejected"],
    )
    print(f"  total_loss = {total_loss.item():.4f}")
    print(f"  L_dpo      = {l_dpo.item():.4f}")
    print(f"  L_sym      = {l_sym.item():.4f}")
    print(f"  L_margin   = {l_margin.item():.4f}")
    print(f"  L_ancpo    = {l_ancpo.item():.4f}")
    assert not torch.isnan(total_loss), "Loss is NaN!"
    assert not torch.isinf(total_loss), "Loss is Inf!"
    print("✓ Loss computation complete (finite values)")

    # ── 9. Backward pass ────────────────────────────────────────────────
    _sep("Step 9: Backward pass")
    total_loss.backward()

    grad_norms = {}
    for name, param in policy_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    n_with_grad = len(grad_norms)
    n_trainable = sum(1 for p in policy_model.parameters() if p.requires_grad)
    print(f"  Trainable params:    {n_trainable}")
    print(f"  Params with grads:   {n_with_grad}")

    lora_grads = {k: v for k, v in grad_norms.items() if "lora" in k.lower()}
    proj_grads = {k: v for k, v in grad_norms.items() if "multi_modal_projector" in k}
    print(f"  LoRA params with grad: {len(lora_grads)}")
    print(f"  Projector params with grad: {len(proj_grads)}")
    assert len(lora_grads) > 0, "LoRA params should have gradients!"
    assert len(proj_grads) > 0, "Projector params should have gradients!"

    # No vision tower grads
    vt_grads = {k: v for k, v in grad_norms.items() if "vision_tower" in k}
    assert len(vt_grads) == 0, f"Vision tower should have NO grads! Got: {list(vt_grads.keys())}"
    print("  No vision tower grads: ✓")

    nan_grads = [k for k, v in grad_norms.items() if v != v]
    inf_grads = [k for k, v in grad_norms.items() if v == float("inf")]
    assert len(nan_grads) == 0, f"NaN gradients in: {nan_grads}"
    assert len(inf_grads) == 0, f"Inf gradients in: {inf_grads}"
    print("  No NaN/Inf gradients: ✓")
    print("✓ Backward pass complete")

    # ── 10. Optimizer step ──────────────────────────────────────────────
    _sep("Step 10: Optimizer step")
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    optimizer.zero_grad()
    print("✓ Optimizer step complete")

    # ── 11. Checkpoint save (projector path check) ──────────────────────
    _sep("Step 11: Checkpoint save path check")
    projector_state = {
        k: v.cpu()
        for k, v in policy_model.base_model.model.model.multi_modal_projector.state_dict().items()
    }
    assert len(projector_state) > 0, "Projector state dict is empty!"
    print(f"  Projector state keys: {list(projector_state.keys())}")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy_model.save_pretrained(tmp_dir)
        torch.save(projector_state, os.path.join(tmp_dir, "multi_modal_projector.pt"))
    print("✓ Projector and LoRA checkpoint saved to disk successfully")

    # ── 12. CLI Argument Parsing ────────────────────────────────────────
    _sep("Step 12: sbatch CLI argument parsing")
    import shlex
    import argparse
    from transformers import HfArgumentParser
    from src.rl.dpo_train import ScriptArguments
    from transformers import TrainingArguments

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
            print("  Note: Bypassing bf16 CPU validation error for smoke test parsing.")
            args_list.extend(["--use_cpu", "True"])
            script_args, training_args = parser.parse_args_into_dataclasses(args=args_list)
        else:
            raise

    assert script_args.lora_r == 64
    assert script_args.symbol_demo is True
    assert training_args.gradient_checkpointing is True
    assert training_args.bf16 is True
    assert "tensorboard" in training_args.report_to
    print("✓ dpo_train.py arguments parsed successfully")

    dataset_parser = argparse.ArgumentParser()
    dataset_parser.add_argument("--output-dir", type=Path, default=None)
    dataset_parser.add_argument("--max-per-illusion", type=int, default=60)
    dataset_parser.add_argument("--control-ratio", type=float, default=0.2)
    dataset_parser.add_argument("--seed", type=int, default=42)
    
    dataset_args_list = shlex.split("--max-per-illusion 60 --control-ratio 0.2 --seed 42")
    ds_args = dataset_parser.parse_args(dataset_args_list)
    assert ds_args.max_per_illusion == 60
    assert ds_args.control_ratio == 0.2
    assert ds_args.seed == 42
    print("✓ dataset.py CLI arguments parsed successfully")

    # ── 13. Gradient Checkpointing Compatibility ────────────────────────
    _sep("Step 13: Gradient Checkpointing check")
    try:
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("✓ gradient checkpointing enabled on model")
    except Exception as e:
        assert False, f"Gradient checkpointing failed: {e}"

    # ── 14. Autocast Context Compatibility ──────────────────────────────
    _sep("Step 14: AMP Autocast check (bfloat16)")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    try:
        with torch.amp.autocast("cuda", dtype=dtype, enabled=training_args.bf16):
            _ = torch.tensor(1.0).cuda() if torch.cuda.is_available() else torch.tensor(1.0)
        print(f"✓ AMP Autocast context works with {dtype}")
    except Exception as e:
        assert False, f"AMP autocast context failed: {e}"

    # ── Done ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _sep(f"ALL CHECKS PASSED  ({elapsed:.1f}s)")
    print(
        "\n  The full pipeline (model build → LoRA → freeze → collation →\n"
        "  forward → loss → backward → optimizer → checkpoint path)\n"
        "  completed without errors.\n"
        "  You can safely submit to the cluster.\n"
    )


if __name__ == "__main__":
    main()
