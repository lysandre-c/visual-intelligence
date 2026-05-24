#!/usr/bin/env python3
"""Symmetric Polarity-Inverted DPO training for LLaVA-1.5 / 1.6.

This script:
1. Loads LLaVA with a frozen vision encoder, unfrozen MLP projector,
   and LoRA on the language backbone.
2. Builds a frozen reference model copy for DPO ratio computation.
3. Trains using the four-term Symmetric Polarity Preference Loss on
   polarity-inverted illusion prompts.
4. Logs all component losses to WandB / TensorBoard.
5. Saves LoRA + projector checkpoints compatible with the existing
   ``LLaVAProber(adapter_path=...)`` evaluation pipeline.

Usage
-----
    python src/rl/dpo_train.py \\
        --output_dir results/rl_alignment \\
        --dataset_path data/rl/dataset.jsonl \\
        --max_steps 1000 --save_steps 100

See ``--help`` for all arguments.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Project root on sys.path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.rl.loss import SymmetricPolarityPreferenceLoss
from src.rl.collator import SymmetricPolarityCollator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Arguments
# ────────────────────────────────────────────────────────────────────────


@dataclass
class ScriptArguments:
    """Arguments specific to Symmetric Polarity DPO training."""

    dataset_path: str = field(
        default="data/rl/dataset.jsonl",
        metadata={"help": "Path to JSONL dataset (relative to project root)."},
    )
    model_name_or_path: str = field(
        default="llava-hf/llava-1.5-7b-hf",
        metadata={"help": "HuggingFace model ID or local path."},
    )
    push_to_hub_repo: Optional[str] = field(
        default=None,
        metadata={"help": "HF Hub repository ID to push final weights to (e.g. 'username/llava-dpo')."},
    )

    # ── Loss hyperparameters ────────────────────────────────────────────
    beta: float = field(default=0.1, metadata={"help": "DPO temperature β."})
    gamma: float = field(
        default=1.0, metadata={"help": "Weight for symmetric polarity loss."}
    )
    label_lambda: float = field(
        default=0.5, metadata={"help": "Weight for margin consistency loss."}
    )
    eta: float = field(
        default=0.1, metadata={"help": "Weight for anchored preference loss."}
    )

    # ── LoRA configuration ──────────────────────────────────────────────
    lora_r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha scaling."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})

    # ── Data processing ─────────────────────────────────────────────────
    max_length: int = field(
        default=1024, metadata={"help": "Max total sequence length."}
    )
    symbol_demo: bool = field(
        default=True, metadata={"help": "Apply Symbol Demonstration (SymDPO)."}
    )
    option_shuffle: bool = field(
        default=True, metadata={"help": "Dynamically shuffle MC options."}
    )


# ────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────


class PolarityDPODataset(Dataset):
    """Simple map-style dataset that reads the JSONL line by line."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self.entries: list[dict[str, Any]] = []
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.is_absolute():
            jsonl_path = PROJECT_ROOT / jsonl_path
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))
        logger.info("Loaded %d entries from %s", len(self.entries), jsonl_path)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.entries[idx]


# ────────────────────────────────────────────────────────────────────────
# Model setup helpers
# ────────────────────────────────────────────────────────────────────────


def _load_model_and_processor(
    model_name: str,
    lora_config: LoraConfig,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, Any]:
    """Load LLaVA with frozen vision, unfrozen projector, and LoRA LM.

    Returns (policy_model, processor).
    """
    from transformers import LlavaForConditionalGeneration

    logger.info("Loading base model: %s", model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # ── Step 1: Freeze vision tower (CLIP backbone) ─────────────────
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    logger.info("Frozen vision_tower: %d params",
                sum(p.numel() for p in model.vision_tower.parameters()))

    # ── Step 2: Unfreeze MLP projector ──────────────────────────────
    model.multi_modal_projector.requires_grad_(True)
    proj_params = sum(
        p.numel() for p in model.multi_modal_projector.parameters() if p.requires_grad
    )
    logger.info("Unfrozen multi_modal_projector: %d trainable params", proj_params)

    # ── Step 3: Apply LoRA to language backbone ─────────────────────
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def _create_reference_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> nn.Module:
    """Load a separate frozen reference model for DPO log-ratios."""
    from transformers import LlavaForConditionalGeneration

    logger.info("Loading frozen reference model: %s", model_name)
    ref_model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("Reference model loaded and frozen.")
    return ref_model


# ────────────────────────────────────────────────────────────────────────
# Log-probability extraction
# ────────────────────────────────────────────────────────────────────────


def compute_sequence_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-sequence log-probabilities for a batch.

    Only the tokens where ``labels != -100`` contribute (completion
    tokens).  Returns shape ``(batch_size,)``.

    Parameters
    ----------
    model : The LLaVA model (policy or reference).
    input_ids : ``(B, L)``
    attention_mask : ``(B, L)``
    labels : ``(B, L)`` with -100 for masked (prompt) tokens.
    pixel_values : ``(B, C, H, W)`` or None.
    """
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values

    outputs = model(**kwargs)
    logits = outputs.logits  # (B, L, V)

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]  # (B, L-1, V)
    shift_labels = labels[:, 1:]  # (B, L-1)

    # Per-token log-probs
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)
    token_logps = log_probs.gather(
        dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)  # (B, L-1)

    # Mask out prompt and padding tokens
    mask = (shift_labels != -100).float()
    sequence_logps = (token_logps * mask).sum(dim=-1)  # (B,)

    return sequence_logps


# ────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────


def train(
    script_args: ScriptArguments,
    training_args: TrainingArguments,
) -> None:
    """Main training function."""
    # ── LoRA config ─────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Load models ─────────────────────────────────────────────────────
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    policy_model, processor = _load_model_and_processor(
        script_args.model_name_or_path, lora_config, torch_dtype=dtype,
    )
    ref_model = _create_reference_model(
        script_args.model_name_or_path, torch_dtype=dtype,
    )

    # ── Dataset & collator ──────────────────────────────────────────────
    dataset = PolarityDPODataset(script_args.dataset_path)
    collator = SymmetricPolarityCollator(
        processor=processor,
        project_root=PROJECT_ROOT,
        max_length=script_args.max_length,
        symbol_demo=script_args.symbol_demo,
        option_shuffle=script_args.option_shuffle,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Loss & optimiser ────────────────────────────────────────────────
    criterion = SymmetricPolarityPreferenceLoss(
        beta=script_args.beta,
        gamma=script_args.gamma,
        label_lambda=script_args.label_lambda,
        eta=script_args.eta,
    )

    # Collect trainable parameters: LoRA params + projector params
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )

    # Linear warmup + cosine decay
    from transformers import get_cosine_schedule_with_warmup

    num_training_steps = training_args.max_steps or (
        len(dataloader) * training_args.num_train_epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.05),
        num_training_steps=num_training_steps,
    )

    # ── Gradient checkpointing ──────────────────────────────────────────
    if training_args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ── Training state ──────────────────────────────────────────────────
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    grad_accum_steps = training_args.gradient_accumulation_steps
    max_steps = training_args.max_steps or float("inf")
    save_steps = training_args.save_steps or 100

    # Optional: TensorBoard writer
    tb_writer = None
    if training_args.report_to and "tensorboard" in training_args.report_to:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))

    # Optional: WandB
    use_wandb = (
        training_args.report_to
        and "wandb" in training_args.report_to
    )
    if use_wandb:
        import wandb
        wandb.init(
            project="visual-intelligence-polarity-dpo",
            name=training_args.run_name or "symmetric-polarity-dpo",
            config={
                "beta": script_args.beta,
                "gamma": script_args.gamma,
                "lambda": script_args.label_lambda,
                "eta": script_args.eta,
                "lora_r": script_args.lora_r,
                "lora_alpha": script_args.lora_alpha,
                "lr": training_args.learning_rate,
                "max_steps": training_args.max_steps,
            },
        )

    logger.info("Starting training for %s steps ...", max_steps)
    policy_model.train()

    running_loss = 0.0
    epoch = 0

    while global_step < max_steps:
        epoch += 1
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= max_steps:
                break

            # Move batch to device
            device = next(policy_model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # ── Forward: policy model ───────────────────────────────────
            with torch.amp.autocast("cuda", dtype=dtype, enabled=training_args.bf16):
                # Original chosen
                policy_chosen_logps = compute_sequence_logps(
                    policy_model,
                    batch["orig_chosen_input_ids"],
                    batch["orig_chosen_attention_mask"],
                    batch["orig_chosen_labels"],
                    batch.get("orig_chosen_pixel_values"),
                )
                # Original rejected
                policy_rejected_logps = compute_sequence_logps(
                    policy_model,
                    batch["orig_rejected_input_ids"],
                    batch["orig_rejected_attention_mask"],
                    batch["orig_rejected_labels"],
                    batch.get("orig_rejected_pixel_values"),
                )
                # Inverted chosen
                policy_chosen_logps_inv = compute_sequence_logps(
                    policy_model,
                    batch["inv_chosen_input_ids"],
                    batch["inv_chosen_attention_mask"],
                    batch["inv_chosen_labels"],
                    batch.get("inv_chosen_pixel_values"),
                )
                # Inverted rejected
                policy_rejected_logps_inv = compute_sequence_logps(
                    policy_model,
                    batch["inv_rejected_input_ids"],
                    batch["inv_rejected_attention_mask"],
                    batch["inv_rejected_labels"],
                    batch.get("inv_rejected_pixel_values"),
                )

            # ── Forward: reference model (no grad) ──────────────────────
            with torch.no_grad():
                ref_chosen_logps = compute_sequence_logps(
                    ref_model,
                    batch["orig_chosen_input_ids"],
                    batch["orig_chosen_attention_mask"],
                    batch["orig_chosen_labels"],
                    batch.get("orig_chosen_pixel_values"),
                )
                ref_rejected_logps = compute_sequence_logps(
                    ref_model,
                    batch["orig_rejected_input_ids"],
                    batch["orig_rejected_attention_mask"],
                    batch["orig_rejected_labels"],
                    batch.get("orig_rejected_pixel_values"),
                )
                ref_chosen_logps_inv = compute_sequence_logps(
                    ref_model,
                    batch["inv_chosen_input_ids"],
                    batch["inv_chosen_attention_mask"],
                    batch["inv_chosen_labels"],
                    batch.get("inv_chosen_pixel_values"),
                )
                ref_rejected_logps_inv = compute_sequence_logps(
                    ref_model,
                    batch["inv_rejected_input_ids"],
                    batch["inv_rejected_attention_mask"],
                    batch["inv_rejected_labels"],
                    batch.get("inv_rejected_pixel_values"),
                )

            # ── Compute joint loss ──────────────────────────────────────
            with torch.amp.autocast("cuda", dtype=dtype, enabled=training_args.bf16):
                total_loss, l_dpo, l_sym, l_margin, l_ancpo = criterion(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    policy_chosen_logps_inv,
                    policy_rejected_logps_inv,
                    ref_chosen_logps_inv,
                    ref_rejected_logps_inv,
                )

                # Scale for gradient accumulation
                scaled_loss = total_loss / grad_accum_steps

            # ── Backward ────────────────────────────────────────────────
            scaled_loss.backward()

            # ── Gradient accumulation step ──────────────────────────────
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                running_loss += total_loss.item()

                # ── Logging ─────────────────────────────────────────────
                if global_step % training_args.logging_steps == 0:
                    avg_loss = running_loss / training_args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    log_msg = (
                        f"step={global_step}  loss={avg_loss:.4f}  "
                        f"L_dpo={l_dpo.item():.4f}  L_sym={l_sym.item():.4f}  "
                        f"L_margin={l_margin.item():.4f}  L_ancpo={l_ancpo.item():.4f}  "
                        f"lr={lr:.2e}"
                    )
                    logger.info(log_msg)

                    if tb_writer is not None:
                        tb_writer.add_scalar("loss/total", avg_loss, global_step)
                        tb_writer.add_scalar("loss/dpo_m", l_dpo.item(), global_step)
                        tb_writer.add_scalar("loss/symmetric", l_sym.item(), global_step)
                        tb_writer.add_scalar("loss/margin", l_margin.item(), global_step)
                        tb_writer.add_scalar("loss/ancpo", l_ancpo.item(), global_step)
                        tb_writer.add_scalar("lr", lr, global_step)

                    if use_wandb:
                        import wandb
                        wandb.log({
                            "loss/total": avg_loss,
                            "loss/dpo_m": l_dpo.item(),
                            "loss/symmetric": l_sym.item(),
                            "loss/margin": l_margin.item(),
                            "loss/ancpo": l_ancpo.item(),
                            "lr": lr,
                        }, step=global_step)

                    running_loss = 0.0

                # ── Checkpointing ───────────────────────────────────────
                if global_step % save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    # Save LoRA adapter
                    policy_model.save_pretrained(str(ckpt_dir))
                    # Save projector weights separately
                    projector_state = {
                        k: v.cpu()
                        for k, v in policy_model.base_model.model.multi_modal_projector.state_dict().items()
                    }
                    torch.save(
                        projector_state,
                        str(ckpt_dir / "multi_modal_projector.pt"),
                    )
                    logger.info("Saved checkpoint: %s", ckpt_dir)

    # ── Final save ──────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(str(final_dir))
    projector_state = {
        k: v.cpu()
        for k, v in policy_model.base_model.model.multi_modal_projector.state_dict().items()
    }
    torch.save(projector_state, str(final_dir / "multi_modal_projector.pt"))
    logger.info("Training complete. Final model saved to %s", final_dir)

    # ── Push to Hugging Face Hub ────────────────────────────────────────
    if script_args.push_to_hub_repo:
        logger.info("Pushing final weights to Hugging Face Hub: %s", script_args.push_to_hub_repo)
        try:
            # The PEFT model's push_to_hub uploads the LoRA adapter (adapter_model.safetensors)
            policy_model.push_to_hub(script_args.push_to_hub_repo, safe_serialization=True)
            
            # Use HfApi to upload the custom multi_modal_projector.pt file alongside it
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(final_dir / "multi_modal_projector.pt"),
                path_in_repo="multi_modal_projector.pt",
                repo_id=script_args.push_to_hub_repo,
                repo_type="model",
            )
            logger.info("Successfully pushed all weights to Hub at %s", script_args.push_to_hub_repo)
        except Exception as e:
            logger.error("Failed to push to Hub (Ensure HF_TOKEN is set): %s", e)

    if tb_writer is not None:
        tb_writer.close()
    if use_wandb:
        import wandb
        wandb.finish()


# ────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))

    # Allow passing a YAML/JSON config file
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".json")):
        script_args, training_args = parser.parse_yaml_file(
            sys.argv[1], allow_extra_keys=True
        )
    else:
        script_args, training_args = parser.parse_args_into_dataclasses()

    # Ensure output dir exists
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    train(script_args, training_args)
