"""
train.py -- joint SFT+DPO interleaved training

Implements a custom training loop that alternates between SFT and DPO batches.
SFT data is cycled if exhausted before all groups complete.

Training schedule (100 groups):
  Each group: DPO steps then SFT steps at constant --SFT ratio.
  SFT per group = ceil(DPO per group * SFT)

Usage:
  python train.py \
    --model_name_or_path ../Attack/LocalModels/models/Qwen2.5-Coder-7B-Instruct \
    --dpo_data          ../Defense/data/qwencoder7b/dpo.json \
    --sft_data          ../Defense/data/qwencoder7b/sft.json \
    --SFT               2.5 \
    --output_dir        saves/qwencoder7b \
    --lora_r            16 \
    --lora_alpha        32 \
    --learning_rate      1e-5 \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 4
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging to both stdout and a file under log_dir."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    logger = logging.getLogger("train_method2")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

def _format_prompt_with_chat_template(prompt_text, tokenizer, system_prompt):
    """Apply chat_template to a single user prompt, returning the text with generation prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class DPODataset(Dataset):
    """Dataset for DPO data in ShareGPT format.

    Each record: {conversations: [...], chosen: {...}, rejected: {...}}
    Returns tokenized chosen and rejected sequences.
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048,
                 use_chat_template: bool = False, system_prompt: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict] = []

        for rec in data:
            prompt_text = rec["conversations"][0]["value"]
            chosen_text = rec["chosen"]["value"]
            rejected_text = rec["rejected"]["value"]

            if use_chat_template:
                # Apply ChatML format: <|im_start|>system/user<|im_end|><|im_start|>assistant
                formatted_prompt = _format_prompt_with_chat_template(prompt_text, tokenizer, system_prompt)
                prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
            else:
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            # Tokenize chosen response
            chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=False) + [tokenizer.eos_token_id]

            # Tokenize rejected response
            rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=False) + [tokenizer.eos_token_id]

            # Build full sequences
            chosen_full = prompt_ids + chosen_ids
            rejected_full = prompt_ids + rejected_ids

            # Truncate
            if len(chosen_full) > max_length:
                chosen_full = chosen_full[:max_length]
            if len(rejected_full) > max_length:
                rejected_full = rejected_full[:max_length]

            # Labels mask prompt (set to -100 for prompt tokens)
            chosen_labels = [-100] * len(prompt_ids) + chosen_full[len(prompt_ids):]
            rejected_labels = [-100] * len(prompt_ids) + rejected_full[len(prompt_ids):]

            self.samples.append({
                "chosen_input_ids": chosen_full,
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_full,
                "rejected_labels": rejected_labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SFTDataset(Dataset):
    """Dataset for SFT data in ShareGPT format.

    Each record: {conversations: [{from: human, value: ...}, {from: gpt, value: ...}]}
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048,
                 use_chat_template: bool = False, system_prompt: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict] = []

        for rec in data:
            messages = rec["conversations"]

            if use_chat_template:
                # Build proper ChatML conversation with role mapping
                chat_messages = []
                if system_prompt:
                    chat_messages.append({"role": "system", "content": system_prompt})
                for msg in messages:
                    role = "user" if msg["from"] == "human" else "assistant"
                    chat_messages.append({"role": role, "content": msg["value"]})
                full_text = tokenizer.apply_chat_template(chat_messages, tokenize=False)
            else:
                # Concatenate all messages as raw text
                full_text = ""
                for msg in messages:
                    full_text += msg["value"]
                    if msg != messages[-1]:
                        full_text += "\n"

            # Tokenize
            all_ids = tokenizer.encode(full_text, add_special_tokens=False) + [tokenizer.eos_token_id]

            # Simple approach: mask everything except assistant response
            # We use a simple heuristic: labels = input_ids for all tokens,
            # since we want model to learn all token positions
            if len(all_ids) > max_length:
                all_ids = all_ids[:max_length]

            self.samples.append({
                "input_ids": all_ids,
                "labels": all_ids.copy(),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

def dpo_collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate DPO batch: concatenate chosen and rejected into one batch."""

    def pad_sequences(sequences, pad_value):
        max_len = max(len(s) for s in sequences)
        padded = []
        masks = []
        for s in sequences:
            pad_len = max_len - len(s)
            padded.append(s + [pad_value] * pad_len)
            masks.append([1] * len(s) + [0] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    chosen_input_ids = [s["chosen_input_ids"] for s in batch]
    chosen_labels = [s["chosen_labels"] for s in batch]
    rejected_input_ids = [s["rejected_input_ids"] for s in batch]
    rejected_labels = [s["rejected_labels"] for s in batch]

    # Concatenate chosen + rejected
    all_input_ids = chosen_input_ids + rejected_input_ids
    all_labels = chosen_labels + rejected_labels

    input_ids_tensor, attention_mask = pad_sequences(all_input_ids, pad_token_id)
    labels_tensor, _ = pad_sequences(all_labels, -100)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
    }


def sft_collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Collate SFT batch."""

    def pad_sequences(sequences, pad_value):
        max_len = max(len(s) for s in sequences)
        padded = []
        masks = []
        for s in sequences:
            pad_len = max_len - len(s)
            padded.append(s + [pad_value] * pad_len)
            masks.append([1] * len(s) + [0] * pad_len)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    input_ids = [s["input_ids"] for s in batch]
    labels = [s["labels"] for s in batch]

    input_ids_tensor, attention_mask = pad_sequences(input_ids, pad_token_id)
    labels_tensor, _ = pad_sequences(labels, -100)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_dpo_loss(
    model,
    raw_model,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute DPO loss on a concatenated batch.

    First half of batch = chosen, second half = rejected.
    raw_model is the PEFT model (without DDP wrapper), used for ref logprobs with LoRA disabled.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    batch_size = input_ids.size(0) // 2

    # Compute per-token log probabilities
    def get_logprobs(logits, labels_tensor):
        logits = logits[:, :-1, :].contiguous()
        labels_tensor = labels_tensor[:, 1:].contiguous()
        log_probs = F.log_softmax(logits, dim=-1)

        valid_labels = labels_tensor.clone()
        valid_labels[valid_labels == -100] = 0

        per_token_logps = torch.gather(log_probs, dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels_tensor != -100).float()
        return (per_token_logps * mask).sum(dim=-1)

    # Compute ref logprobs with LoRA disabled (reuses training model, no extra GPU memory)
    with torch.no_grad():
        raw_model.disable_adapter_layers()
        ref_outputs = raw_model(input_ids=input_ids, attention_mask=attention_mask)
        raw_model.enable_adapter_layers()
        ref_logits = ref_outputs.logits

    chosen_logps_ref = get_logprobs(ref_logits[:batch_size], labels[:batch_size])
    rejected_logps_ref = get_logprobs(ref_logits[batch_size:], labels[batch_size:])
    del ref_logits, ref_outputs

    # Now compute policy logprobs
    policy_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    policy_logits = policy_outputs.logits

    chosen_logps_policy = get_logprobs(policy_logits[:batch_size], labels[:batch_size])
    rejected_logps_policy = get_logprobs(policy_logits[batch_size:], labels[batch_size:])

    log_ratios = chosen_logps_policy - rejected_logps_policy
    ref_log_ratios = chosen_logps_ref - rejected_logps_ref

    losses = -F.logsigmoid(beta * (log_ratios - ref_log_ratios))
    return losses.mean()


def compute_sft_loss(
    model,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Standard cross-entropy SFT loss."""
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_method2(
    model_name_or_path: str,
    dpo_data_path: str,
    sft_data_path: str,
    output_dir: str,
    k: int = None,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 5e-5,
    dpo_beta: float = 0.1,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.0,
    seed: int = 42,
    num_groups: int = 100,
    epochs: float = 2.0,
    save_steps: int = 500,
    logging_steps: int = 10,
    bf16: bool = True,
    max_memory_gb: Optional[int] = None,
    load_in_4bit: bool = False,
    SFT: float = 2.5,
    dpo_early_stop_loss: Optional[float] = None,
    use_chat_template: bool = False,
    system_prompt: Optional[str] = None,
):
    torch.manual_seed(seed)

    # ---- logging ----
    log = setup_logging(output_dir)

    # ---- load data ----
    log.info(f"[load] DPO data: {dpo_data_path}")
    with open(dpo_data_path, "r", encoding="utf-8") as f:
        dpo_all = json.load(f)
    log.info(f"  {len(dpo_all)} DPO samples")

    log.info(f"[load] SFT data: {sft_data_path}")
    with open(sft_data_path, "r", encoding="utf-8") as f:
        sft_all = json.load(f)
    log.info(f"  {len(sft_all)} SFT samples")

    # ---- load tokenizer and model ----
    log.info(f"[load] model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if max_memory_gb is not None:
        import re
        gpu_ids = list(range(torch.cuda.device_count()))
        max_mem = {i: f"{max_memory_gb}GiB" for i in gpu_ids}
        model_kwargs["max_memory"] = max_mem
        log.info(f"max_memory per GPU: {max_memory_gb} GiB")

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.pop("dtype", None)
        log.info("Using 4-bit quantization (QLoRA)")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    # ---- checkpoint resume state (check BEFORE applying LoRA) ----
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_path / "checkpoint-latest"
    state_file = output_path / "trainer_state.json"
    start_epoch = 0
    start_group = 0
    global_step = 0
    resuming = ckpt_dir.exists() and state_file.exists()

    if resuming:
        log.info(f"[resume] Loading checkpoint from {ckpt_dir}")
        model = PeftModel.from_pretrained(model, str(ckpt_dir), is_trainable=True)
        model.print_trainable_parameters()
        with open(state_file, "r") as f:
            state = json.load(f)
        start_epoch = state.get("epoch", 0)
        start_group = state.get("group", 0)
        global_step = state.get("global_step", 0)
        log.info(f"[resume] epoch={start_epoch}, group={start_group}, global_step={global_step}")
    else:
        log.info("[resume] No checkpoint found, starting from scratch")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ---- reference model: reuse PEFT model with LoRA disabled ----
    # Saves ~14GB vs loading a separate frozen model.
    raw_model = model

    # ---- build datasets ----
    dpo_dataset = DPODataset(dpo_all, tokenizer, max_length=max_length,
                             use_chat_template=use_chat_template, system_prompt=system_prompt)
    sft_dataset = SFTDataset(sft_all, tokenizer, max_length=max_length,
                             use_chat_template=use_chat_template, system_prompt=system_prompt)

    # ---- partition into groups ----
    dpo_per_group = len(dpo_dataset) // num_groups
    sft_per_group = max(1, int(dpo_per_group * SFT))

    dpo_chunks = []
    sft_chunks = []
    sft_offset = 0
    sft_total = len(sft_dataset)
    for g in range(num_groups):
        start_dpo = g * dpo_per_group
        end_dpo = start_dpo + dpo_per_group if g < num_groups - 1 else len(dpo_dataset)
        dpo_chunks.append(torch.utils.data.Subset(dpo_dataset, range(start_dpo, end_dpo)))

        end_sft = min(sft_offset + sft_per_group, sft_total)
        sft_chunks.append(torch.utils.data.Subset(sft_dataset, range(sft_offset, end_sft)))
        sft_offset = end_sft
        if sft_offset >= sft_total:
            sft_offset = 0  # cycle SFT data if needed

    log.info(f"{num_groups} groups, each: DPO={dpo_per_group} SFT={sft_per_group} (ratio={SFT}:1, total SFT={sft_total})")

    # ---- setup optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # ---- estimate total steps and setup scheduler ----
    total_dpo_batches_per_pass = sum(
        math.ceil(len(chunk) / (per_device_batch_size * gradient_accumulation_steps))
        for chunk in dpo_chunks
    )
    total_sft_batches_per_pass = sum(
        math.ceil(len(chunk) / (per_device_batch_size * gradient_accumulation_steps))
        for chunk in sft_chunks
    )
    total_steps_per_pass = total_dpo_batches_per_pass + total_sft_batches_per_pass
    total_steps = int(total_steps_per_pass * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    log.info(f"Total steps: {total_steps} (per pass: {total_steps_per_pass}), warmup: {warmup_steps}")
    log.info(f"Hyperparams: lr={learning_rate}, beta={dpo_beta}, batch={per_device_batch_size}, "
             f"grad_acc={gradient_accumulation_steps}, lora_r={lora_r}, lora_alpha={lora_alpha}")

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if resuming:
        opt_path = ckpt_dir / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        sched_path = ckpt_dir / "scheduler.pt"
        if sched_path.exists():
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    # ---- training loop ----
    def save_checkpoint(step: int, epoch: int, group: int):
        """Save model, optimizer, scheduler, and training position."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        torch.save(optimizer.state_dict(), str(ckpt_dir / "optimizer.pt"))
        torch.save(scheduler.state_dict(), str(ckpt_dir / "scheduler.pt"))
        with open(state_file, "w") as f:
            json.dump({"epoch": epoch, "group": group, "global_step": step}, f)
        # Also save a numbered copy
        numbered = output_path / f"checkpoint-{step}"
        model.save_pretrained(str(numbered))
        log.info(f"[save] checkpoint -> {ckpt_dir}  (step={step})")

    model.train()
    dpo_early_stopped = False

    for epoch_idx in range(start_epoch, int(epochs)):
        log.info(f"===== EPOCH {epoch_idx + 1}/{int(epochs)} =====")

        # Skip groups already completed in this epoch
        group_start = start_group if epoch_idx == start_epoch else 0

        for group_idx in range(group_start, num_groups):
            # ---- DPO phase (skip if early-stopped) ----
            if dpo_early_stopped:
                dpo_loss_total = 0.0
                dpo_step_count = 1
            else:
                dpo_chunk = dpo_chunks[group_idx]
                dpo_loader = DataLoader(
                    dpo_chunk,
                    batch_size=per_device_batch_size,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=lambda b: dpo_collate_fn(b, tokenizer.pad_token_id),
                )

                dpo_loss_total = 0.0
                dpo_step_count = 0

                for batch_idx, batch in enumerate(dpo_loader):
                    loss = compute_dpo_loss(model, raw_model, batch, beta=dpo_beta)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    dpo_loss_total += loss.item() * gradient_accumulation_steps
                    dpo_step_count += 1

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        if global_step % logging_steps == 0:
                            lr = scheduler.get_last_lr()[0]
                            log.info(f"step={global_step} lr={lr:.2e} "
                                     f"dpo_loss={dpo_loss_total/max(1,dpo_step_count):.4f} "
                                     f"[{group_idx+1}/{num_groups} DPO]")

                        if save_steps > 0 and global_step % save_steps == 0:
                            save_checkpoint(global_step, epoch_idx, group_idx)

                # Handle remaining gradients in DPO phase
                if dpo_step_count % gradient_accumulation_steps != 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Check for DPO early stopping
                if dpo_early_stop_loss is not None and dpo_step_count > 0:
                    avg_dpo = dpo_loss_total / dpo_step_count
                    if avg_dpo < dpo_early_stop_loss:
                        dpo_early_stopped = True
                        log.info(f"[dpo-early-stop] group={group_idx+1} avg_dpo_loss={avg_dpo:.4f} < {dpo_early_stop_loss} — skipping DPO for remaining groups")

            # ---- SFT phase ----
            sft_chunk = sft_chunks[group_idx]
            sft_loader = DataLoader(
                sft_chunk,
                batch_size=per_device_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=lambda b: sft_collate_fn(b, tokenizer.pad_token_id),
            )

            sft_loss_total = 0.0
            sft_step_count = 0

            for batch_idx, batch in enumerate(sft_loader):
                loss = compute_sft_loss(model, batch)
                loss = loss / gradient_accumulation_steps
                loss.backward()

                sft_loss_total += loss.item() * gradient_accumulation_steps
                sft_step_count += 1

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        lr = scheduler.get_last_lr()[0]
                        log.info(f"step={global_step} lr={lr:.2e} "
                                 f"sft_loss={sft_loss_total/max(1,sft_step_count):.4f} "
                                 f"[{group_idx+1}/{num_groups} SFT]")

                    if save_steps > 0 and global_step % save_steps == 0:
                        save_checkpoint(global_step, epoch_idx, group_idx)

            # Handle remaining gradients in SFT phase
            if sft_step_count % gradient_accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Save at end of each group (robust resume granularity)
            save_checkpoint(global_step, epoch_idx, group_idx + 1)

            log.info(f"[group {group_idx+1}/{num_groups}] "
                     f"dpo_loss={dpo_loss_total/max(1,dpo_step_count):.4f} "
                     f"sft_loss={sft_loss_total/max(1,sft_step_count):.4f}")

    # ---- save final model ----
    final_path = output_path / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    log.info(f"[done] final model saved to {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Method2 joint SFT+DPO interleaved training")
    ap.add_argument("--model_name_or_path", required=True,
                    help="Path to base model")
    ap.add_argument("--dpo_data", required=True,
                    help="Path to DPO data JSON (from build_dpo_method2.py)")
    ap.add_argument("--sft_data", required=True,
                    help="Path to SFT data JSON (from build_dpo_method2.py)")
    ap.add_argument("--k", type=int, default=None,
                    help="k value (number of BAD prompts used in data construction, unused in training)")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to save checkpoints and final model")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--dpo_beta", type=float, default=0.1)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_groups", type=int, default=100,
                    help="Number of interleaving groups (default 100)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--no_bf16", action="store_true",
                    help="Disable bfloat16 (use fp32)")
    ap.add_argument("--max_memory_gb", type=int, default=None,
                    help="Limit per-GPU memory allocation (GiB) to avoid OOM when sharing GPUs")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Use 4-bit quantization (QLoRA) to reduce memory")
    ap.add_argument("--SFT", type=float, default=2.5,
                    help="Constant SFT:DPO ratio (default 2.5). SFT per group = dpo_per_group * SFT")
    ap.add_argument("--dpo_early_stop_loss", type=float, default=None,
                    help="Skip DPO after avg loss drops below this threshold (e.g. 0.05)")
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Apply tokenizer.apply_chat_template() to format prompts with ChatML")
    ap.add_argument("--system_prompt", type=str, default=None,
                    help="System prompt for chat_template (e.g. 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.')")
    args = ap.parse_args()

    train_method2(
        model_name_or_path=args.model_name_or_path,
        dpo_data_path=args.dpo_data,
        sft_data_path=args.sft_data,
        k=args.k,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        dpo_beta=args.dpo_beta,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_groups=args.num_groups,
        epochs=args.epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=not args.no_bf16,
        max_memory_gb=args.max_memory_gb,
        load_in_4bit=args.load_in_4bit,
        SFT=args.SFT,
        dpo_early_stop_loss=args.dpo_early_stop_loss,
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
