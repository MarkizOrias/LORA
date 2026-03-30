"""
Phi-3 Mini Continued Pretraining (CPT) with LoRA
==================================================
Continued pretraining of Phi-3 Mini 3.8B on raw text from parquet shards
using LoRA, incorporating hyperparameter findings from the autoresearch
winning train.py configuration (val_bpb ≈ 0.53).

This is phase 1 (domain adaptation via next-token prediction on raw text).
Phase 2 (SFT on Q&A pairs using build_dataset.py) is a separate step.

Hardware requirements:
  GPU  : 16 GB VRAM  (bf16 LoRA, not QLoRA)
  RAM  : 16 GB+ system RAM
  Disk : ~15 GB free  (model download + checkpoints)

Usage:
    python finetune_phi3.py \
        --data_dir "C:\\Users\\georg\\AppData\\Local\\autoresearch\\datasets\\tinystories\\data" \
        --output ./phi3-trading-lora \
        [--epochs 3] [--use_qlora]
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("ERROR: unsloth is not installed.")
    print("Install with:")
    print('  pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"')
    print("  pip install --no-deps trl peft accelerate bitsandbytes")
    sys.exit(1)

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME      = "unsloth/Phi-3-mini-4k-instruct"
MAX_SEQ_LEN     = 2048      # Match autoresearch sequence length

# LoRA — depth over width (autoresearch found deeper > wider)
LORA_RANK       = 16        # moderate rank, spread across all layers
LORA_ALPHA      = 32        # alpha = 2x rank
LORA_DROPOUT    = 0.05

# Target ALL linear layers — autoresearch found value projections matter
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

# Autoresearch-derived hyperparameters
LEARNING_RATE   = 5e-4      # aggressive — autoresearch favoured high matrix_lr
WARMUP_RATIO    = 0.0       # zero warmup won
WEIGHT_DECAY    = 0.05      # starting value, will decay linearly
ADAM_BETAS      = (0.8, 0.95)  # autoresearch winning config
ADAM_EPSILON    = 1e-10
FINAL_LR_FRAC   = 0.1       # cosine floor at 10% of peak LR

# Batch sizing for 16 GB VRAM
BATCH_SIZE      = 4         # higher per-device batch = better GPU utilization
GRAD_ACCUM      = 4         # effective batch = 16 sequences
VAL_FRACTION    = 0.05


# ── Custom LR schedule ──────────────────────────────────────────────────────

class CosineFloorScheduler(TrainerCallback):
    """
    Cosine decay from peak LR to FINAL_LR_FRAC * peak LR.
    Matches autoresearch train.py schedule exactly.
    """
    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        if optimizer is None or state.max_steps == 0:
            return
        progress = state.global_step / state.max_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * cosine
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]
            group["lr"] = group["initial_lr"] * multiplier


# ── VRAM check ────────────────────────────────────────────────────────────────

def check_gpu():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This script requires a GPU.")
        sys.exit(1)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU   : {gpu_name}")
    print(f"VRAM  : {vram_gb:.1f} GB")
    if vram_gb < 10:
        print("WARNING: Less than 10 GB VRAM — switch to QLoRA with --use_qlora flag")
    return vram_gb


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_parquet_dataset(data_dir: Path, val_fraction: float = 0.05,
                         max_samples: int = None):
    """
    Load all parquet shards from data_dir for causal language modelling.
    No chat template — raw text, next-token prediction.
    """
    parquet_files = sorted(str(p) for p in data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    raw = load_dataset("parquet", data_files=parquet_files, split="train")

    # Identify the text column (could be "text", "content", etc.)
    text_cols = [c for c in raw.column_names if "text" in c.lower() or "content" in c.lower()]
    text_col = text_cols[0] if text_cols else raw.column_names[0]

    # Rename to "text" if needed (SFTTrainer expects "text")
    if text_col != "text":
        raw = raw.rename_column(text_col, "text")

    # Filter empty entries
    raw = raw.filter(lambda x: len(x["text"].strip()) > 0)

    # Subsample for LoRA CPT — adapters converge much faster than full pretraining
    if max_samples and len(raw) > max_samples:
        print(f"Subsampling {max_samples:,} from {len(raw):,} total examples")
        raw = raw.shuffle(seed=42).select(range(max_samples))

    split = raw.train_test_split(test_size=val_fraction, seed=42)
    print(f"Train examples: {len(split['train']):,}")
    print(f"Val examples:   {len(split['test']):,}")
    return split["train"], split["test"]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(use_qlora: bool):
    """
    Load Phi-3 Mini with LoRA adapters via Unsloth.
    16 GB VRAM → use bf16 LoRA (load_in_4bit=False, dtype=torch.bfloat16)
    < 12 GB VRAM → use QLoRA  (load_in_4bit=True)
    """
    load_in_4bit = use_qlora
    dtype        = None if use_qlora else torch.bfloat16

    print(f"\nLoading model: {MODEL_NAME}")
    print(f"Mode: {'QLoRA (4-bit)' if use_qlora else 'LoRA (bf16)'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
    )

    used_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. GPU memory used: {used_gb:.2f} GB")

    model = FastLanguageModel.get_peft_model(
        model,
        r                  = LORA_RANK,
        target_modules     = TARGET_MODULES,
        lora_alpha         = LORA_ALPHA,
        lora_dropout       = LORA_DROPOUT,
        bias               = "none",
        use_gradient_checkpointing = "unsloth",   # saves ~30% VRAM
        random_state       = 42,
        use_rslora         = True,    # rank-stabilised LoRA — better convergence
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}% of total)")

    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_ds, val_ds, output_dir: Path, epochs: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SFTConfig(
        output_dir              = str(output_dir / "checkpoints"),
        num_train_epochs        = epochs,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = 1,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_ratio            = 0.0,           # zero warmup
        learning_rate           = LEARNING_RATE,
        weight_decay            = WEIGHT_DECAY,
        adam_beta1              = ADAM_BETAS[0],  # 0.8
        adam_beta2              = ADAM_BETAS[1],  # 0.95
        adam_epsilon            = ADAM_EPSILON,
        lr_scheduler_type       = "constant",    # callback handles schedule
        bf16                    = True,
        fp16                    = False,
        optim                   = "adamw_8bit",
        eval_strategy           = "steps",
        eval_steps              = 50,
        save_strategy           = "steps",
        save_steps              = 100,
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        logging_steps           = 10,
        seed                    = 42,
        report_to               = "none",
        max_seq_length          = MAX_SEQ_LEN,
        dataset_text_field      = "text",        # raw text, no chat template
    )

    trainer = SFTTrainer(
        model          = model,
        tokenizer      = tokenizer,
        train_dataset  = train_ds,
        eval_dataset   = val_ds,
        args           = args,
    )

    trainer.add_callback(CosineFloorScheduler())

    print("\nStarting continued pretraining ...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes.")

    return trainer


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Continued pretraining of Phi-3 Mini with LoRA")
    parser.add_argument("--data_dir", type=Path,
        default=Path(r"C:\Users\georg\AppData\Local\autoresearch\datasets\tinystories\data"),
        help="Directory containing parquet shards")
    parser.add_argument("--output", type=Path, default=Path("./phi3-trading-lora"))
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (1 is sufficient for LoRA CPT)")
    parser.add_argument("--max_samples", type=int, default=100_000,
                        help="Max training samples (0 = use all). 100K is plenty for LoRA CPT.")
    parser.add_argument("--use_qlora", action="store_true",
                        help="Use QLoRA (4-bit) instead of LoRA (bf16). "
                             "Use this only if you have <12 GB VRAM.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phi-3 Mini Continued Pretraining (LoRA + Autoresearch)")
    print("=" * 60)

    vram = check_gpu()

    # Auto-select mode based on available VRAM
    if vram >= 14 and not args.use_qlora:
        print("\nUsing full bf16 LoRA (best quality for 16 GB VRAM)")
        use_qlora = False
    else:
        print("\nUsing QLoRA (4-bit) to fit available VRAM")
        use_qlora = True

    model, tokenizer = load_model(use_qlora)
    train_ds, val_ds = load_parquet_dataset(
        args.data_dir, max_samples=args.max_samples or None)

    trainer = train(model, tokenizer, train_ds, val_ds, args.output, args.epochs)

    # Save LoRA adapter
    adapter_path = args.output / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    print(f"\nCPT LoRA adapter saved: {adapter_path}")
    print("Next: optionally run SFT phase 2 with build_dataset.py + Q&A pairs")


if __name__ == "__main__":
    main()
