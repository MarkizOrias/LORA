"""
Phase 2: Phi-3 Mini Trading SFT with LoRA
===========================================
Supervised fine-tuning on trading Q&A pairs, building on the Phase 1 CPT
adapter. Loads the Phase 1 adapter, merges it into the base weights, then
applies a fresh (lighter) LoRA for learning response format.

This is the final training step before deployment. After SFT the model is
exported to GGUF Q4_K_M for Raspberry Pi / llama.cpp serving.

Usage:
    python finetune_phi3_sft.py \
        --dataset ./trading_qa.jsonl \
        --cpt_adapter ./phi3-trading-lora/adapter \
        --output ./phi3-trading-sft \
        [--epochs 3] [--use_qlora] [--skip_gguf]
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
    from unsloth.chat_templates import get_chat_template
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
MAX_SEQ_LEN     = 2048

# SFT LoRA — smaller rank than CPT (learning format, not domain knowledge)
LORA_RANK       = 8
LORA_ALPHA      = 16        # alpha = 2x rank
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

# SFT hyperparameters — gentler than CPT
SFT_LEARNING_RATE   = 1e-4       # 5x lower than CPT
SFT_WARMUP_RATIO    = 0.03       # small warmup for SFT on merged weights
SFT_WEIGHT_DECAY    = 0.01       # lighter regularisation
SFT_BATCH_SIZE      = 2
SFT_GRAD_ACCUM      = 4          # effective batch = 8

# Carried over from autoresearch
ADAM_BETAS           = (0.8, 0.95)
ADAM_EPSILON         = 1e-10
FINAL_LR_FRAC        = 0.1       # cosine floor at 10%

VAL_FRACTION         = 0.1


# ── Custom LR schedule ──────────────────────────────────────────────────────

class CosineFloorScheduler(TrainerCallback):
    """
    Cosine decay from peak LR to FINAL_LR_FRAC * peak LR.
    Matches autoresearch train.py schedule.
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


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_with_cpt_adapter(cpt_adapter_dir: Path, use_qlora: bool = False):
    """
    Load Phi-3, merge the Phase 1 CPT adapter, then apply a fresh LoRA for SFT.
    """
    dtype = None if use_qlora else torch.bfloat16
    load_in_4bit = use_qlora

    # Step 1: Load base model WITH the CPT adapter
    print(f"Loading base model + CPT adapter from {cpt_adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(cpt_adapter_dir),
        max_seq_length = MAX_SEQ_LEN,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
    )

    # Step 2: Merge CPT adapter into base weights permanently
    print("Merging CPT adapter into base weights ...")
    try:
        model = model.merge_and_unload()
    except Exception as e:
        print(f"Unsloth merge_and_unload failed ({e}), trying PeftModel fallback ...")
        from peft import PeftModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = MODEL_NAME,
            max_seq_length = MAX_SEQ_LEN,
            dtype          = dtype,
            load_in_4bit   = load_in_4bit,
        )
        model = PeftModel.from_pretrained(model, str(cpt_adapter_dir))
        model = model.merge_and_unload()

    used_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded + merged. GPU memory used: {used_gb:.2f} GB")

    # Step 3: Apply fresh LoRA for SFT
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = LORA_RANK,
        target_modules             = TARGET_MODULES,
        lora_alpha                 = LORA_ALPHA,
        lora_dropout               = LORA_DROPOUT,
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = 42,
        use_rslora                 = True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"SFT trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_sft_dataset(dataset_path: Path, tokenizer, val_fraction: float = 0.1):
    """
    Load trading_qa.jsonl in Phi-3 ChatML format.
    Each row has a 'messages' field with system/user/assistant turns.
    """
    raw = load_dataset("json", data_files=str(dataset_path), split="train")

    # Apply Phi-3 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="phi-3")

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    formatted = raw.map(format_example, remove_columns=raw.column_names)
    split = formatted.train_test_split(test_size=val_fraction, seed=42)

    print(f"SFT train examples: {len(split['train'])}")
    print(f"SFT val examples:   {len(split['test'])}")
    return split["train"], split["test"], tokenizer


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, tokenizer, train_ds, val_ds, output_dir: Path, epochs: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SFTConfig(
        output_dir              = str(output_dir / "sft-checkpoints"),
        num_train_epochs        = epochs,
        per_device_train_batch_size = SFT_BATCH_SIZE,
        per_device_eval_batch_size  = 1,
        gradient_accumulation_steps = SFT_GRAD_ACCUM,
        warmup_ratio            = SFT_WARMUP_RATIO,
        learning_rate           = SFT_LEARNING_RATE,
        weight_decay            = SFT_WEIGHT_DECAY,
        adam_beta1              = ADAM_BETAS[0],
        adam_beta2              = ADAM_BETAS[1],
        adam_epsilon            = ADAM_EPSILON,
        lr_scheduler_type       = "constant",    # callback handles cosine-with-floor
        bf16                    = True,
        fp16                    = False,
        optim                   = "adamw_8bit",
        eval_strategy           = "steps",
        eval_steps              = 20,
        save_strategy           = "steps",
        save_steps              = 20,
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        logging_steps           = 5,
        seed                    = 42,
        report_to               = "none",
        max_seq_length          = MAX_SEQ_LEN,
        dataset_text_field      = "text",
    )

    trainer = SFTTrainer(
        model          = model,
        tokenizer      = tokenizer,
        train_dataset  = train_ds,
        eval_dataset   = val_ds,
        args           = args,
    )

    trainer.add_callback(CosineFloorScheduler())

    print("\nStarting SFT training ...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nSFT training complete in {elapsed/60:.1f} minutes.")

    return trainer


# ── Inference smoke test ──────────────────────────────────────────────────────

SMOKE_PROMPTS = [
    "BTC just formed a bullish engulfing candle on the 4h chart with RSI at 38. What are the key factors to evaluate before entering a long?",
    "Crude oil is in deep backwardation — 3-month spread is $4. What does this tell me about current supply/demand and how should I position?",
    "My mean-reversion strategy has a Sharpe of 1.8 in trending markets but -0.3 in choppy markets. How do I detect which regime I am in?",
]

def smoke_test(model, tokenizer):
    FastLanguageModel.for_inference(model)
    print("\n── Inference smoke test ─────────────────────────────────────\n")
    for prompt in SMOKE_PROMPTS:
        messages = [
            {"role": "system",    "content": "You are an expert trading analyst."},
            {"role": "user",      "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids  = inputs,
            max_new_tokens = 300,
            temperature    = 0.7,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True
        )
        print(f"Q: {prompt}\n")
        print(f"A: {decoded}\n")
        print("─" * 60 + "\n")


# ── Export ────────────────────────────────────────────────────────────────────

def export_gguf(model, tokenizer, output_dir: Path):
    """
    Merge LoRA adapter into base weights and export as Q4_K_M GGUF.
    This is the file that goes onto the Raspberry Pi.
    """
    gguf_path = output_dir / "phi3-trading-q4_k_m.gguf"
    print(f"\nExporting GGUF to {gguf_path} ...")
    model.save_pretrained_gguf(
        str(output_dir / "phi3-trading-merged"),
        tokenizer,
        quantization_method = "q4_k_m",
    )
    print(f"GGUF export complete.")
    print(f"\nFile to copy to Raspberry Pi:")

    ggufs = list(output_dir.rglob("*.gguf"))
    for g in ggufs:
        size_gb = g.stat().st_size / 1e9
        print(f"  {g}  ({size_gb:.2f} GB)")

    print(f"\nCopy command:")
    if ggufs:
        print(f"  scp {ggufs[0]} pi@trading-node.local:~/models/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: SFT on trading Q&A")
    parser.add_argument("--dataset",      type=Path, default=Path("./trading_qa.jsonl"))
    parser.add_argument("--cpt_adapter",  type=Path, default=Path("./phi3-trading-lora/adapter"),
                        help="Path to Phase 1 CPT adapter directory")
    parser.add_argument("--output",       type=Path, default=Path("./phi3-trading-sft"))
    parser.add_argument("--epochs",       type=int,  default=3)
    parser.add_argument("--use_qlora",    action="store_true",
                        help="Use QLoRA (4-bit) instead of LoRA (bf16). "
                             "Use this only if you have <12 GB VRAM.")
    parser.add_argument("--skip_gguf",    action="store_true",
                        help="Skip GGUF export (faster iteration)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 2: Phi-3 Mini Trading SFT")
    print("  (builds on Phase 1 CPT adapter)")
    print("=" * 60)

    vram = check_gpu()
    use_qlora = args.use_qlora or vram < 14

    if not args.dataset.exists():
        print(f"\nERROR: Dataset not found at {args.dataset}")
        print("Run build_dataset.py first:")
        print("  python build_dataset.py --db ./trading-kb --output ./trading_qa.jsonl")
        sys.exit(1)

    if not args.cpt_adapter.exists():
        print(f"\nERROR: CPT adapter not found at {args.cpt_adapter}")
        print("Run Phase 1 (finetune_phi3.py) first.")
        sys.exit(1)

    # Load model with merged CPT adapter + fresh SFT LoRA
    model, tokenizer = load_model_with_cpt_adapter(args.cpt_adapter, use_qlora)

    # Load Q&A dataset
    train_ds, val_ds, tokenizer = load_sft_dataset(args.dataset, tokenizer)

    # Train
    trainer = train(model, tokenizer, train_ds, val_ds, args.output, args.epochs)

    # Save SFT adapter
    adapter_path = args.output / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nSFT LoRA adapter saved: {adapter_path}")

    # Smoke test
    try:
        smoke_test(model, tokenizer)
    except Exception as e:
        print(f"\nSmoke test failed (non-fatal): {e}")
        print("The adapter was saved successfully — you can test inference separately.")

    # GGUF export
    if not args.skip_gguf:
        try:
            export_gguf(model, tokenizer, args.output)
        except Exception as e:
            print(f"\nGGUF export failed (non-fatal): {e}")
            print("You can export GGUF separately later.")

    print("\n" + "=" * 60)
    print("  Phase 2 complete.")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  SFT adapter : {adapter_path}/")
    if not args.skip_gguf:
        print(f"  GGUF model  : {args.output}/phi3-trading-merged/")
        print(f"\nCopy the .gguf file to Raspberry Pi and serve via llama.cpp / FastAPI")


if __name__ == "__main__":
    main()
