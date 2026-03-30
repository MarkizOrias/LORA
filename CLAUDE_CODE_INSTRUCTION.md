# Claude Code Task: Adapt finetune_phi3.py with Autoresearch Findings

## Context

This folder contains three files:

- `train.py` — the winning autoresearch pretraining script (read-only reference, do NOT modify)
- `finetune_phi3.py` — the LoRA fine-tuning script to be rewritten
- `build_dataset.py` — Q&A dataset builder (keep for future phase 2, do NOT modify now)

The autoresearch loop ran ~100 experiments pretraining a small GPT on raw text
shards and found an optimal training configuration (val_bpb ≈ 0.53). The task is
to rewrite `finetune_phi3.py` so it performs **continued pretraining (CPT)** of
Phi-3 Mini 3.8B on the same raw text corpus using LoRA, incorporating the
hyperparameter and schedule findings from the winning `train.py`.

## What changed: this is now continued pretraining, not SFT

The current `finetune_phi3.py` does supervised fine-tuning on Q&A chat pairs.
We are converting it to do **continued pretraining** — standard causal language
modelling (next-token prediction) on raw text from parquet shards. This means:

- No chat template, no system prompts, no Q&A formatting
- No dependency on `build_dataset.py` or ChromaDB
- Training data = raw text loaded from parquet files
- Loss = standard cross-entropy on next-token prediction
- The SFT phase (using build_dataset.py and Q&A pairs) becomes a separate
  optional phase 2 script later — do not touch build_dataset.py now

## Data locations (Windows paths)

- Parquet shards: `C:\Users\georg\AppData\Local\autoresearch\datasets\tinystories\data\`
- Tokenizer: `C:\Users\georg\AppData\Local\autoresearch\datasets\tinystories\tokenizer\`

Load all `.parquet` files from the data directory. Each shard has a text column
containing raw training text. Concatenate and tokenise for causal LM training.

## Hardware

- GPU: NVIDIA RTX 5060 Ti 16 GB VRAM (Blackwell, compute capability 12.0)
- Full bf16 LoRA (not QLoRA) — 16 GB is sufficient
- OS: Windows, PowerShell terminal

## Step-by-step instructions

### 1. Extract these findings from train.py (read, do not modify)

Open `train.py` and extract the winning hyperparameters. The key values are:

| Parameter | Value in train.py | Notes |
|---|---|---|
| Adam betas | `(0.8, 0.95)` | Lower beta1 = faster domain adaptation |
| Adam epsilon | `1e-10` | |
| Warmup ratio | `0.0` | Zero warmup won — LoRA adapters init near zero, safe to skip |
| Weight decay | `0.05` | Decays linearly to 0 over training: `WD * (1 - progress)` |
| LR schedule | Cosine decay to 10% of peak | `FINAL_LR_FRAC = 0.1` — floor at 10%, not zero |
| Warmdown ratio | `0.3` | Last 30% of training is the decay tail |
| Sequence length | Read `MAX_SEQ_LEN` from prepare.py imports | Match this in LoRA training |
| Batch strategy | `TOTAL_BATCH_SIZE = 2**15` (32768 tokens per step) | Scale grad_accum to approximate |

Also note these architectural insights (informational, affects LoRA config):
- Depth won over width → spread LoRA adapters across ALL layers at moderate rank
- Muon optimizer won for matrix weights → but Muon is for pretraining from scratch;
  for LoRA use AdamW with the winning betas instead
- Value embeddings + gated residuals helped → target v_proj in LoRA modules

### 2. Rewrite finetune_phi3.py

Rewrite the script to do continued pretraining with these requirements:

#### A. Imports and dependencies
- Keep Unsloth (`FastLanguageModel`, `get_peft_model`)
- Keep `SFTTrainer` / `SFTConfig` from `trl` (it supports raw text CLM training)
- Add: `from datasets import load_dataset` for parquet loading
- Add: `from transformers import TrainerCallback` for custom LR schedule
- Remove: all ChromaDB / SentenceTransformer references (those belong to build_dataset.py)
- Remove: chat template application, `phi3_format`, system prompts

#### B. Data loading — replace Q&A loader with parquet CLM loader
```python
def load_parquet_dataset(data_dir: Path, val_fraction: float = 0.05):
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
    
    split = raw.train_test_split(test_size=val_fraction, seed=42)
    print(f"Train examples: {len(split['train']):,}")
    print(f"Val examples:   {len(split['test']):,}")
    return split["train"], split["test"]
```

#### C. LoRA configuration — informed by autoresearch
```python
# LoRA — depth over width (autoresearch found deeper > wider)
LORA_RANK       = 16        # moderate rank, spread across all layers
LORA_ALPHA      = 32        # alpha = 2x rank
LORA_DROPOUT    = 0.05

# Target ALL linear layers — autoresearch found value projections matter
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
```

#### D. Training hyperparameters — directly from autoresearch
```python
# Autoresearch-derived hyperparameters
LEARNING_RATE   = 5e-4      # aggressive — autoresearch favoured high matrix_lr
WARMUP_RATIO    = 0.0       # zero warmup won
WEIGHT_DECAY    = 0.05      # starting value, will decay linearly
ADAM_BETAS      = (0.8, 0.95)  # autoresearch winning config
ADAM_EPSILON    = 1e-10
FINAL_LR_FRAC   = 0.1       # cosine floor at 10% of peak LR

# Batch sizing for 16 GB VRAM
BATCH_SIZE      = 2
GRAD_ACCUM      = 8         # effective batch = 16 sequences
MAX_SEQ_LEN     = 2048      # match autoresearch sequence length
```

#### E. Custom cosine-with-floor LR schedule
Add a `TrainerCallback` that implements cosine decay to `FINAL_LR_FRAC` of peak
(matching train.py's `get_lr_multiplier` function). HuggingFace's default cosine
decays to 0 which is wrong for our config.

```python
import math
from transformers import TrainerCallback

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
```

When constructing the SFTConfig, set `lr_scheduler_type = "constant"` so the
callback controls the schedule entirely. Add the callback to the trainer:
```python
trainer.add_callback(CosineFloorScheduler())
```

#### F. SFTConfig changes
```python
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
```

#### G. Update CLI arguments
```python
parser.add_argument("--data_dir", type=Path,
    default=Path(r"C:\Users\georg\AppData\Local\autoresearch\datasets\tinystories\data"),
    help="Directory containing parquet shards")
parser.add_argument("--output", type=Path, default=Path("./phi3-trading-lora"))
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--use_qlora", action="store_true")
```

Remove the `--dataset` argument that pointed to a JSONL file.

#### H. Update main() flow
```python
def main():
    # ... parse args, check GPU ...
    
    model, tokenizer = load_model(use_qlora)
    
    # Load raw text from parquet shards (NOT Q&A pairs)
    train_ds, val_ds = load_parquet_dataset(args.data_dir)
    
    trainer = train(model, tokenizer, train_ds, val_ds, args.output, args.epochs)
    
    # Save LoRA adapter
    adapter_path = args.output / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    
    # Skip smoke_test (no chat format in CPT phase)
    # Skip GGUF export for now (do after optional SFT phase 2)
    
    print(f"\nCPT LoRA adapter saved: {adapter_path}")
    print("Next: optionally run SFT phase 2 with build_dataset.py + Q&A pairs")
```

#### I. Remove or comment out
- `smoke_test()` function and `SMOKE_PROMPTS` — these test chat responses,
  irrelevant for CPT. Keep as commented code for phase 2.
- `export_gguf()` — defer to after SFT phase 2 or run separately.
- All chat template logic.

#### J. Update docstring
Replace the module docstring to reflect that this is now a continued pretraining
script, not an SFT script. Mention that SFT (phase 2) is a separate step using
build_dataset.py.

### 3. Do NOT modify these files
- `train.py` — read-only reference
- `build_dataset.py` — needed later for SFT phase 2, leave untouched

### 4. Test
After rewriting, the script should run as:
```powershell
python finetune_phi3.py --data_dir "C:\Users\georg\AppData\Local\autoresearch\datasets\tinystories\data" --output ./phi3-trading-lora --epochs 3
```

Expected behaviour:
- Loads Phi-3 Mini with bf16 LoRA (not QLoRA) on 16 GB VRAM
- Loads parquet shards as raw text
- Trains with autoresearch-derived schedule (cosine to 10% floor, zero warmup, betas 0.8/0.95)
- Saves LoRA adapter to output directory
- No chat formatting, no GGUF export, no smoke test

## Summary of what transfers from autoresearch

| Finding | How it maps to LoRA CPT |
|---|---|
| Adam betas (0.8, 0.95) | Set directly in SFTConfig |
| Zero warmup | `warmup_ratio = 0.0` |
| Cosine → 10% floor | Custom `CosineFloorScheduler` callback |
| Weight decay 0.05 decaying | Starting WD in config (linear decay is optional enhancement) |
| Depth > width | Rank 16 across all layers, not rank 64 on fewer |
| Value projections matter | Include `v_proj` in target modules |
| High matrix LR | `5e-4` (2.5x default) |
| Muon optimizer | Does NOT transfer — use AdamW with winning betas instead |
