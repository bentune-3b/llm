#!/usr/bin/env python3
"""
LoRA fine-tuning of LLaMA-3 3.2B on one A100 (ASU SOL)
• DeepSpeed ZeRO-2 (no CPU offload)
• 8-bit quantisation via bitsandbytes
• Flash-Attention 2 if installed, otherwise torch-SDPA
• Sequence length 2048, packed batches
"""

import os
from functools import partial
import importlib.util

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    set_seed,
    LlamaConfig,
)

# ─── 0. Paths & Constants ─────────────────────────────────────────────────────
BASE_MODEL_DIR = "./model/vanilla-llama-3.2-3b-bf16"
TRAIN_FILE     = "./model/train_set_cleaned.jsonl"
VAL_FILE       = "./model/val_set_cleaned.jsonl"
OUTPUT_DIR     = "./model/output_model"
DS_CONFIG      = os.path.abspath("model/ds_config.json")  # must exist
SEQ_LEN        = 2048
set_seed(42)

# ─── 1. Prepare Dataset ───────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as concisely as possible while remaining accurate."
)

def build_prompt(instr: str, inp: str | None = None) -> str:
    parts = [
        "<s>[INST] <<SYS>>\n", SYSTEM_PROMPT, "\n<</SYS>>\n\n",
        instr.strip()
    ]
    if inp:
        parts.append(f"\n{inp.strip()}")
    parts.append("\n[/INST]")
    return "".join(parts)

raw = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    split=None
)
train_ds, eval_ds = raw["train"], raw["validation"]

def format_ex(ex):
    return {
        "text": f"{build_prompt(ex['instruction'], ex.get('input'))}\n{ex['output'].strip()}</s>"
    }

train_ds = train_ds.map(format_ex, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(format_ex,  remove_columns=eval_ds.column_names)

tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
tok.pad_token = tok.eos_token

tokenize = partial(tok, truncation=True, padding=False, max_length=SEQ_LEN)

def tok_fn(batch):
    out = tokenize(batch["text"])
    out["labels"] = out["input_ids"].copy()
    return out

train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
eval_ds  = eval_ds.map(tok_fn,  batched=True, remove_columns=["text"])

# ─── 2. Load Model + LoRA + 8-bit ──────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
cfg     = LlamaConfig.from_pretrained(BASE_MODEL_DIR)

# pick attention implementation
if importlib.util.find_spec("flash_attn"):
    attn_impl = "flash_attention_2"
else:
    attn_impl = "sdpa"  # torch's fused scaled-dot-product

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    config=cfg,
    quantization_config=bnb_cfg,
    attn_implementation=attn_impl,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base.gradient_checkpointing_enable()
base.config.use_cache = False

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)

# ─── 3. Packed Data Collator ─────────────────────────────────────────────────
class PackedCollator:
    def __init__(self, tokenizer, max_len=SEQ_LEN):
        self.eos, self.pad = tokenizer.eos_token_id, tokenizer.pad_token_id
        self.max_len = max_len

    def __call__(self, features):
        ids   = [f["input_ids"] for f in features]
        buf   = []
        batch = []
        for seq in ids:
            if len(buf) + len(seq) + 1 > self.max_len:
                batch.append(buf)
                buf = []
            buf += seq + [self.eos]
        if buf:
            batch.append(buf)

        # pad to full length
        padded = [s + [self.pad] * (self.max_len - len(s)) for s in batch]
        tens = torch.tensor(padded, dtype=torch.long)
        return {
            "input_ids": tens,
            "attention_mask": (tens != self.pad).long(),
            "labels": tens.clone(),
        }

collate_fn = PackedCollator(tok)

# ─── 4. TrainingArguments ─────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    bf16=True,
    deepspeed=DS_CONFIG,
    optim="adamw_bnb_8bit",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    dataloader_num_workers=2,            # lowered to avoid worker warnings
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_strategy="steps",
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=["tensorboard"],
)

# ─── 5. Metrics & Trainer ────────────────────────────────────────────────────
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    shift_logits  = logits[..., :-1, :].contiguous()
    shift_labels  = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return {"eval_loss": loss.item(), "perplexity": torch.exp(loss).item()}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
