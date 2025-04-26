#!/usr/bin/env python3
"""
LoRA supervised fine-tuning for LLaMA-3 3.2B on ASU SOL
(using relative paths for cleaned train and validation JSONL)
"""

import os
from functools import partial

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# 1. Reproducibility
set_seed(42)

# 2. Paths (relative to project root)
BASE_MODEL_DIR = "./model/downloaded_models/vanilla-llama-3.2-3b-bf16"
TRAIN_FILE     = "./model/train_set_cleaned.jsonl"
VAL_FILE       = "./model/val_set_cleaned.jsonl"
OUTPUT_DIR     = "./model/output_model"

# 3. Prompt template
SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Always answer as concisely as possible while remaining accurate."
)

def build_prompt(instruction: str, inp: str | None = None) -> str:
    parts = [
        "<s>[INST] <<SYS>>\n",
        SYSTEM_PROMPT,
        "\n<</SYS>>\n\n",
        instruction.strip()
    ]
    if inp:
        parts.append(f"\n{inp.strip()}")
    parts.append("\n[/INST]")
    return "".join(parts)

# 4. Load and preprocess cleaned datasets
raw = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    split=None
)
train_ds = raw["train"]
eval_ds  = raw["validation"]

def format_example(ex):
    prompt   = build_prompt(ex["instruction"], ex.get("input"))
    response = ex["output"].strip()
    return {"text": f"{prompt}\n{response}</s>"}

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(format_example,  remove_columns=eval_ds.column_names)

# 5. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

tokenize = partial(
    tokenizer,
    truncation=True,
    padding="max_length",
    max_length=8192,
    return_tensors=None,
)

def tokenize_fn(batch):
    tokens = tokenize(batch["text"])
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_ds  = eval_ds.map(tokenize_fn,  batched=True, remove_columns=["text"])

# 6. Model + LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_model.gradient_checkpointing_enable()

peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, peft_cfg)

# 7. Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_strategy="steps",
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=["tensorboard", "wandb"],
)

# 8. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# 9. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    perplexity = torch.exp(loss).cpu().item()
    return {"eval_loss": loss.cpu().item(), "perplexity": perplexity}

# 10. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

if __name__ == "__main__":
    trainer.train()
    # Save final adapter for downstream merging or inference
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
