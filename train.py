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
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
    LlamaConfig,
)

# 1. Reproducibility
set_seed(42)

# 2. Paths (relative to project root)
BASE_MODEL_DIR = "./model/vanilla-llama-3.2-3b-bf16"
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

# 5. Tokenizer & tokenization
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

tokenize = partial(
    tokenizer,
    truncation=True,
    padding=False,
    max_length=2048,      # reduced sequence length
    return_tensors=None,
)

def tokenize_fn(batch):
    tokens = tokenize(batch["text"])
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_ds  = eval_ds.map(tokenize_fn,  batched=True, remove_columns=["text"])

# 6. Model + LoRA
cfg = LlamaConfig.from_pretrained(BASE_MODEL_DIR)
# rope_scaling removed

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    config=cfg,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
base_model.gradient_checkpointing_enable()
base_model.config.use_cache = False  # disable KV cache with gradient checkpointing

peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, peft_cfg)

# 7. Efficient sequence-packing collator
class PackedDataCollator:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        all_ids = [f["input_ids"] for f in features]
        packed_input_ids = []
        packed_labels = []
        buffer = []

        for ids in all_ids:
            # flush if buffer + ids + EOS would overflow max_length
            if len(buffer) + len(ids) + 1 > self.max_length:
                if buffer:  # only if buffer has data
                    seq = buffer
                    pad_len = self.max_length - len(seq)
                    seq += [self.pad_id] * pad_len
                    packed_input_ids.append(seq)
                    packed_labels.append(seq.copy())
                    buffer = []
            # now add the new ids and EOS
            if len(ids) + 1 <= self.max_length:
                buffer += ids + [self.eos_id]
            else:
                # truncate ids if too long individually
                truncated = ids[:self.max_length - 1]
                buffer += truncated + [self.eos_id]

        if buffer:
            seq = buffer
            pad_len = self.max_length - len(seq)
            seq += [self.pad_id] * pad_len
            packed_input_ids.append(seq)
            packed_labels.append(seq.copy())

        batch_input = torch.tensor(packed_input_ids, dtype=torch.long)
        attention_mask = (batch_input != self.pad_id).long()
        batch_labels = torch.tensor(packed_labels, dtype=torch.long)

        return {
            "input_ids": batch_input,
            "attention_mask": attention_mask,
            "labels": batch_labels,
        }

data_collator = PackedDataCollator(tokenizer, max_length=2048)

# 8. Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    bf16=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=10,    
    prediction_loss_only=True,                                   
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
    report_to=["tensorboard", "wandb"]
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
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
