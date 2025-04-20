from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import os

def main():
    set_seed(42)

    model_dir = os.path.expanduser("~/bentune/model/downloaded_models/vanilla-llama-3.2-3b-bf16")
    data_file = os.path.expanduser("~/bentune/model/data/processed/instruction_tuning_data.jsonl")
    output_dir = os.path.expanduser("~/bentune/model/lora/experiments/v1_test1/output_model")

    # load and split dataset
    raw = load_dataset("json", data_files=data_file, split="train")
    ds = raw.train_test_split(test_size=0.1)
    train_ds, eval_ds = ds["train"], ds["test"]

    # formatting
    def format_example(ex):
        prompt = ex["instruction"].strip()
        if ex.get("input", ""):
            prompt += "\n" + ex["input"].strip()
        return {"text": f"{prompt}\n### Response:\n{ex['output'].strip()}"}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_example, remove_columns=eval_ds.column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # tokenization
    def tokenize_fn(batch):
        toks = tokenizer(batch["text"], truncation=True, max_length=1024)
        toks["labels"] = toks["input_ids"].copy()
        return toks

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    # LoRA
    peft_config = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    # compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ppl = torch.exp(loss)
        return {"eval_loss": loss.item(), "perplexity": ppl.item()}

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()
