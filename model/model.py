# model.py
# ------------------------
# fine-tune llama 3.2 3b on instruction tuned dataset with QLoRA
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)
from datasets import load_dataset, Dataset
import torch
import torch.backends.cudnn as cudnn
import os
import random
import numpy as np
from sklearn.metrics import f1_score

# --- configuration ---
model_id = "meta-llama/Llama-3.2-3B"
save_directory = "./model/llama-3.2-3b-finetuned"
dataset_jsonl_path = "model/train_set.jsonl"

# hyperparams
TRAINING_DATA_SAMPLE_SIZE = 100000  # increase later
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE_PER_DEVICE = 4
GRADIENT_ACCUMULATION_STEPS = 2
MAX_SEQUENCE_LENGTH = 512

DTYPE = torch.bfloat16

# --- device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Status: Using device: {device}")
cudnn.benchmark = True

# --- validate dataset ---
if not os.path.exists(dataset_jsonl_path):
    print(f"Error: Dataset file '{dataset_jsonl_path}' not found. Please run dataset.py first to generate it.")
    exit()

# --- load dataset ---
print(f">>> Status: Loading dataset from {dataset_jsonl_path}...")
full_dataset = load_dataset('json', data_files=dataset_jsonl_path, split='train')
print(f"Full dataset loaded with {len(full_dataset)} entries.")

# --- sampling ---
if TRAINING_DATA_SAMPLE_SIZE < len(full_dataset):
    random.seed(42)
    indices = random.sample(range(len(full_dataset)), TRAINING_DATA_SAMPLE_SIZE)
    train_dataset = full_dataset.select(indices)
    print(f">>> Status: Sampled dataset size: {len(train_dataset)}")
else:
    train_dataset = full_dataset

# --- load model and tokenizer ---
print(f">>> Status: Loading model and tokenizer: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- padding token setup ---
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# --- load model in 16-bit precision ---
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    device_map="auto"
)

model.to(device)        #move to gpu

# Enable gradient checkpointing for memory efficiency on A100
model.gradient_checkpointing_enable()

# resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# --- tokenize dataset ---
print(">>> Status: Tokenizing the dataset...")
def tokenize_function(examples):
    text = []
    for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
         # Formatting the example following the Alpaca-like instruction tuning format
         formatted_example = (
             f"Below is an instruction that describes a task, paired with an input that provides further context. "
             f"Write a response that appropriately completes the request.\n\n"
             f"### Instruction:\n{instruction}\n\n"
             f"### Input:\n{input}\n\n"
             f"### Response:\n{output}{tokenizer.eos_token}"
         )
         text.append(formatted_example)

    return tokenizer(text, truncation=True, max_length=MAX_SEQUENCE_LENGTH)

tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "input", "output"],
    num_proc=os.cpu_count()
)
print(">>> Status: Dataset tokenization complete.")

# --- create eval split ---
dataset_split = train_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

# --- compute metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    preds = np.argmax(shift_logits, axis=-1)
    mask = shift_labels != tokenizer.pad_token_id
    preds = preds[mask]
    labels_filt = shift_labels[mask]
    score = f1_score(labels_filt, preds, average="micro")
    return {"f1": score}

# --- fine tune setup ---
training_args = TrainingArguments(
    output_dir=save_directory,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_steps=500,
    logging_steps=50,
    report_to="none",
    save_total_limit=2,
    bf16=True,
    fp16=False,
    warmup_ratio=0.05,
    weight_decay=0.001,
    ddp_find_unused_parameters=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# --- trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.label_names = ["labels"]

# --- actual fine-tune fr fr ---
print(">>> Status: Starting fine-tuning...")
trainer.train()
print(">>> Status: Fine-tuning finished.")

# --- save model ---
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Fine-tuned model (LoRA adapters) and tokenizer saved to {save_directory}")