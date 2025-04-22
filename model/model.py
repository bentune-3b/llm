# model.py
# ------------------------
# run once to build an instance of the model and
# save the weights in local directory
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from prompt import inject_cot
import os

model_id = "meta-llama/Llama-3.2-3B"
save_directory = "./llama-3.2-3b-4bit-finetuned"

# --- load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id)

# padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# --- prepare dataset ---
dataset_path = "model/train_set.jsonl"
if not os.path.exists(dataset_path):
    print(f"Error: Dataset file '{dataset_path}' not found. Please run dataset.py first.")
    exit()

# load the dataset
dataset = load_dataset('json', data_files=dataset_path, split='train')

# --- inject chain of thought ---
cot_injected_dataset_path = "model/train_set_cot.jsonl"
print(f"Injecting Chain of Thought into {dataset_path}...")
inject_cot(
    input_path=dataset_path,
    output_path=cot_injected_dataset_path,
    model=model,
    tokenizer=tokenizer,
    sample_frac=1.0         # inject cot into all samples
)
print(f"CoT injected dataset saved to {cot_injected_dataset_path}")

# Load the CoT injected dataset for training
train_dataset = load_dataset('json', data_files=cot_injected_dataset_path, split='train')

# --- tokenize dataset ---
def tokenize_function(examples):
    # Format the instruction, input, and output into a single text for fine-tuning
    # This format is based on the Alpaca instruction tuning format
    # You might need to adjust this based on the exact format expected by Llama-3.2
    text = [f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}{tokenizer.eos_token}"
            for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"])]
    return tokenizer(text, padding="max_length", truncation=True, max_length=512) # Adjust max_length as needed

tokenized_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "input", "output"])

# --- fine-tune setup ---
training_args = TrainingArguments(
    output_dir=save_directory,  # Directory to save the fine-tuned model
    num_train_epochs=3,          # Number of training epochs
    per_device_train_batch_size=4, # Batch size per GPU/device
    gradient_accumulation_steps=2, # Accumulate gradients over steps
    optim="paged_adamw_8bit",    # Optimizer
    learning_rate=2e-4,          # Learning rate
    lr_scheduler_type="cosine",  # Learning rate scheduler
    save_steps=500,              # Save checkpoint every 500 steps
    logging_steps=50,            # Log training progress every 50 steps
    report_to="none",            # Disable reporting to platforms like W&B
    push_to_hub=False,           # Do not push to Hugging Face Hub
    # Add these parameters for QLoRA
    bf16=True,                   # Enable bfloat16 if supported by your GPU
    # fp16=False,                # Disable fp16 if bf16 is enabled
    warmup_ratio=0.05,           # Warmup ratio for learning rate
    weight_decay=0.001,          # Weight decay
    ddp_find_unused_parameters=False # Needed for DDP with gradient accumulation
)

# data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# --- start fine-tuning ---
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning finished.")

# --- save model ---
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Fine-tuned model and tokenizer saved to {save_directory}")