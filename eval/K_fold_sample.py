!pip install datasets
!pip install evaluate
!pip install tokenizers

import numpy as np
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import evaluate

# Sample dataset
data = {
    "text": [
        "I loved this product!",
        "Terrible experience, would not recommend",
        "It was okay, nothing special",
        "Absolutely fantastic service",
        "The worst purchase I've ever made"
    ],
    "label": [1, 0, 2, 1, 0]  # 1=positive, 0=negative, 2=neutral
}

dataset = Dataset.from_dict(data)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Number of folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initiate list to store results
fold_results = []

texts = np.array(dataset["text"])
labels = np.array(dataset["label"])

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
    print(f"\n=== Fold {fold + 1}/{k} ===")

    # Split data
    train_texts = texts[train_idx]
    train_labels = labels[train_idx]
    val_texts = texts[val_idx]
    val_labels = labels[val_idx]

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        load_best_model_at_end=False,
        logging_dir=f"./logs_fold_{fold}",
    )

    # Initialize model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )

  
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    fold_results.append(eval_results)

    print(f"Fold {fold + 1} results:", eval_results)


avg_accuracy = np.mean([result["eval_accuracy"] for result in fold_results])
print(f"\nAverage accuracy across {k} folds: {avg_accuracy:.4f}")
