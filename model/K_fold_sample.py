!pip install -q transformers accelerate bitsandbytes

# login
from huggingface_hub import notebook_login
notebook_login()

# Load model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
print(generate_response("What is the capital of Mexico?"))

few_shot_prompt = """Classify the sentiment of the following reviews.

Review: I absolutely loved the product. Will buy again!
Sentiment: Positive

Review: The item arrived broken. Not satisfied.
Sentiment: Negative

Review: It was okay, not great but not bad either.
Sentiment: Neutral

Review: The customer service was chill.
Sentiment:"""

inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(model.device)
input_length = inputs["input_ids"].shape[1]
outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, temperature=0.6)
generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
print(generated_text.strip())

##################################################################################################################################

!pip install datasets

from datasets import Dataset

data = {
    "text": [
        "I loved this movie!", "It was terrible.", "Not bad", 
        "Really enjoyed it", "Awful experience", "Just okay"
    ],
    "label": [1, 0, 2, 1, 0, 2]  # 0 = negative, 1 = positive, 2 = neutral
}

dataset = Dataset.from_dict(data)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

import torch
from sklearn.model_selection import KFold
import json
import os

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
texts = dataset["text"]
labels = dataset["label"]


id2label = {0: "negative", 1: "positive", 2: "neutral"}
label2id = {v: k for k, v in id2label.items()}

os.makedirs("kfold_results", exist_ok=True)

fold = 1
for train_idx, val_idx in kf.split(texts):
    print(f"\n Fold {fold}")

    fold_results = []
    for i in val_idx:
        input_text = texts[i]
        true_label_id = labels[i]
        true_label = id2label[true_label_id]

        # Run inference
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        pred_label = id2label[pred_id]

        fold_results.append({
            "input": input_text,
            "label": true_label,
            "prediction": pred_label
        })

    # Save fold results
    with open(f"kfold_results/fold_{fold}_results.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    
    print(f" Saved fold_{fold}_results.json")
    fold += 1

from sklearn.metrics import accuracy_score

accuracies = []

for filename in sorted(os.listdir("kfold_results")):
    if filename.endswith(".json"):
        with open(os.path.join("kfold_results", filename), "r") as f:
            data = json.load(f)
            y_true = [d["label"] for d in data]
            y_pred = [d["prediction"] for d in data]
            acc = accuracy_score(y_true, y_pred)
            print(f" {filename} → Accuracy: {acc:.4f}")
            accuracies.append(acc)

print(f"\n Average Accuracy over {k} folds: {sum(accuracies)/len(accuracies):.4f}")

from sklearn.metrics import accuracy_score

accuracies = []

for filename in sorted(os.listdir("kfold_results")):
    if filename.endswith(".json"):
        with open(os.path.join("kfold_results", filename), "r") as f:
            data = json.load(f)
            y_true = [d["label"] for d in data]
            y_pred = [d["prediction"] for d in data]
            acc = accuracy_score(y_true, y_pred)
            print(f" {filename} → Accuracy: {acc:.4f}")
            accuracies.append(acc)

print(f"\n Average Accuracy over {k} folds: {sum(accuracies)/len(accuracies):.4f}")
