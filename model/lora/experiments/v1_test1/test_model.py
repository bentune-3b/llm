from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_dir = "~/bentune/model/downloaded_models/vanilla-llama-3.2-3b-bf16"
adapter_dir = "model/lora/experiments/v1_test1/output_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False)

# base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    adapter_dir,
    torch_dtype=torch.bfloat16
)
model.to(device)
model.eval()

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"exit", "quit"}:
        break
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.95
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model:", response)
