from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

base_model_dir = os.path.expanduser("~/bentune/model/downloaded_models/vanilla-llama-3.2-3b-bf16")
adapter_dir = os.path.expanduser("~/bentune/model/lora/experiments/v1_test1/output_model/checkpoint-135")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

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
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model:", response)
