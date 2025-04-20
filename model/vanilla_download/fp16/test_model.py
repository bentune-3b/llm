from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_path = os.path.expanduser("~/bentune/model/downloaded_models/vanilla-llama-3.2-3b-fp16")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"exit", "quit"}:
        break
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model:", response)
