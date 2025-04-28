from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_path = os.path.expanduser("./llama-3.2-3b-finetuned")

tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
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