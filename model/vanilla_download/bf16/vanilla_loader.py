import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./vanilla-llama-3.2-3b-bf16")
args = parser.parse_args()

SAVE_DIR = args.save_dir
MODEL_NAME = "meta-llama/Llama-3.2-3B"

if torch.cuda.is_available():
    print("found CUDA")
    device = "cuda"
else:
    print("no CUDA :(")
    device = "cpu"

print("Creating save directory:", SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Saving tokenizer")
tokenizer.save_pretrained(SAVE_DIR)

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, # maxxing out raahh
    device_map="auto",
    trust_remote_code=True
)
print("Saving model")
model.save_pretrained(SAVE_DIR)

print(f"Model and tokenizer saved to: {SAVE_DIR}")
