from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if torch.cuda.is_available():
    print("found CUDA")
    device = "cuda"
else:
    print("no CUDA :(")
    device = "cpu"

MODEL_NAME = "meta-llama/Llama-3.2-3B"
SAVE_DIR = "./vanilla-llama-3.2-3b"

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Saving tokenizer")
tokenizer.save_pretrained(SAVE_DIR)

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16, # maxxing out raahh
    device_map="auto"
)
print("Saving model")
model.save_pretrained(SAVE_DIR)

print(f"Model and tokenizer saved to: {SAVE_DIR}")
