# model.py
# ------------------------
# run once to build an instance of the model and
# save the weights in local directory
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

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

save_directory = "./llama-3.2-3b-4bit"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)