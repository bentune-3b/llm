# ------ handler.py ------

# Model Handler functions for sagemaker endpoint

# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model = None
tokenizer = None

def model_fn(model_dir):
    """
    load model from sagemaker directory
    """
    global model, tokenizer
    logger.info(f"Loading model from {model_dir}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model

def predict_fn(data: Dict, model) -> List[Dict[str, str]]:
    """
    run prediction on the input prompt.
    """
    logger.info(f"Received input: {data}")

    prompt = data.get("inputs")
    if not prompt:
        return [{"generated_text": "Missing 'inputs' field in request"}]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [{"generated_text": text}]
