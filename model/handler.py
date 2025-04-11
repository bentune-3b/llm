# ------ model.py ------

# Contains the implementation of the model

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
    global model, tokenizer
    logger.info(f"Loading model from {model_dir}.....")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model

def predict_fn(data: Dict)