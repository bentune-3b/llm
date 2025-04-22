# prompt.py
# ------------------------
# functions to inject chain of thought and few shot
# prompting into the dataset
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

import json
import random
import torch
from typing import List, Dict

def generate_cot(
        model,
        tokenizer,
        question,
        answer,
        cot_prefix="Let's think step by step:",
        max_new_tokens=256,
        temperature=0.7
):
    """
    generates a chain of thought prompt

    :param model: model instance
    :param tokenizer: tokenizer instance
    :param question: string of the question
    :param answer: answer string
    :param cot_prefix: chain of thought prompt
    :param max_new_tokens: int
    :param temperature: randomness
    :return: cot prompt
    """
    prompt = f"Question: {question}\nAnswer: {answer}\n\n{cot_prefix}"
    tokens = tokenizer(prompt, return_tensors="pt")
    tokens = {k: v.to(model.device) for k,v in tokens.items()}

    #in eval mode, generate output to the intermediate step
    model.eval()
    with torch.no_grad():
        out_ids = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )[0]

    #decode and store
    text = tokenizer.decode(out_ids, skip_special_tokens=True)

    # remove the prompt prefix so only the CoT remains
    return text.split(cot_prefix, 1)[1].strip()

def inject_cot(
    input_path: str,
    output_path: str,
    model,
    tokenizer,
    sample_frac: float = 1.0,
    seed: int = 42,
    gen_kwargs: dict = None
) -> None:
    """
    load a JSONL of {"instruction","input","output"} samples,
    generate CoT for a random subset using the provided model & tokenizer,
    replace `output` with the full rationale, and write to a new JSONL.

    :param input_path: jsonl input path
    :param output_path: output path for injected CoT
    :param model: model instance
    :param tokenizer: tokenizer instance
    :param sample_frac: fraction of samples to inject cot in
    :param seed: randomization seed
    :param gen_kwargs: ignore
    :return: nothing
    """
    random.seed(seed)

    #read all samples
    with open(input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    #choose samples to inject cot
    total = len(samples)
    k = int(total * sample_frac)
    indices = list(range(total))
    random.shuffle(indices)
    chosen = set(indices[:k])

    #generate cot and inject
    for i, sample in enumerate(samples):
        if i in chosen:
            cot = generate_cot(
                model, tokenizer, sample["input"], sample["output"],
                **(gen_kwargs or {})
            )
            sample["output"] = cot

    #update output
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
