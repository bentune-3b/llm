# dataset.py
# ------------------------
# prepares instruction tuned dataset for training
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

import argparse
import json
import random
from datasets import load_dataset

# --- normalizing functions ---
def normalize_gsm8k(item):
    return {
        "instruction": "Solve the following grade-school math problem step by step.",
        "input": item["question"].strip(),
        "output": item["answer"].strip(),
    }

def normalize_grade_school_instructions(item):
    return {
        "instruction": item.get("instruction", "").strip(),
        "input": item.get("input", "").strip(),
        "output": item.get("output", "").strip(),
    }

def normalize_mgsm_mt(item):
    return {
        "instruction": "Resuelva el siguiente problema de matemáticas paso a paso.",
        "input": item["question"].strip(),
        "output": item["answer"].strip(),
    }

def normalize_modernized(item):
    instr = "Solve this math problem with clear reasoning."
    inp   = item.get("Problem", item.get("question", "")).strip()
    out   = item.get("Solution", item.get("answer", "")).strip()
    return {"instruction": instr, "input": inp, "output": out}

def normalize_math(item):
    return {
        "instruction": "Solve the following symbolic math problem:",
        "input": item.get("question", "").strip(),
        "output": item.get("solution", item.get("answer", "")).strip(),
    }

def normalize_hotpot_qa(item):
    ctx = item.get("context", "")
    ctx_text = " ".join(ctx) if isinstance(ctx, list) else ctx
    return {
        "instruction": "Answer the following question using the given context:",
        "input": f"Question: {item.get('question','').strip()}\nContext: {ctx_text.strip()}",
        "output": item.get("answer", "").strip(),
    }

def normalize_hhh_alignment(item):
    ans = item.get("answer")
    # choice-based
    if isinstance(ans, dict) and "choices" in ans and "labels" in ans:
        choices = ans["choices"]
        labels  = ans["labels"]
        if 1 in labels:
            out = choices[labels.index(1)]
        else:
            out = choices[0]
    else:
        out = item.get("answer", "")
    return {
        "instruction": "Provide a helpful and harmless response to the instruction.",
        "input": item.get("question", "").strip(),
        "output": out.strip(),
    }

def normalize_mmlu(item):
    return {
        "instruction": "Answer the following multiple-choice question:",
        "input": item.get("question", "").strip(),
        "output": item.get("answer", "").strip(),
    }

def normalize_strategy_qa(item):
    facts = item.get("facts", "").strip()
    ans   = "Yes" if item.get("answer") else "No"
    return {
        "instruction": "Answer the following yes/no question using background knowledge:",
        "input": item.get("question", "").strip(),
        "output": f"{facts}\nAnswer: {ans}",
    }

def normalize_trivia_qa(item):
    return {
        "instruction": "Answer the following trivia question:",
        "input": item.get("question", "").strip(),
        "output": item.get("answer", "").strip(),
    }

# --- constants ---
DATASETS = {
    "openai/gsm8k":                                    normalize_gsm8k,
    "qwedsacf/grade-school-math-instructions":         normalize_grade_school_instructions,
    "juletxara/mgsm_mt":                               normalize_mgsm_mt,
    "EkBass/Grade-School-Math-Modernized-Dataset":     normalize_modernized,
    "math":                                            normalize_math,
    "hotpot_qa":                                       normalize_hotpot_qa,
    "hhh_alignment":                                   normalize_hhh_alignment,
    "mmlu":                                            normalize_mmlu,
    "strategy_qa":                                     normalize_strategy_qa,
    "trivia_qa":                                       normalize_trivia_qa,
    "truthful_qa":                                     normalize_trivia_qa,  # same format
}

# --- processing ---
def process(name, split="train"):
    """
    processes given hugging face dataset
    -- includes instruction tuning
    :param name: hf dataset id
    :param split: train or test
    :return: dataset
    """
    print(f">>> Loading {name}:{split}")
    ds = load_dataset(name, split=split)    #load dataset from hf
    norm = DATASETS[name]                   #fetch the normalizing function from dict
    out = []                                # array to maintain all dataset
    for ex in ds:
        s = norm(ex)
        if s["instruction"] and s["output"]:
            out.append(s)
    print(f"--> {len(out)} valid samples from {name}")
    return out

def main():
    #iterate over all datasets
    all_samples = []
    for ds_name in DATASETS:
        all_samples.extend(process(ds_name, split="train"))

    #shuffle the training dataset
    random.seed(42)
    random.shuffle(all_samples)

    output_file = "train_set.jsonl"
    print(f">>> Writing {len(all_samples)} total samples to {output_file}")
    with open(output_file, "w", encoding="utf-8") as fw:
        for sample in all_samples:
            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
