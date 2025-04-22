# dataset.py
# ------------------------
# prepares instruction tuned dataset for training
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

import argparse
import json
import random
import pandas as pd
import pyarrow as pa
from datasets import load_dataset, Dataset

# --- normalizing functions ---
def normalize_gsm8k(item):
    return {
        "instruction": "Solve the following grade-school math problem step by step.",
        "input": safe_strip(item["question"]),
        "output": safe_strip(item["answer"]),
    }

def normalize_grade_school_instructions(item):
    return {
        "instruction": safe_strip(item.get("INSTRUCTION", "")),
        "input": "",
        "output": safe_strip(item.get("RESPONSE", "")),
    }

def normalize_mgsm_mt(item):
    return {
        "instruction": "Resuelva el siguiente problema de matemáticas paso a paso.",
        "input": safe_strip(item.get("question", "")),
        "output": safe_strip(item.get("answer", "")),
    }

def normalize_math(item):
    return {
        "instruction": "Solve the following symbolic math problem:",
        "input": safe_strip(item.get("question", "")),
        "output": safe_strip(item.get("solution", item.get("answer", ""))),
    }

def normalize_hotpot_qa(item):
    # Extract and clean question
    raw_q = item.get("question", "")
    q_text = safe_strip(raw_q)

    # Extract and clean context list
    raw_ctx = item.get("context", "")
    if isinstance(raw_ctx, list):
        parts = []
        for c in raw_ctx:
            parts.append(safe_strip(c))
        ctx_text = " ".join(parts)
    else:
        ctx_text = safe_strip(raw_ctx)

    # Extract and clean answer
    raw_a = item.get("answer", "")
    a_text = safe_strip(raw_a)

    return {
        "instruction": "Answer the following question using the given context:",
        "input": f"Question: {q_text}\nContext: {ctx_text}",
        "output": a_text,
    }

def normalize_hhh_alignment(item):
    # Extract the question from 'input' or fallback to 'question'
    question = item.get("input") 
    answer = ""

    # Try to get multiple-choice answers from 'targets'
    targets = item.get("targets", {})
    choices = targets.get("choices", [])
    labels = targets.get("labels", [])

    if 1 in labels:
        answer = choices[labels.index(1)]

    return {
        "instruction": "Provide a helpful and harmless response to the instruction.",
        "input": safe_strip(question),
        "output": safe_strip(answer),
    }


# Letter keys for multiple choice options
CHOICE_LETTERS = ["A", "B", "C", "D"]
def normalize_mmlu(item):
    # Extract all four answer choices in order
    choices = [item.get(letter, "") for letter in CHOICE_LETTERS]

    # Get the correct answer key
    answer_key = item.get("Answer", "").strip().upper()

    # Convert letter key to index 
    answer_index = CHOICE_LETTERS.index(answer_key) 

    # Select the correct answer text if index is valid
    answer_text = choices[answer_index] 

    # Format question with choices
    input_text = f"{safe_strip(item.get('Question', ''))}\nChoices:\n" + \
                 "\n".join([f"{letter}) {safe_strip(choice)}" for letter, choice in zip(CHOICE_LETTERS, choices)])

    return {
        "instruction": "Answer the following multiple-choice question:",
        "input": input_text,
        "output": safe_strip(answer_text),
    }


def normalize_strategy_qa(item):
    # Extract and clean question
    question = safe_strip(item.get("question", ""))

    # Normalize answer to "Yes"/"No"
    answer = "Yes" if item.get("answer") else "No"

    # Drop problematic fields like 'evidence' that cause Arrow conversion issues
    return {
        "instruction": "Answer the following yes/no question:",
        "input": question,
        "output": answer,
    }


def normalize_trivia_qa(item):
    return {
        "instruction": "Answer the following trivia question:",
        "input": safe_strip(item.get("question", "")),
        "output": safe_strip(item.get("answer", "")),
    }

def normalize_truthful_qa(item):
    return {
        "instruction": "Answer the following trivia question truthfully:",
        "input": safe_strip(item.get("Question", "")),
        "output": safe_strip(item.get("Best Answer", "")),
    }



# -----------------

# --- constants ---
DATASETS = {
    "qwedsacf/grade-school-math-instructions":         normalize_grade_school_instructions,
    "juletxara/mgsm_mt":                               normalize_mgsm_mt,
    "deepmind/math_dataset":                           normalize_math,
    "openai/gsm8k":                                    normalize_gsm8k,
    "hotpot_qa":                                       normalize_hotpot_qa,
    "HuggingFaceH4/hhh_alignment":                     normalize_hhh_alignment,
    "openai/MMMLU":                                    normalize_mmlu,
    "trivia_qa":                                       normalize_trivia_qa,
    "domenicrosati/TruthfulQA":                        normalize_truthful_qa
}

CONFIGS = {
    "openai/gsm8k": "main",
    "juletxara/mgsm_mt": "nllb-200-distilled-600M",
    "deepmind/math_dataset": "arithmetic__add_or_sub",
    "hotpot_qa": "distractor",
    "HuggingFaceH4/hhh_alignment": "harmless",
    "voidful/StrategyQA": "default",
    "trivia_qa": "unfiltered",
}

# datasets requiring streaming to avoid Arrow conversion issues
STREAMING_DATASETS = ["voidful/StrategyQA"]

# override split names for datasets without a 'train' split
SPLIT_OVERRIDES = {
    "openai/MMMLU": "test",
    "HuggingFaceH4/hhh_alignment": "test",
}

# helper to safely strip non-string values
def safe_strip(value):
    if hasattr(value, "strip"):
        return value.strip()
    return str(value)

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

    # determine effective split
    ds_split = SPLIT_OVERRIDES.get(name, split)
    # prepare load parameters
    load_kwargs = {"split": ds_split, "trust_remote_code": True}
    if name in STREAMING_DATASETS:
        load_kwargs["streaming"] = True

    try:
        if name in CONFIGS:
            ds = load_dataset(name, CONFIGS[name], **load_kwargs)
        else:
            ds = load_dataset(name, **load_kwargs)
    except Exception as e:
        print(f"--> Skipping {name} due to load error: {e}")
        return []

    norm = DATASETS[name]                       #fetch the normalizing function from dict
    out = []                                    # array to maintain all dataset
    for ex in ds:
        s = norm(ex)
        if s["instruction"] and s["output"]:
            out.append(s)
    print(f"--> {len(out)} valid samples from {name}")
    return out

def main():
    # Collect all normalized examples from defined datasets
    all_samples = []
    for ds_name in DATASETS:
        all_samples.extend(process(ds_name, split="train"))

    # ---- Load StrategyQA manually from local JSON file ----
    try:
        # Load raw JSON and normalize
        raw_data = [normalize_strategy_qa(item) for item in json.load(open("strategyqa_train.json"))]
        # Keep only well-formed examples
        clean_data = [item for item in raw_data if item["instruction"] and item["output"]]
        # Convert to Hugging Face Dataset via Pandas -> PyArrow
        df = pd.DataFrame(clean_data)
        table = pa.Table.from_pandas(df)
        hf_dataset = Dataset(table)
        # Append StrategyQA data to all_samples
        all_samples.extend(hf_dataset.to_list())
        print(f"--> {len(clean_data)} StrategyQA samples added.")
    except Exception as e:
        print(f"--> Skipping StrategyQA due to load error: {e}")

    # Shuffle all combined samples
    random.seed(42)
    random.shuffle(all_samples)

    # Write to JSONL file
    output_file = "train_set.jsonl"
    print(f">>> Writing {len(all_samples)} total samples to {output_file}")
    with open(output_file, "w", encoding="utf-8") as fw:
        for sample in all_samples:
            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
