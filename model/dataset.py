# dataset.py
# ------------------------
# Prepare a mixed, token‐balanced JSONL training + validation split
# for LLaMA-3.2-3B instruction + reasoning + QA + alignment fine-tuning.
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

import json
import os

from datasets import load_dataset, interleave_datasets, DatasetDict, Dataset

# dataset-specific load arguments for configs
LOAD_KWARGS = {
    "gsm8k": {"name": "main"},
    "hotpot_qa": {"name": "distractor"},
    "mandarjoshi/trivia_qa": {"name": "rc"},
    "domenicrosati/TruthfulQA": {},
}

# all subsets for maveriq/bigbenchhard
BIGBENCH_SUBSETS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

# ——— user configs ———
VAL_SPLIT = 0.1  # frac
SEED = 42

# --- token shares ---
# none -> use all examples
# use number of examples
MIX_CONFIG = {
    "general": {
        "datasets": [
            ("yahma/alpaca-cleaned", None),
            ("Cleanlab/databricks-dolly-15k-cleaned", None),
            ("Nutanix/oasst1-processed-kto-cleaned", None),
        ],
        "weight": 0.40
    },
    "reasoning": {
        "datasets": [
            ("gsm8k", None),
            ("maveriq/bigbenchhard", None),
            ("voidful/StrategyQA", None),
            ("tau/commonsense_qa", None),
        ],
        "weight": 0.30
    },
    "knowledge": {
        "datasets": [
            ("hotpot_qa", None),
            ("sentence-transformers/natural-questions", None),
            ("mandarjoshi/trivia_qa", None),
        ],
        "weight": 0.15
    },
    "alignment": {
        "datasets": [
            ("Anthropic/hh-rlhf", None),
            ("domenicrosati/TruthfulQA", None),
            ("openai/webgpt_comparisons", None),
        ],
        "weight": 0.15
    }
}


def normalize(example, source):
    if source == "voidful/StrategyQA":
        return normalize_strategyqa(example)
    """
    map each set to its fields
    tweak field names per-source as you include more datasets
    """
    if source == "gsm8k":
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": example["question"],
            "output": example["answer"],
        }
    if source == "yahma/alpaca-cleaned":
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": example.get("input", ""),
            "output": example["output"],
        }

    if source == "Cleanlab/databricks-dolly-15k-cleaned":
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": example.get("context", ""),
            "output": example["response"],
        }

    if source == "Anthropic/hh-rlhf":
        # convert chosen/rejected → ORPO/DPO triplet
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": "",
            "output": f"<good>\n{example['chosen']}\n<bad>\n{example['rejected']}",
        }

    if source == "Nutanix/oasst1-processed-kto-cleaned":
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": example.get("prompt", ""),
            "output": example["completion"],
        }

    if source == "maveriq/bigbenchhard":
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": example.get("input", ""),
            "output": example["target"],
        }

    if source == "hotpot_qa":
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": f"Context: {example.get('context', '')}\nQuestion: {example.get('question', '')}",
            "output": example.get("answer", "")
        }

    if source == "sentence-transformers/natural-questions":
        # flatten natural-questions to instruction schema
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": f"Context: {example.get('context_document', '')}\nQuestion: {example.get('question', '')}",
            "output": example.get("value", "")
        }
    if source == "mandarjoshi/trivia_qa":
        # TriviaQA answers come as nested dicts; extract the text value
        raw_answer = example.get("answer", "")
        if isinstance(raw_answer, dict):
            raw_answer = raw_answer.get("value", "")
        return {
            "instruction": "You are an honest, respectful and helpful assistant.",
            "input": example.get("question", ""),
            "output": raw_answer,
        }

    # fallback --
    return {
        "instruction": "You are an honest, respectful and helpful assistant.",
        "input": example.get("input", ""),
        "output": example.get("output", example.get("answer", "")),
    }

def normalize_strategyqa(item):
    """
    Turn a StrategyQA example into the {instruction, input, output} schema:
      - instruction: a step-by-step yes/no QA task
      - input: the question plus all facts
      - output: "Yes" or "No"
    """
    instruction = "Answer the following yes/no question using the given facts, showing your reasoning step by step."
    facts_text = " ".join(item.get("facts", []))
    input_text = f"Question: {item.get('question', '')}\nFacts: {facts_text}"
    output_text = "Yes" if item.get("answer", False) else "No"
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def load_and_prep(source, split="train", num_samples=None):
    """
    dataset loader function
    """
    # special case: include all BigBenchHard subsets
    if source == "maveriq/bigbenchhard":
        parts = []
        for subset in BIGBENCH_SUBSETS:
            parts.append(load_dataset(source, name=subset, split=split))
        ds = interleave_datasets(parts, seed=SEED)

    elif source == "voidful/StrategyQA":
        # load and preprocess local StrategyQA data
        file_path = os.path.join("model", f"strategyqa_{split}.json")
        with open(file_path, "r") as f:
            raw = json.load(f)
        processed = []
        for item in raw:
            # keep only question, facts, and answer with consistent types
            q = item.get("question", "")
            facts = item.get("facts", [])
            if not isinstance(facts, list):
                facts = []
            ans = item.get("answer", False)
            processed.append({"question": q, "facts": facts, "answer": ans})
        ds = Dataset.from_list(processed)

    else:
        # existing generic load logic
        load_args = {"split": split}
        cfg = LOAD_KWARGS.get(source, {})
        for k, v in cfg.items():
            if v is not None:
                load_args[k] = v
        ds = load_dataset(source, **load_args)
    if num_samples:
        ds = ds.shuffle(seed=SEED).select(range(num_samples))
    # map to unified schema
    return ds.map(lambda ex: normalize(ex, source), remove_columns=ds.column_names)


def main():
    """
    main function

    -- load all datasets into buckets
    -- mix buckets
    -- split into training and validation sets
    -- write to jsonl
    """
    # --- 1 load & sample each bucket
    buckets = {}
    for bucket, cfg in MIX_CONFIG.items():
        parts = []
        for (ds_name, n) in cfg["datasets"]:
            parts.append(load_and_prep(ds_name, split="train", num_samples=n))
        # interleave within the bucket *evenly*
        buckets[bucket] = interleave_datasets(parts, seed=SEED)

    # 2 mix all buckets by weight
    mixed = interleave_datasets(
        [buckets[b] for b in MIX_CONFIG],
        probabilities=[MIX_CONFIG[b]["weight"] for b in MIX_CONFIG],
        seed=SEED
    )

    # 3 split into train / val
    ds_dict = mixed.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    ds: DatasetDict = ds_dict

    # 4 write out to JSONL
    for split in ["train", "test"]:
        out_path = "model/train_set.jsonl" if split == "train" else "model/val_set.jsonl"
        with open(out_path, "w") as f:
            for ex in ds[split]:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(ds[split])} examples → {out_path}")

if __name__ == "__main__":
    main()