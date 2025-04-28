# dataset.py
# ------------------------
# Build a mixed, token-balanced JSONL corpus for LLaMA-3.2-3B fine-tuning
# Team: Bentune-3B  |  Authors: Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale
# ------------------------

import json
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict

# ------------------------------ #
# Config tables
# ------------------------------ #

LOAD_KWARGS: Dict[str, Dict[str, Any]] = {
    "gsm8k": {"name": "main"},
    "hotpot_qa": {"name": "distractor"},
    "mandarjoshi/trivia_qa": {"name": "rc"},
    "sentence-transformers/natural-questions": {"name": "pair"},
    "domenicrosati/TruthfulQA": {},
    "Anthropic/hh-rlhf": {"name": "default"},
}

BIGBENCH_SUBSETS: List[str] = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation",
    "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies",
    "word_sorting",
]

VAL_SPLIT: float = 0.10
SEED: int = 42
TARGET_TOTAL: int = 150_000  # Target total number of examples

MIX_CONFIG: Dict[str, Dict[str, Any]] = {
    "general": {
        "datasets": [
            ("yahma/alpaca-cleaned", None),
            ("Cleanlab/databricks-dolly-15k-cleaned", None),
            ("Nutanix/oasst1-processed-kto-cleaned", None),
        ],
        "weight": 0.40,
    },
    "reasoning": {
        "datasets": [
            ("gsm8k", None),
            ("maveriq/bigbenchhard", None),
            ("voidful/StrategyQA", None),
            ("tau/commonsense_qa", None),
        ],
        "weight": 0.30,
    },
    "knowledge": {
        "datasets": [
            ("hotpot_qa", None),
            ("sentence-transformers/natural-questions", None),
            ("mandarjoshi/trivia_qa", None),
        ],
        "weight": 0.15,
    },
    "alignment": {
        "datasets": [
            ("Anthropic/hh-rlhf", None),
            ("domenicrosati/TruthfulQA", None),
            ("openai/webgpt_comparisons", None),
        ],
        "weight": 0.15,
    },
}

SYS_PROMPT = "You are an honest, respectful and helpful assistant."


def _norm_generic(input_text: str, output_text: str) -> Dict[str, str]:
    return {"instruction": SYS_PROMPT, "input": input_text, "output": output_text}


def normalize(example: Dict[str, Any], source: str) -> Dict[str, str]:
    """Map each supported source to the common schema and tag source."""
    if source == "voidful/StrategyQA":
        entry = normalize_strategyqa(example)
    elif source == "gsm8k":
        entry = _norm_generic(example["question"], example["answer"])
    elif source == "yahma/alpaca-cleaned":
        entry = _norm_generic(example.get("input", ""), example["output"])
    elif source == "Cleanlab/databricks-dolly-15k-cleaned":
        entry = _norm_generic(example.get("context", ""), example["response"])
    elif source == "Anthropic/hh-rlhf":
        good = example["chosen"].strip()
        bad = example["rejected"].strip()
        merged = f"<good>\n{good}\n<bad>\n{bad}"
        entry = _norm_generic("", merged)
    elif source == "Nutanix/oasst1-processed-kto-cleaned":
        entry = _norm_generic(example.get("prompt", ""), example["completion"])
    elif source == "maveriq/bigbenchhard":
        entry = _norm_generic(example.get("input", ""), example["target"])
    elif source == "hotpot_qa":
        ctx = example.get("context", "")
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        inp = f"Context: {ctx}\nQuestion: {example.get('question', '')}"
        entry = _norm_generic(inp, example.get("answer", ""))
    elif source == "sentence-transformers/natural-questions":
        answers = example.get("answer") or example.get("answers") or example.get("value", "")
        if isinstance(answers, list):
            answers = "; ".join(answers)
        inp = f"Question: {example.get('query', example.get('question', ''))}"
        entry = _norm_generic(inp, answers)
    elif source == "mandarjoshi/trivia_qa":
        raw = example.get("answer", "")
        if isinstance(raw, dict):
            raw = raw.get("value", "")
        entry = _norm_generic(example.get("question", ""), raw)
    else:
        entry = _norm_generic(example.get("input", ""), example.get("output", example.get("answer", "")))

    entry["source"] = source
    return entry


def normalize_strategyqa(item: Dict[str, Any]) -> Dict[str, str]:
    """Custom mapping for StrategyQA examples."""
    question = item.get("question", "")
    facts = " ".join(item.get("facts", []))
    answer_text = "Yes" if item.get("answer", False) else "No"
    inp = f"Question: {question}\nFacts: {facts}"
    return _norm_generic(inp, answer_text)


def load_and_prep(source: str, split: str = "train", num_samples: int | None = None) -> Dataset:
    """Load a dataset, normalise it, and optionally subsample."""
    if source == "maveriq/bigbenchhard":
        parts = [
            load_dataset(source, name=sub, split=split)
            for sub in BIGBENCH_SUBSETS
        ]
        ds = interleave_datasets(parts, seed=SEED)

    elif source == "voidful/StrategyQA":
        # load local JSON and coerce fields to uniform types
        file_path = Path("model") / f"strategyqa_{split}.json"
        raw_items = json.loads(file_path.read_text(encoding="utf-8"))

        clean_items = []
        for itm in raw_items:
            q = str(itm.get("question", ""))
            facts = itm.get("facts", [])
            if not isinstance(facts, list):
                facts = []
            ans = bool(itm.get("answer", False))
            clean_items.append({"question": q, "facts": facts, "answer": ans})

        ds = Dataset.from_list(clean_items)

    else:
        kwargs = {"split": split} | LOAD_KWARGS.get(source, {})
        ds = load_dataset(source, **kwargs)

    if num_samples is not None:
        ds = ds.shuffle(seed=SEED).select(range(num_samples))

    return ds.map(lambda x, src=source: normalize(x, src),
                  remove_columns=ds.column_names)


def main() -> None:
    out_dir = Path("model")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and report counts per source
    buckets: Dict[str, Dataset] = {}
    for bucket_name, cfg in MIX_CONFIG.items():
        parts: List[Dataset] = []
        print(f"\nLoading bucket '{bucket_name}':")
        for ds_name, n in cfg["datasets"]:
            ds_part = load_and_prep(ds_name, "train", n)
            count = len(ds_part)
            print(f"  {ds_name}: {count} examples")
            parts.append(ds_part)
        buckets[bucket_name] = interleave_datasets(
            parts,
            seed=SEED,
            stopping_strategy="all_exhausted"
        )
        print(f"Bucket '{bucket_name}' total after interleaving: {len(buckets[bucket_name])} examples")

    # 2. Mix buckets by weight and report
    mixed = interleave_datasets(
        [buckets[k] for k in MIX_CONFIG],
        probabilities=[MIX_CONFIG[k]["weight"] for k in MIX_CONFIG],
        seed=SEED,
        stopping_strategy="all_exhausted"
    )
    print(f"\nMixed dataset total: {len(mixed)} examples")

    # 2.b Cap to TARGET_TOTAL examples (preserving proportions)
    total_avail = len(mixed)
    if total_avail > TARGET_TOTAL:
        mixed = mixed.shuffle(seed=SEED).select(range(TARGET_TOTAL))
        print(f"Capped mixed dataset to {TARGET_TOTAL} examples (from {total_avail})")
    else:
        print(f"Only {total_avail} examples available; proceeding with all")

    # 3. Split into train / validation
    ds_dict: DatasetDict = mixed.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    print(f"\nSplit sizes: train={len(ds_dict['train'])}, validation={len(ds_dict['test'])}")

    # 4. Detailed breakdown by source for each split
    for split_key in ("train", "test"):
        counter = Counter(ex["source"] for ex in ds_dict[split_key])
        print(f"\nBreakdown for '{split_key}':")
        for src, cnt in sorted(counter.items()):
            print(f"  {src}: {cnt} examples")

    # 5. Dump to JSONL
    for split_key, file_name in [("train", "train_set.jsonl"), ("test", "val_set.jsonl")]:
        path = out_dir / file_name
        with path.open("w", encoding="utf-8") as f:
            for ex in ds_dict[split_key]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(ds_dict[split_key])} → {path}")


if __name__ == "__main__":
    main()
