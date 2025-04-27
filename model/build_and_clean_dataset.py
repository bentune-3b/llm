#!/usr/bin/env python3
"""
build_and_clean_dataset.py

Full pipeline: load → normalize → clean → dedupe → mix → cap → split → write.
Combines logic from dataset.py and clean_dataset.py.
"""

import json
import unicodedata
import re
from hashlib import md5
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

from datasets import load_dataset, interleave_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer

# ----- Config ----- #

# How many examples you aim to end up with (after cleaning)
TARGET_TOTAL = 165_000

# Fraction for validation split
VAL_SPLIT = 0.10

# Random seed
SEED = 42

# Max tokens allowed in output
MAX_TOK = 2048

# Your mix buckets and weights
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
            ("hotpot_qa", 20000),
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

# System prompt used in normalization
SYS_PROMPT = "You are an honest, respectful and helpful assistant."

# ----- Cleaning tools ----- #

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", use_fast=True)
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")
PROFANITY_RE = re.compile(r"\b(fuck|sex|sexy|dick)\b", re.I)
_seen_hashes = set()

def _normalize_text(txt: Optional[str]) -> str:
    if not txt:
        return ""
    txt = unicodedata.normalize("NFC", txt)
    txt = "".join(c for c in txt if not unicodedata.category(c).startswith("C"))
    txt = URL_RE.sub("[URL]", txt)
    return WHITESPACE_RE.sub(" ", txt).strip()

def _num_tokens(txt: str) -> int:
    return len(TOKENIZER(txt, add_special_tokens=False).input_ids)

def _length_ok(txt: str, min_toks: int = 1, max_toks: int = MAX_TOK) -> bool:
    n = _num_tokens(txt)
    return min_toks <= n <= max_toks

def _contains_profanity(txt: str) -> bool:
    return bool(PROFANITY_RE.search(txt))

def clean_and_dedupe(ex: Dict[str, str]) -> bool:
    # normalize fields
    ex["instruction"] = _normalize_text(ex.get("instruction", ""))
    ex["input"]       = _normalize_text(ex.get("input", ""))
    ex["output"]      = _normalize_text(ex.get("output", ""))

    # filter by length and profanity
    if not _length_ok(ex["output"]):
        return False
    if _contains_profanity(ex["instruction"] + ex["input"] + ex["output"]):
        return False

    # dedupe by hash of triple
    h = md5(f"{ex['instruction']}|||{ex['input']}|||{ex['output']}".encode()).hexdigest()
    if h in _seen_hashes:
        return False
    _seen_hashes.add(h)
    return True

# ----- Normalization per source ----- #

def _norm_generic(i: str, o: str) -> Dict[str, str]:
    return {"instruction": SYS_PROMPT, "input": i or "", "output": o or ""}

def normalize_strategyqa(item: Dict[str, Any]) -> Dict[str, str]:
    q = item.get("question", "")
    facts = " ".join(item.get("facts", [])) if isinstance(item.get("facts", []), list) else ""
    out = "Yes" if item.get("answer", False) else "No"
    inp = f"Question: {q}\nFacts: {facts}"
    return _norm_generic(inp, out)

def normalize(example: Dict[str, Any], source: str) -> Dict[str, str]:
    if source == "voidful/StrategyQA":
        entry = normalize_strategyqa(example)
    elif source == "maveriq/bigbenchhard":
        entry = _norm_generic(example.get("input", ""), example.get("target", ""))
    elif source == "gsm8k":
        entry = _norm_generic(example["question"], example["answer"])
    elif source == "yahma/alpaca-cleaned":
        entry = _norm_generic(example.get("input", ""), example["output"])
    elif source == "Cleanlab/databricks-dolly-15k-cleaned":
        entry = _norm_generic(example.get("context", ""), example["response"])
    elif source == "Anthropic/hh-rlhf":
        good = example["chosen"].strip()
        bad  = example["rejected"].strip()
        merged = f"<good>\n{good}\n<bad>\n{bad}"
        entry = _norm_generic("", merged)
    elif source == "Nutanix/oasst1-processed-kto-cleaned":
        entry = _norm_generic(example.get("prompt", ""), example["completion"])
    elif source == "hotpot_qa":
        ctx = example.get("context", "")
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        inp = f"Context: {ctx}\nQuestion: {example.get('question','')}"
        entry = _norm_generic(inp, example.get("answer",""))
    elif source == "sentence-transformers/natural-questions":
        ans = example.get("answer") or example.get("answers") or example.get("value","")
        if isinstance(ans, list):
            ans = "; ".join(ans)
        inp = f"Question: {example.get('query', example.get('question',''))}"
        entry = _norm_generic(inp, ans)
    elif source == "mandarjoshi/trivia_qa":
        raw = example.get("answer","")
        if isinstance(raw, dict):
            raw = raw.get("value","")
        entry = _norm_generic(example.get("question",""), raw)
    else:
        entry = _norm_generic(
            example.get("input",""),
            example.get("output", example.get("answer",""))
        )
    entry["source"] = source
    return entry

# ----- Loader + Prep ----- #

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
def load_and_prep(source: str, split: str = "train", num_samples: Optional[int] = None) -> Dataset:
    if source == "maveriq/bigbenchhard":
        parts = [
            load_dataset(source, name=sub, split=split)
            for sub in BIGBENCH_SUBSETS
        ]
        ds = interleave_datasets(parts, seed=SEED)
    elif source == "voidful/StrategyQA":
        # local JSON load
        path = Path("model") / f"strategyqa_{split}.json"
        raw = json.loads(path.read_text(encoding="utf8"))
        clean = []
        for itm in raw:
            clean.append({
                "question": str(itm.get("question","")),
                "facts": itm.get("facts",[]),
                "answer": bool(itm.get("answer",False))
            })
        from datasets import Dataset as HFDataset
        ds = HFDataset.from_list(clean)
    else:
        kwargs = {"split": split} | LOAD_KWARGS.get(source, {})
        ds = load_dataset(source, **kwargs)

    if num_samples is not None:
        ds = ds.shuffle(seed=SEED).select(range(num_samples))

    # normalize to common schema
    ds = ds.map(lambda x, src=source: normalize(x, src),
                remove_columns=ds.column_names,
                num_proc=4)

    # clean + dedupe
    ds = ds.filter(clean_and_dedupe,
                   batched=False,
                   num_proc=4)

    return ds

# ----- Main pipeline ----- #

def main():
    out_dir = Path("model")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Build each bucket
    buckets: Dict[str, Dataset] = {}
    for name, cfg in MIX_CONFIG.items():
        parts = []
        print(f"Loading bucket '{name}'")
        for ds_name, n in cfg["datasets"]:
            ds_part = load_and_prep(ds_name, "train", n)
            if len(ds_part) > 0:  # Only add non-empty datasets
                print(f"  → {ds_name}: {len(ds_part)} post-clean examples")
                parts.append(ds_part)
            else:
                print(f"  → {ds_name}: skipped (0 examples after cleaning)")
        
        if parts:  # Only create bucket if there are non-empty datasets
            buckets[name] = interleave_datasets(
                parts,
                seed=SEED,
                stopping_strategy="all_exhausted"
            )
        else:
            print(f"  → Warning: Bucket '{name}' is empty after cleaning")
            buckets[name] = None

    # 2. Mix buckets by weight
    valid_buckets = {k: v for k, v in buckets.items() if v is not None}
    if not valid_buckets:
        raise ValueError("No valid datasets found after cleaning")
        
    mixed = interleave_datasets(
        [valid_buckets[k] for k in MIX_CONFIG if k in valid_buckets],
        probabilities=[MIX_CONFIG[k]["weight"] for k in MIX_CONFIG if k in valid_buckets],
        seed=SEED,
        stopping_strategy="all_exhausted"
    )
    print(f"Mixed total before cap: {len(mixed)}")

    # 3. Cap to TARGET_TOTAL
    if len(mixed) > TARGET_TOTAL:
        mixed = mixed.shuffle(seed=SEED).select(range(TARGET_TOTAL))
        print(f"Capped to {TARGET_TOTAL}")
    else:
        print("Under target; using all available")

    # 4. Split
    ds_dict: DatasetDict = mixed.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    print(f"Split sizes → train: {len(ds_dict['train'])}, val: {len(ds_dict['test'])}")

    # 5. Detailed breakdown
    for split in ("train","test"):
        counter = Counter(ex["source"] for ex in ds_dict[split])
        print(f"Breakdown for {split}:")
        for src, count in counter.most_common():
            print(f"  • {src}: {count}")

    # 6. Write JSONL
    for split, fname in [("train","train_set_cleaned.jsonl"),
                         ("test","val_set_cleaned.jsonl")]:
        path = out_dir / fname
        with path.open("w", encoding="utf8") as f:
            for ex in ds_dict[split]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(ds_dict[split])} → {path}")

if __name__ == "__main__":
    main()
