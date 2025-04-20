# -----------------------------------------------------------------------------
# Script: prepare_instruction_data.py
# Purpose: Convert dev dataset (multi-source) into Alpaca-style .jsonl format 
#          for instruction tuning with HuggingFace + LoRA (LLaMA base model).
#
# If you're using a different dataset:
# - Replace `INPUT_FILE` with your JSON file path.
# - Update or extend the `format_sample()` function to handle any new sources.
# - Output format: {"instruction": "...", "input": "...", "output": "..."}
# -----------------------------------------------------------------------------

import json
from pathlib import Path

INPUT_FILE = "model/data/raw/dev_data.json"
OUTPUT_FILE = "model/data/processed/instruction_tuning_data.jsonl"

def format_sample(item):
    source = item.get("source", "").lower()
    instruction = input_text = output = ""

    if source == "gsm8k":
        instruction = "Solve the following math word problem:"
        input_text = item.get("question", "")
        output = item.get("answer", "")
    elif source == "math":
        instruction = "Solve the following symbolic math problem:"
        input_text = item.get("question", "")
        output = item.get("solution", item.get("answer", ""))
    elif source == "hotpot_qa":
        instruction = "Answer the following question using the given context:"
        ctx = item.get("context", "")
        ctx_text = " ".join(ctx) if isinstance(ctx, list) else ctx
        input_text = f"Question: {item.get('question','')}\nContext: {ctx_text}"
        output = item.get("answer", "")
    elif source == "hhh_alignment":
        ans = item.get("answer")
        if isinstance(ans, dict):
            choices = ans.get("choices", [])
            labels = ans.get("labels", [])
            if 1 in labels:
                output = choices[labels.index(1)]
                instruction = "Provide a helpful and harmless response to the instruction."
                input_text = item.get("question", "")
        else:
            instruction = "Answer the following question using the provided evidence:"
            ctx = item.get("context", {})
            sents = ctx.get("sentences", []) if isinstance(ctx, dict) else []
            ctx_text = " ".join(" ".join(s) for s in sents)
            input_text = f"Question: {item.get('question','')}\nContext: {ctx_text}"
            output = item.get("answer", "")
    elif source == "mmlu":
        instruction = "Answer the following multiple-choice question:"
        input_text = item.get("question", "")
        output = item.get("answer", "")
    elif source == "strategy_qa":
        instruction = "Answer the following yes/no question using background knowledge:"
        input_text = item.get("question", "")
        explanation = item.get("facts", "")
        answer = "Yes" if item.get("answer") else "No"
        output = f"{explanation}\nAnswer: {answer}"
    elif source in {"trivia_qa", "truthful_qa"}:
        instruction = "Answer the following trivia question:"
        input_text = item.get("question", "")
        output = item.get("answer", "")
    else:
        instruction = "Respond appropriately to the following query:"
        input_text = item.get("question", "")
        output = item.get("answer", "")

    if not instruction or not output:
        return None

    return {"instruction": instruction, "input": input_text, "output": output}


def main():
    p = Path(INPUT_FILE)
    if p.suffix == ".jsonl":
        data = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    else:
        data = json.loads(p.read_text(encoding="utf-8"))

    processed = [s for item in data if (s := format_sample(item))]

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed)} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
