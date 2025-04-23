# # deploy.py
# # ------------------------
# # handles model inference
# # ------------------------
# # Team: Bentune 3b
# # Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

# import torch
# import os
# import json
# import random
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel
# from collections import Counter

# # --- configs ---
# BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"
# ADAPTER_PATH = "./model/llama-3.2-3b-4bit-finetuned"
# FEW_SHOT_EXAMPLES_PATH = "model/few_shot_examples.jsonl"

# # params
# DEFAULT_GEN_PARAMS = {
#     "max_new_tokens": 512,
#     "temperature": 0.7,
#     "top_k": 50,
#     "top_p": 0.95,
#     "do_sample": True,
#     "pad_token_id": None,
#     "eos_token_id": None,
# }

# # --- device setup ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f">>> Deploy Status: Using device: {device}")

# # --- load few shot examples ---
# few_shot_examples = []
# if os.path.exists(FEW_SHOT_EXAMPLES_PATH):
#     print(f">>> Deploy Status: Loading few-shot examples from {FEW_SHOT_EXAMPLES_PATH}")
#     with open(FEW_SHOT_EXAMPLES_PATH, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 example = json.loads(line)
#                 if all(k in example for k in ["instruction", "input", "output"]):
#                      few_shot_examples.append(example)
#             except json.JSONDecodeError:
#                 print(f">>> Deploy Warning: Skipping invalid JSON line in {FEW_SHOT_EXAMPLES_PATH}: {line.strip()}")
#     print(f">>> Deploy Status: Loaded {len(few_shot_examples)} few-shot examples.")
# else:
#     print(f">>> Deploy Warning: Few-shot examples file not found at {FEW_SHOT_EXAMPLES_PATH}. Few-shot prompting will be disabled.")

# # --- load model and tokenizer ---
# model = None
# tokenizer = None
# try:
#     print(f">>> Deploy Status: Loading base model {BASE_MODEL_ID} with quantization...")
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL_ID,
#         quantization_config=quant_config,
#         torch_dtype=torch.bfloat16,
#     )

#     print(f">>> Deploy Status: Loading PEFT adapter from {ADAPTER_PATH}...")
#     model = PeftModel.from_pretrained(model, ADAPTER_PATH)
#     model.to(device)

#     #set model to eval mode
#     model.eval()

#     print(f">>> Deploy Status: Loading tokenizer from {ADAPTER_PATH}...")
#     tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

#     #setup padding token configs
#     DEFAULT_GEN_PARAMS["pad_token_id"] = tokenizer.eos_token_id
#     DEFAULT_GEN_PARAMS["eos_token_id"] = tokenizer.eos_token_id

#     print(">>> Deploy Status: Model and Tokenizer loaded successfully.")

# except Exception as e:
#     print(f">>> Deploy Error: Failed to load model or tokenizer: {e}")
#     model = None
#     tokenizer = None

# # --- parse answers ---
# def parse_answer_from_cot(generated_text: str):
#     """
#     attempts to extract the final answer from a CoT response
#     """
#     parts = generated_text.split('####')
#     if len(parts) > 1:
#         answer_text = parts[-1].strip()
#         try:
#             return str(float(answer_text)) # Normalize numbers
#         except ValueError:
#             return answer_text # Return as string if not a simple number
#     return None # No clear final answer found

# # --- helper - build prompt ---
# def build_prompt(instruction: str, user_input: str, num_few_shot: int = 0):
#     """
#     constructs the full prompt string including instruction, few-shot examples, and user input.
#     """
#     prompt_parts = []

#     if num_few_shot > 0 and few_shot_examples:
#         selected_examples = random.sample(few_shot_examples, min(num_few_shot, len(few_shot_examples)))
#         for example in selected_examples:
#             prompt_parts.append(f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}")

#     prompt_parts.append(f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:")

#     # Use the training format prefix
#     full_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n" + "\n\n".join(prompt_parts)

#     return full_prompt

# # --- api facing function ---
# def generate_response(
#     instruction: str,
#     user_input: str,
#     num_few_shot: int = 0,
#     num_self_consistency_samples: int = 1,
#     gen_params: dict = None
# ):
#     """
#     function for api interface

#     :param instruction: instruction tuning string
#     :param user_input: user prompt string
#     :param num_few_shot: number of few-shot examples
#     :param num_self_consistency_samples: self consistency samples
#     :param gen_params: overrides
#     :return:
#     """
#     if model is None or tokenizer is None:
#         print(">>> Deploy Error: Model or Tokenizer not loaded. Cannot generate response.")
#         return "Error: Model not available.", []

#     current_gen_params = DEFAULT_GEN_PARAMS.copy()
#     if gen_params:
#         current_gen_params.update(gen_params)

#     if num_self_consistency_samples > 1:
#         current_gen_params["do_sample"] = True
#         if current_gen_params.get("temperature", 0) <= 0.01:
#              current_gen_params["temperature"] = 0.7

#     prompt_text = build_prompt(instruction, user_input, num_few_shot)
#     input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

#     generated_texts = []
#     parsed_answers = []

#     print(f">>> Deploy Status: Generating {num_self_consistency_samples} response(s)...")
#     with torch.no_grad():
#         for i in range(num_self_consistency_samples):
#             print(f">>> Deploy Status: Generating sample {i+1}/{num_self_consistency_samples}...")
#             try:
#                 output_ids = model.generate(
#                     input_ids,
#                     **current_gen_params,
#                 )
#                 generated_sequence = output_ids[0, input_ids.shape[-1]:]
#                 generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
#                 generated_texts.append(generated_text)

#                 parsed_answer = parse_answer_from_cot(generated_text)
#                 parsed_answers.append(parsed_answer)
#                 print(f">>> Deploy Status: Sample {i+1} generated. Parsed Answer: {parsed_answer}")

#             except Exception as e:
#                 print(f">>> Deploy Error: Generation failed for sample {i+1}: {e}")
#                 generated_texts.append(f"Error during generation for sample {i+1}: {e}")
#                 parsed_answers.append(None)

#     # --- self consistency! ---
#     if num_self_consistency_samples > 1:
#         print(">>> Deploy Status: Performing self-consistency voting...")
#         valid_answers = [ans for ans in parsed_answers if ans is not None and str(ans).strip()]

#         if valid_answers:
#             answer_counts = Counter(valid_answers)
#             most_common_answer = answer_counts.most_common(1)[0][0]
#             print(f">>> Deploy Status: Most common parsed answer: {most_common_answer}")

#             for i, answer in enumerate(parsed_answers):
#                  if answer == most_common_answer:
#                      print(f">>> Deploy Status: Returning full text from sample {i+1}.")
#                      return generated_texts[i], generated_texts
#             # Fallback
#             print(">>> Deploy Warning: Could not find matching text for most common answer, returning first sample.")
#             return generated_texts[0], generated_texts
#         else:
#             print(">>> Deploy Warning: No valid answers parsed. Returning first sample.")
#             return generated_texts[0], generated_texts
#     else:
#         return generated_texts[0], generated_texts

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

model_path = os.path.expanduser("model/llama-3.2-3b-4bit-finetuned")

tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.to(device)
model.eval()

while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"exit", "quit"}:
        break
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model:", response)