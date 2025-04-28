# Llama 3.2 Fine-Tuning

This repository contains the code to fine-tune a LLaMA 3.2 3B model through **instruction tuning** and **chain-of-thought reasoning**.

We utilize ASU's SOL supercomputing cluster, specifically using **4 CPUs** and a **NVIDIA A100 GPU** for training.

## Datasets Used

- **gsm8k** (main split)
- **hotpot_qa** (distractor split)
- **mandarjoshi/trivia_qa** (rc split)
- **sentence-transformers/natural-questions** (pair split)
- **domenicrosati/TruthfulQA** (default split)
- **Anthropic/hh-rlhf** (default split)

Additionally, the following BIG-Bench subsets were used:

- boolean_expressions
- causal_judgement
- date_understanding
- disambiguation_qa
- dyck_languages
- formal_fallacies
- geometric_shapes
- hyperbaton
- logical_deduction_five_objects
- logical_deduction_seven_objects
- logical_deduction_three_objects
- movie_recommendation
- multistep_arithmetic_two
- navigate
- object_counting
- penguins_in_a_table
- reasoning_about_colored_objects
- ruin_names
- salient_translation_error_detection
- snarks
- sports_understanding
- temporal_sequences
- tracking_shuffled_objects_five_objects
- tracking_shuffled_objects_seven_objects
- tracking_shuffled_objects_three_objects
- web_of_lies
- word_sorting

## SOL Setup Instructions

1. Enable Cisco VPN and connect.

2. SSH into SOL:
   ```bash
   ssh <your_asurite>@login.sol.rc.asu.edu
   ```

3. Load Mamba:
   ```bash
   module load mamba/latest
   ```

4. Create and activate the environment:
   ```bash
   mamba create -n bentune python=3.10 -y
   source activate bentune
   ```

5. Install Python packages:
   ```bash
   pip install --upgrade pip
   pip install transformers accelerate torch sentencepiece safetensors huggingface_hub datasets
   ```

6. Authenticate Hugging Face CLI:
   ```bash
   huggingface-cli login
   ```
   Paste your HF token when prompted.

7. Clone the repository:
   ```bash
   git clone https://<your-username>:<your-token>@github.com/bentune-3b/llm.git bentune
   ```

---

## Contributors

- Deep Goyal
- Namita Shah
- Jay Pavuluri
- Evan Zhu
- Navni Athale
