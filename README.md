<div align="center">
  <img src="assets/bentune.png" alt="BenTune Logo" width="600" style="margin-bottom: -10px;" />
  <p style="margin-top: 0;">An instruction-tuned LLaMA-3.2B assistant developed at Arizona State University</p>
</div>

---
## Overview

BenTune-3B is a lightweight, fine-tuned version of Meta’s LLaMA-3.2B model, optimized for general-purpose instruction following. Developed entirely on ASU’s SOL cluster, this project aims to explore low-latency inference, modular fine-tuning, and evaluation of open-weight LLMs under academic resource constraints.

The model is capable of answering diverse instruction-based prompts ranging from logical reasoning and factual queries to summarization and coding tasks.
<div align="center">
<img src="assets/bentune is all you need.png" alt="Report Cover" width="600" style="margin-bottom: -10px;" />
</div>

---
## BenTune v1 vs v2

| Feature                          | **BenTune v1**                                                  | **BenTune v2**                                                                |
|----------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Training Samples                 | 85,000 examples                                                  | 135,000 examples                                                              |
| Epochs                           | 3 epochs                                                         | 2 epochs                                                                      |
| Additional Datasets              | None                                                             | 3 small coding datasets; more knowledge-heavy and safety-aligned sources      |
| Focus Areas                      | General instruction-following and reasoning                      | Knowledge coverage and safety alignment                                       |
| Response Quality (General/Reasoning) | Higher quality responses due to focused dataset                  | Slightly degraded in general/reasoning tasks due to distributional shift      |
| Alignment Behavior               | Less constrained, more natural responses                         | More cautious and safety-aware responses                                       |

> *Note:* While v2 saw improvements in factuality and safety alignment, it often produced more conservative or less precise outputs on abstract or multi-step reasoning tasks. BenTune v1, with its narrower dataset focus, generated more robust general-purpose and logic-based completions.
---
## ARC Evaluation Comparison

**Legend**  
- **ZS (Zero-shot):** Model receives the question only, with no examples.  
- **5-shot:** Model is shown five examples before the test question.  
- **CoT (Chain-of-Thought):** Prompts include intermediate reasoning steps to encourage logical breakdowns.

| **Evaluation Setting**           | **Meta Instruct** | **BenTune v1** | **BenTune v2** |
|----------------------------------|-------------------|----------------|----------------|
| ARC-Challenge (Zero-shot)        | **43.69%**        | 40.70%         | 38.74%         |
| ARC-Challenge (5-shot)           | **46.67%**        | 42.15%         | 41.64%         |
| ARC-Challenge (Zero-shot + CoT)  | **41.89%**        | 39.51%         | 36.86%         |
| ARC-Challenge (5-shot + CoT)     | **46.16%**        | **43.17%**     | 43.00%         |
| ARC-Easy (Zero-shot)             | 73.95%            | **74.41%**     | 71.25%         |


### Highlights

- **BenTune v1** delivers performance close to **Meta Instruct**, especially in 5-shot + CoT, and surpasses it on **ARC-Easy**, demonstrating strong general reasoning.
- **BenTune v2** shows slightly lower accuracy across all settings, reflecting a tradeoff made for improved alignment and safety handling.
- **Meta Instruct** leads in most settings, but **BenTune v1** performs competitively despite having significantly fewer training examples and using PEFT-based fine-tuning.

### Interpretation

- The results show that **BenTune v1** effectively captures logical and general-purpose reasoning with compact training.
- **BenTune v2**, while better aligned for safety and factual knowledge, sacrifices performance on abstract or logic-heavy benchmarks.
---
## Inference Strategies

We tested three key decoding techniques during inference:

- **Forced Chain-of-Thought (CoT)**  
- **Dynamic CoT Triggering**  
- **Self-Consistency Decoding**

Performance varied based on the nature of the prompt. While some reasoning tasks benefited from CoT strategies, others did not show significant improvement. Due to time constraints, we were unable to fully explore these behaviors across all benchmark categories.


---

## Model Response Comparison

You can view all the question responses from **Meta Instruct**, **BenTune v1**, and **BenTune v2** side by side [here](https://github.com/bentune-3b/.github/blob/main/profile/comparison.md). The questions cover different categories like instruction following, reasoning, and factual QA.

### Quick Contrast

- **BenTune v1** gave the most natural and solid completions overall. Especially for general-purpose tasks and logic-based prompts, it was often clearer and more direct.
- **BenTune v2** was safer and more factual, but sometimes a bit too cautious. In a few reasoning cases, it overexplained or lost track of the main question.
- **Meta Instruct** still leads in a few settings, but BenTune v1 got surprisingly close despite being trained on fewer samples and with lighter compute.

We didn’t get time to deeply test all decoding modes across every prompt, but the differences in tone and style across models were very noticeable. Each has its own strengths depending on the kind of task you're running.

---

## Team

- Jaya Adithya Pavuluri  
- Deep Goyal  
- Namita Shah  
- Evan Zhu  
- Navni Athale  

Project developed for CSE 476: Intro to NLP at Arizona State University.

