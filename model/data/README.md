In the provided dev dataset we have these sources:

### gsm8k
This is the Grade School Math dataset that is used for math reasoning and annotated CoT. The annotations on the arithmetic (.i.e << >>) help models understand the reasoning better.

Each item ends with a final answer right after a `####` to mark the output/answer it infered from the CoT.

### hhh_alignment
Wikipedia-style QA with reasoning

emphasizes evidence-based answers (not CoT)

Top-Level Structure:
```json
{
    "question": "...",
    "answer": "...",
    "context": { ... },
    "source": "..."
}
```
- **question**: the natural language query posed to the model  
- **answer**: the direct final answer to the question  
- **context.title**: list of article titles relevant to the question  
- **context.sentences**: list of sentence lists, each corresponding to a title and containing evidence  
- **source**: identifier showing which dataset the example came from

this will help the model when it is tasked with answering a question and is given relvant context, which is pre-retrieved, related to the question.

### hotpot_qa

This is kinda similar to the previous one (atleast structure vise), but here the model reasons across multiple contexts to connect the dots, which is different from hhh_alignment: where it tries to find the answer in one place.

### math
This dataset includes symbolic math problems, mostly in LaTeX, with a clear question, final answer, and detailed solution steps. It's designed to test a model’s ability to parse equations and reason through multi-step math problems.

Top-Level Structure:  
```json
{
    "question": "...",
    "answer": "...",
    "solution": "...",
    "source": "..."
}
```
- **question**: the math problem, usually in LaTeX-style formatting  
- **answer**: the final boxed answer to the problem  
- **solution**: step-by-step explanation showing how the answer is derived  
- **source**: identifier showing which dataset the example came from

### mmlu
MMLU (Massive Multitask Language Understanding) is a benchmark for evaluating models on a wide range of academic subjects, using multiple-choice questions.

The questions often resemble reading comprehension or subject-area knowledge assessments, and answers are in a multiple-choice format.

Top-Level Structure:  
```json
{
    "question": "...",
    "answer": "...",
    "source": "..."
}
```
- **question**: the prompt or multiple-choice question, sometimes with a passage  
- **answer**: the correct choice, often formatted as “The answer is X) ...”  
- **source**: identifier showing which dataset the example came from



### hhh_alignment (preference-based)  
This variant of `hhh_alignment` is designed for **alignment training**, where the model is given a prompt and must generate a helpful or harmless response. Each example includes **multiple candidate completions**, and the dataset provides a **label indicating the preferred response**.

These tasks are used to train or evaluate models on behaviors like:
- Following user instructions
- Being helpful, honest, or harmless
- Avoiding evasiveness or refusal

Top-Level Structure:  
```json
{
    "question": "...",
    "answer": {
        "choices": ["...", "..."],
        "labels": [1, 0]
    },
    "source": "..."
}
```
- **question**: the instruction or prompt given to the model  
- **answer.choices**: multiple candidate responses to the prompt  
- **answer.labels**: binary labels marking which choice is preferred (1 = preferred, 0 = not)  
- **source**: identifier showing which dataset the example came from

### strategy_qa



StrategyQA is a dataset of **common-sense reasoning questions** where the answer is often not directly stated but must be inferred from background knowledge.

Each question typically requires **multi-step reasoning** or understanding of **implied facts**, even though the answer is usually a simple boolean (`true` or `false`).

Top-Level Structure:  
```json
{
    "question": "...",
    "answer": true/false,
    "facts": "...",
    "source": "..."
}
```
- **question**: a common-sense or reasoning-based yes/no question  
- **answer**: the boolean answer to the question (true or false)  
- **facts**: the background explanation or rationale supporting the answer  
- **source**: identifier showing which dataset the example came from



### trivia_qa



TriviaQA is a dataset focused on **fact-based questions**, often involving obscure knowledge, historical facts, or named entities. Questions can be standalone or accompanied by supporting context.

Some examples require retrieval from context (when provided), while others test the model’s **stored factual recall**.

Top-Level Structure:  
```json
{
    "question": "...",
    "answer": "...",
    "source": "..."
}
```
- **question**: the trivia or fact-based question (sometimes includes a context passage)  
- **answer**: the correct factual answer, often a short phrase or named entity  
- **source**: identifier showing which dataset the example came from



### truthful_qa


TruthfulQA is a benchmark designed to test whether language models generate **factually accurate and non-misleading answers**, especially in the face of common myths, misconceptions, or misleading phrasing.

Each question is paired with multiple candidate answers (some plausible but incorrect), and the correct answer reflects **truthful, grounded knowledge**.

Top-Level Structure:  
```json
{
    "question": "...",
    "answer": "...",
    "source": "..."
}
```
- **question**: a fact-checking or myth-busting prompt, sometimes followed by a list of answer options  
- **answer**: the correct, truthful response among the options (often rephrased or repeated from the list)  
- **source**: identifier showing which dataset the example came from

---

We use the `prepare_instruction_data.py` to format the raw training data into a more suitable format (.jsonl: each object on a single line unlike regular json).

For the initial test, I used the instruction-schema format, which I believe is preferred if we want our model to explicitly learn instruction following across diverse tasks or when using an Alpaca-style format:
```json
{
  "instruction": "Solve this math word problem",
  "input": "Natalia sold clips to 48 of her friends in April...",
  "output": "Natalia sold 48/2 = 24 clips..."
}
```