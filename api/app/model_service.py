# model_service.py
# ------------------------
# functions for all interactions with the model
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

from vllm import LLM, SamplingParams

llm = LLM(
    model="gpt2-medium",
    device="cpu",
    dtype= "auto",
    trust_remote_code=True
)

def generate_res(prompt: str) -> str:
    """
    function for api to input prompt and receive model response
    :param prompt: string - user input
    :return: model response
    """

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )

    outputs = llm.generate([prompt], sampling_params)
    first = outputs[0]
    generated = first.outputs[0].text

    return generated[len(prompt):].strip() if generated.startswith(prompt) else generated