# ------ model_service.py ------

# functions for all interactions with the model

# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

def generate_res(ques: str) -> str:
    """
    gets the response from the model and formats the response to return

    :param ques: question string
    :return: model response as a string
    """
    return f"You asked: {ques}"