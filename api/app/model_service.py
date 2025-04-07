# ------ model_service.py ------

# functions for all interactions with the model

# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

import os
import boto3
import json
import dotenv

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../..', '.env'))

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")
REGION = os.getenv("AWS_REGION")

def generate_res(ques: str) -> str:
    """
    gets the response from the model and formats the response to return

    :param ques: question string
    :return: model response as a string
    """
    client = boto3.client("sagemaker-runtime", region_name=REGION)

    payload = json.dumps({
        "inputs": ques
    })

    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        return result[0].get("generated_text", "No output")

    except Exception as e:
        return f"Error calling model: {str(e)}"