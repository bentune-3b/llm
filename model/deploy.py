# ------ deploy.py ------

# Contains the implementation to deploy the model
# in AWS SageMaker

# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import boto3
from dotenv import load_dotenv
import os

# load .env for aws configuration
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

region = os.getenv('AWS_REGION')
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# dummy model
hub = {
    'HF_MODEL_ID': 'distilgpt2',
    'HF_TASK': 'text-generation'
}

huggingface_model = HuggingFaceModel(
    env=hub,
    role=os.getenv("AWS_ROLE"),
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    sagemaker_session=sagemaker_session,
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=os.getenv("SAGEMAKER_ENDPOINT_NAME"),
)