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
role = os.getenv("AWS_ROLE")
endpoint_name = os.getenv("AWS_ENDPOINT_NAME")

#paths
model_dir = os.path.join(os.path.dirname(__file__), 'llama-3.2-3b-4bit.tar.gz')
entry_point = os.path.join(os.path.dirname(__file__), 'handler.py')
s3_model_path = sagemaker_session.upload_data(path=model_dir, key_prefix="llama-3.2-3b-model")

# make a bucket
hf_model = HuggingFaceModel(
    model_data=s3_model_path,
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    entry_point=entry_point,
    source_dir=os.path.dirname(__file__),
    sagemaker_session=sagemaker_session
)

#deploy
predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name=endpoint_name
)

print(f"Model deployed! -- success at : {predictor.endpoint_name}")