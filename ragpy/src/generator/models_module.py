import argparse
from langchain_fireworks import Fireworks
import requests
from langchain.chat_models import ChatOpenAI
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import os
import yaml
from huggingface_hub import InferenceApi
from dotenv import load_dotenv
import warnings
import sys
class models_mod():
    def __init__(self,max_tokens=256,config=None):
        self.max_tokens = max_tokens
        self.data = config
        if self.data is not None:
            self.temp = self.data["generator"]["model_config"]["temperature"]

    def openai(self,temp=None,model_name=None):
        try:
            llm = ChatOpenAI(model=model_name,temperature=temp)  # type: ignore()
            return llm
        except Exception as e:
            print(e)
            sys.exit(1)  # Exit with a non-zero status code
            

    def hugging_face(self,temp=None,model_name=None):
        try:
            url = "https://api-inference.huggingface.co/models"
            model_location = f"{url}/{model_name}"
            response = requests.get(model_location)
            if response.status_code == 200:
                print("response", response.status_code)
                result = response.json()
                if result["gated"] == False:
                    llm = HuggingFaceEndpoint(repo_id=model_name, max_new_tokens=249, temp=temp)
                    return llm
                else:
                    print("The repo is gated. Try with another one.")
                    sys.exit(1)  # Exit with a non-zero status code
            else:
                print("Check for the API or maybe your internet connection is wonky.")
                sys.exit(1)  # Exit with a non-zero status code
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)  # Exit with a non-zero status code

    def main(self,model_type,model_name,temp):
        if model_type=="openai":
            model = self.openai(temp,model_name)
            return model
        else:
            model = self.hugging_face(temp,model_name)
            return model
if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    # Create the parser
    parser = argparse.ArgumentParser(description='Select the model type for the generator.')
    parser.add_argument('--model_type', type=str,default = "openai", choices=['openai', 'hugging_face'],
                        help='The type of model to use. Can be "openai", "hugging_face"')
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo", help='The name of the model to use.')
    parser.add_argument('--temperature',nargs='+',default=[0.7,0.1],help="Temperature of the model. Default is 0.7.")
    # Parse the arguments
    args = parser.parse_args()
    
    if args.model_type:
        data["generator"]["models"]["model_type"] = args.model_type
    if args.model_name:
        data["generator"]["models"]["model_name"] = args.model_name
