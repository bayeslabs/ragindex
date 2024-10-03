import argparse
import requests
from langchain.chat_models import ChatOpenAI
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import os
import yaml
from huggingface_hub import InferenceApi
from dotenv import load_dotenv
import warnings
import sys 
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class models_mod():
    def __init__(self,max_tokens=256,config=None):
        self.max_tokens = max_tokens
        self.data = config       
        if self.data is not None:
            self.temp = self.data["generator"]["model_config"]["temperature"]

    def openai(self,temp=None,model_name=None):
        try:
            llm = ChatOpenAI(model=model_name,temperature=temp) 
            return llm
        except Exception as e:  
            logging.info(e) 
            sys.exit(1)
            

    def hugging_face(self,temp=None,model_name=None):
        try:
            url = "https://api-inference.huggingface.co/models"
            model_location = f"{url}/{model_name}"
            response = requests.get(model_location)
            if response.status_code == 200:
                logging.info("response", response.status_code)
                result = response.json()
                if result["gated"] == False:
                    logging.info(temp)
                    llm = HuggingFaceEndpoint(repo_id=model_name, max_new_tokens=249, temperature=temp)
                    return llm
                else:
                    logging.info("The repo is gated. Try with another one.")
                    sys.exit(1)
            else:
                logging.info("Check for the API or maybe your internet connection is wonky.")
                sys.exit(1)
        except Exception as e:
            logging.info(f"An error occurred: {e}")
            sys.exit(1)

    def main(self,model_type,model_name,temp):
        if model_type=="openai":
            model = self.openai(temp,model_name)
            return model
        else:
            if isinstance(temp, list): 
                models = []
                for t in temp:
                    models.append(self.hugging_face(t, model_name))
                return models 
            else:
                return self.hugging_face(temp, model_name)


if __name__ == "__main__": 
    with open('./config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description='Select the model type for the generator.')
    parser.add_argument('--model_type', type=str,default = "openai", choices=['openai', 'hugging_face'],
                        help='The type of model to use. Can be "openai", "hugging_face"')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help='The name of the model to use.')
    parser.add_argument('--temperature', type=float, nargs='+', default=0.7, help="Temperature of the model. Default is 0.7.")

    args = parser.parse_args()

    logging.info(args.model_type)
    logging.info(args.model_name)
    if args.model_type:
        data["generator"]["models"]["model_type"] = args.model_type
    if args.model_name:
        data["generator"]["models"]["model_name"] = args.model_name
    if args.temperature:
        data["generator"]["model_config"]["temperature"]=args.temperature

    method=models_mod(data)
    llms={}
    if len(data["generator"]["model_config"]["temperature"])==1:
        temp=data["generator"]["model_config"]["temperature"]
        llms[temp[0]]=method.main(data["generator"]["models"]["model_type"], data["generator"]["models"]["model_name"], temp=temp[0])
    else:
        for temo in data["generator"]["model_config"]["temperature"]:
            llms[temo]=method.main(data["generator"]["models"]["model_type"],data["generator"]["models"]["model_name"],temp=temo)
    logging.info("llms are loaded", llms)
