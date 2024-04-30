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

# dotenv_path = '.\src\.env'
# load_dotenv(dotenv_path)
# with open('./config/sample_config.yaml', 'r') as file:
#     data = yaml.safe_load(file)

# os.environ["FIREWORKS_API_KEY"] = os.getenv("fireworks")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("hugging_face")

class models_mod():
    def __init__(self,max_tokens=256,config=None):
        self.max_tokens = max_tokens
        self.data = config
        # self.temp = temp
        # self.model_name = model_name
    
    def openai(self,temp=None,model_name=None):
        # model_name = data['generator']['models']['open_ai_model']
        llm = ChatOpenAI(model=model_name,temperature=temp)  # type: ignore()
        return llm
    # def fireworks(self):
    #     model_name = data['generator']['models']['open_source_model']
    #     llm = Fireworks(model=model_name,
    #         max_tokens=self.max_tokens)
    #     return llm
    
    def hugging_face(self,temp=None,model_name=None):
        # model_name =  data['generator']['models']['hugging_face_model']
        url = "https://api-inference.huggingface.co/models"
        model_location = f"{url}/{model_name}"
        response = requests.get(model_location)
        result = response.json()
        if result["gated"] == False:
            llm = HuggingFaceEndpoint(repo_id=model_name,max_new_tokens=249,temp = temp)  # type: ignore()
            return llm
        else:
            print("--------------------------------------------------------------------")
            warnings.warn("the repo is gated try another one or use fireworks api only") 
            print("--------------------------------------------------------------------")
        
            return "model_unknown"

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
    parser.add_argument('--model_type', type=str,default = "openai", choices=['openai', 'hugging_face', 'fireworks'],
                        help='The type of model to use. Can be "openai", "hugging_face", or "fireworks".')
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo", help='The name of the model to use.')
    parser.add_argument('--temp', type=float, default=0.7, help='The temperature for the model.')
    # Parse the arguments
    args = parser.parse_args()
    data["generator"]["models"]["model_type"] = args.model_type
    if data["generator"]["models"]["model_type"]:
        data["generator"]["models"]["model_type"] = args.model_type

    # Initialize the models_mod class
    models = models_mod()
    # Select the model based on the command line argument
    model = models.main(args.model_type,args.model_name,args.temp)

    print(model)
