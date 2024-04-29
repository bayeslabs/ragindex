from langchain_fireworks import Fireworks
import requests
from langchain.chat_models import ChatOpenAI
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import os
import yaml
from huggingface_hub import InferenceApi
from dotenv import load_dotenv
import warnings
load_dotenv()

# from main_body import data
with open('config.yaml', 'r') as file:
    data = yaml.safe_load(file)

os.environ["FIREWORKS_API_KEY"] = os.getenv("fireworks")
os.environ["OPENAI_API_KEY"] = os.getenv("openai")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("hugging_face")
class models_mod():
    def __init__(self,max_tokens=256):
        self.max_tokens = max_tokens
    
    def openai(self):
        model_name = data['generator']['models']['open_ai_model']
        llm = ChatOpenAI(model=model_name,max_tokens=self.max_tokens)  # type: ignore()
        return llm
    def fireworks(self):
        model_name = data['generator']['models']['open_source_model']
        llm = Fireworks(model=model_name,
            max_tokens=self.max_tokens)
        return llm
    
    def hugging_face(self):
        model_name =  data['generator']['models']['hugging_face_model']
        url = "https://api-inference.huggingface.co/models"
        model_location = f"{url}/{model_name}"
        response = requests.get(model_location)
        result = response.json()
        if result["gated"] == False:
            llm = HuggingFaceEndpoint(repo_id=model_name,max_new_tokens=249)  # type: ignore()
            return llm
        else:
            print("--------------------------------------------------------------------")
            warnings.warn("the repo is gated try another one or use fireworks api only") 
            print("--------------------------------------------------------------------")
        
            return "model_unknown"

    def main(self,model_type):
        if model_type=="openai":
            model = self.openai()
            return model
        elif model_type=="hugging_face":
            model = self.hugging_face()
            return model
        else:
            model = self.fireworks()
            return model
