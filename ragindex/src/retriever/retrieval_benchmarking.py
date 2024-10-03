import yaml 
import pandas as pd
import os,ast
import pickle
from datasets import Dataset
from ragas import evaluate
from statistics import harmonic_mean 
import argparse
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
import logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

os.environ["OPENAI_API_KEY"] = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

class RetrievalBenchmarking:
    def __init__(self,config,datasets_dir_path):
        d={}
        
        for df_name in os.listdir(datasets_dir_path):
          if ".csv" in df_name:
            d[df_name]=pd.read_csv(os.path.join(datasets_dir_path,df_name))

        self.dict_data = d

        config_metrics = config['retriever']['retriever_benchmark_metrics']
        model_name=config["generator"]["models"]["open_ai_model"][0] or "gpt-3.5-turbo"
        logging.info("model:",model_name)
        self.llm=ChatOpenAI(model=model_name,api_key=os.environ.get("OPENAI_API_KEY"))
        default_metrics = [context_precision,context_recall]        
        self.filtered_metrics = []

        for metric in default_metrics:
            if config_metrics[metric.name]:
                self.filtered_metrics.append(metric)

    def validate_dataframe(self):
        """
        Validates the format of the input DataFrame `df_retrieval`.
        
        Performs benchmarking using RAGAS
        """
        
        max_combo = None
        max_benchmark_avg = 0
        for key, df_retrieval in self.dict_data.items():
            df_retrieval['contexts'] = df_retrieval['contexts'].apply(ast.literal_eval)

            if not isinstance(df_retrieval, pd.DataFrame):
                logging.info("Input is not a pandas DataFrame.")
                return False
            
            required_columns = ["question", "ground_truth","contexts"]
            if not set(required_columns).issubset(df_retrieval.columns):
                logging.info("DataFrame is missing one or more required columns.")
                return False
            
            testsetdf = Dataset.from_pandas(df_retrieval)
            logging.info(testsetdf,self.filtered_metrics)
            print("evkauation")
            result = evaluate(
                    dataset=testsetdf, 
                    metrics = self.filtered_metrics,
                    raise_exceptions=False,
                    llm=self.llm,
                    run_config=RunConfig(max_workers=4,timeout=10)
                    )
            scores_list = []

            for metric, score in result.items(): 
                scores_list.append(score)
            
            avg = harmonic_mean(scores_list)

            if avg >= max_benchmark_avg:
                max_benchmark_avg=avg
                max_combo=key

        for key,df in self.dict_data.items():
            if key == max_combo:
                return df,max_combo
             
            
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Retrieval Benchmarking')
    parser.add_argument('--retrieved_data_dir',type=str, required=True,help="folder should contain benchmarking dataset with columns as question ground truth and contexts ")                  
    parser.add_argument("--config", type=str, default="./config.yaml")

    args = parser.parse_args() 
    datasets_dir = args.retrieved_data_dir
    
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    df,max_combo = RetrievalBenchmarking(datasets_dir_path=datasets_dir, config=config).validate_dataframe()
    logging.info("max cobo found at{}".format(max_combo))
