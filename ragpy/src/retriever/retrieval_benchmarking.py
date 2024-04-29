import yaml 
import pandas as pd
import os
import pickle
from datasets import Dataset
from ragas import evaluate
from statistics import mean 
import argparse
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    context_precision,
    context_recall,
)
# with open('/teamspace/studios/this_studio/ragKIT/Data/mydict.pkl', 'rb') as fp:
#     dict_data = pickle.load(fp)
    
# with open('config.yaml', 'r') as file:
#     config_file = yaml.safe_load(file)

# import pandas as pd

# # Sample data
# data = {
#     'question': [
#         'What is the capital of France?'
       
#     ],
#     'ground_truth': [
#         'The capital of France is Paris.'
        
#     ],
#     'contexts': [[
#         'Paris is the capital and most populous city of France. It has an area of 105 square kilometers and a population of 2,161,000 residents.',
#         'Harper Lee was an American novelist best known for her 1960 novel To Kill a Mockingbird. It won the Pulitzer Prize for Fiction in 1961 and has become a classic of modern American literature.',
#         'Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined.'
#     ]]
# }

# # Create the DataFrame
# df_retrieval = pd.DataFrame(data)
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

class DataFrameValidator:
    def __init__(self,config,testset_df):
        self.config = config
        self.dict_data = testset_df
    
    def validate_dataframe(self):
        """
        Validates the format of the input DataFrame `df_retrieval`.
        
        Performs benchmarking using RAGAS
        """
        
        max_combo = None
        max_benchmark_avg = 0
        for key, df_retrieval in self.dict_data.items():
            metrics = [context_precision,context_recall]        
            print(type(df_retrieval))
            print(df_retrieval.head(1))
            config = self.config['retriever']['retriever_benchmark_metrics']
            filtered_metrics = []

            for metric in metrics:
                if config[metric.name]:
                    filtered_metrics.append(metric)

            # Check if the input is a DataFrame
            if not isinstance(df_retrieval, pd.DataFrame):
                print("Input is not a pandas DataFrame.")
                return False
            
            # Check if the required columns exist
            required_columns = ["question", "ground_truth","contexts"]
            if not set(required_columns).issubset(df_retrieval.columns):
                print("DataFrame is missing one or more required columns.")
                return False
            testsetdf = Dataset.from_pandas(df_retrieval)
            result = evaluate(
                    dataset=testsetdf, 
                    metrics = filtered_metrics,
                    )
            scores_list = []
            for metric, score in result.items(): 
                scores_list.append(score)
            avg = mean(scores_list)
            print("average_metrics", avg)
            if avg > max_benchmark_avg:
                max_benchmark_avg=avg
                max_combo=key

        for key,df in self.dict_data.items():
            if key == max_combo:
                return df

             
            
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Retrieval Benchmarking')
    parser.add_argument('--dict_data',type=str, required=True)                  
    parser.add_argument("--config", type=str, default="/teamspace/studios/this_studio/ragpy/config/sample_config.yaml")

    args = parser.parse_args() 
    dataset = args.dict_data

    config_path = "/teamspace/studios/this_studio/ragpy/config/sample_config.yaml"

    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    df = DataFrameValidator(testset_df=dataset, config=config).validate_dataframe()



# obj = DataFrameValidator()
# obj.validate_dataframe(dict_data,config_file)




