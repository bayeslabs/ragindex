import yaml 
import pandas as pd
import os,ast
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

class RetrievalBenchmarking:
    def __init__(self,config,datasets_dir_path):
        self.config = config
        d={}
        
        for df_name in os.listdir(datasets_dir_path):
          if ".csv" in df_name:
            d[df_name]=pd.read_csv(os.path.join(datasets_dir_path,df_name)).head(1)

        self.dict_data = d
    
    def validate_dataframe(self):
        """
        Validates the format of the input DataFrame `df_retrieval`.
        
        Performs benchmarking using RAGAS
        """
        
        max_combo = None
        max_benchmark_avg = 0
        for key, df_retrieval in self.dict_data.items():
            default_metrics = [context_precision,context_recall]        
            # print(type(df_retrieval))
            # print(df_retrieval.head(1))
            df_retrieval['contexts'] = df_retrieval['contexts'].apply(ast.literal_eval)
            config_metrics = self.config['retriever']['retriever_benchmark_metrics']
            filtered_metrics = []

            for metric in default_metrics:
                if config_metrics[metric.name]:
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
    print("max cobo found at{}".format(max_combo))




