import yaml 
import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    context_precision,
    context_recall,
)

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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class DataFrameValidator:
    def __init__(self):
        pass

    def validate_dataframe(self, df_retrieval, config_file):
        """
        Validates the format of the input DataFrame `df_retrieval`.
        
        Performs benchmarking using RAGAS
        """
        
        # with open(config_file, 'r') as file:
        #     return yaml.safe_load(file)
        # metrics = config['metrics']
        # for key,value in config_file['retriever']['retriever_benchmark_metrics'].items():
        #     if value == True:
        #         metrics.append(key)

        metrics = [context_precision,context_recall]        
        config = config_file['retriever']['retriever_benchmark_metrics']
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
        testset_df = Dataset.from_pandas(df_retrieval)
        result = evaluate(
                dataset=testset_df, 
                metrics = filtered_metrics,
                )
        print(result)
        

# obj = DataFrameValidator()
# obj.validate_dataframe(df_retrieval,config_file)




