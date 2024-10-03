from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_similarity
)
import pandas as pd
from ragas.testset.generator import TestsetGenerator
import yaml
import pandas as pd
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, multi_context
import argparse
import statistics
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class SyntheticDataGenerator:
    """
    A class for generating synthetic test datasets using language models.

    Args:
        documents (list): A list of documents to generate test data from.
        config (dict): A dictionary containing configuration parameters.
    
    Methods:
        generate_testset(num_docs=8): Generates a synthetic test dataset and saves it as a CSV file.
    """

    def __init__(self, documents, config):
        
        self.documents = documents
        model_name=config["generator"]["models"]["open_ai_model"][0] or "gpt-3.5-turbo"
        logging.info("model:",model_name)
        self.generator_llm = ChatOpenAI(model=model_name)
        
        self.critic_llm = ChatOpenAI(model=model_name,temperature=0.3)

        self.embeddings = OpenAIEmbeddings()
        self.generator = TestsetGenerator.from_langchain(
            self.generator_llm,
            self.critic_llm,
            self.embeddings,
        )
        self.save_dir = config["data"]["save_dir"]
        self.distributions = {
            simple: 0.8,
            multi_context: 0.2,
        }
        

    def generate_testset(self, num_docs=5):
        """
        Generates a synthetic test dataset and saves it as a CSV file.

        Args:
            num_docs (int, optional): Number of documents to generate (default is 5).
        
        Returns:
            Synthetic dataset amd saves a copy to the current working directory or any custom path. 
        """
        testset = self.generator.generate_with_langchain_docs(self.documents, num_docs, self.distributions, raise_exceptions=False)
        save_path=self.save_dir + "/synthetic_data/syntheticdataset.csv"
        testset.to_dataset().remove_columns(['contexts', 'evolution_type','episode_done','metadata']).to_csv(save_path)
        return save_path



class Generation_Benchmarking:
    """
   Benchmarks and evaluates response generators on a test dataset.

   Attributes:
       testset_df (pandas.DataFrame): Test dataset with questions, ground truth, and generator outputs.
       config (dict): yaml file containing relevant instructions  
    """
    def __init__(self, testset_df, config):
        self.testset_df = testset_df
        metrics = [answer_relevancy, answer_similarity]
        
        model_name=config["generator"]["models"]["open_ai_model"][0] or "gpt-3.5-turbo"
        logging.info("model:",model_name)
        self.llm = ChatOpenAI(model=model_name)
        config = config['generator']['generation_benchmark_metrics']
        self.filtered_metrics = []
        for metric in metrics:
            if config[metric.name]:
                self.filtered_metrics.append(metric)
        
        

    def run_benchmarks(self):
        """
       Evaluate response generators and return the best performer with scores.

       Returns:
           dict: Best generator column and its evaluation scores.

       Raises:
           ValueError: If required columns are missing.
        """
            
        required_columns = ['question', 'ground_truth', 'contexts']

        response_columns = [col for col in self.testset_df.columns if col not in required_columns]

        if not set(required_columns).issubset(set(self.testset_df.columns)):
            raise ValueError("The required columns 'question', 'ground_truth', and 'contexts' are missing.")
        
        dataset = Dataset.from_pandas(self.testset_df)
        generator_benchmarks = {}

        for col in response_columns:
            if col!="answer":
                dataset = dataset.rename_column(col, 'answer')
            else:
                pass
            generator_benchmarks[col] = evaluate(dataset, metrics=self.filtered_metrics, raise_exceptions=False,llm=self.llm)
            dataset = dataset.remove_columns('answer')
        
        average_scores = {}


        for combination in generator_benchmarks.keys():
            scores = list(generator_benchmarks[combination].values())
            harmonic_mean = statistics.harmonic_mean(scores)
            average_scores[combination] = harmonic_mean    
        

        best_combination = max(average_scores, key=average_scores.get)

        return {best_combination: generator_benchmarks[best_combination]}

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Generation Benchmarking')

    parser.add_argument('--testset_file', type=str, required=True,
                        help='Path to the testset file')

    args = parser.parse_args() 
    
    save_dir = pd.read_csv(args.testset_file)

    with open("config.yaml", 'r') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

    if args.testset_file:
        conf['data']['benchmark_data'] =  args.config_file
        
    gen_bench = Generation_Benchmarking(testset_df=save_dir, config=conf).run_benchmarks()
    
    logging.info(gen_bench)
