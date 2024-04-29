from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    answer_correctness
)
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from ragas.testset.generator import TestsetGenerator
from ragas.metrics import answer_relevancy, answer_similarity, answer_correctness
import yaml
import ast
import sys
import pandas as pd
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, multi_context
import os



class SyntheticDataGenerator:
    def __init__(self, documents, save_dir):
        self.documents = documents
        self.generator_llm = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0.3)
        self.critic_llm = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0.3)
        self.embeddings = OpenAIEmbeddings()
        self.generator = TestsetGenerator.from_langchain(
            self.generator_llm,
            self.critic_llm,
            self.embeddings,
        )
        self.save_dir = save_dir
        self.distributions = {
            simple: 0.8,
            multi_context: 0.2,
        }

    def generate_testset(self, num_docs=8):
        testset = self.generator.generate_with_langchain_docs(self.documents, num_docs, self.distributions, raise_exceptions=False)
        return testset.to_dataset().remove_columns(['contexts', 'evolution_type','episode_done','metadata']).to_csv(self.save_dir + "syntheticdataset.csv")

class Generation_Benchmarking:
    def __init__(self, testset_df, config):

        self.testset_df = testset_df
        metrics = [answer_relevancy,answer_similarity,answer_correctness]        
        config = config['generator']['generation_benchmark_metrics']
        self.filtered_metrics = []

        for metric in metrics:
            if config[metric.name]:
                self.filtered_metrics.append(metric)

    def run_benchmarks(self):
        desired_columns = ['question', 'ground_truth','contexts','answer']
        if list(self.testset_df.columns) != desired_columns:
            raise ValueError("Column names do not match the desired names.")

        generator_benchmark = evaluate(
            Dataset.from_pandas(self.testset_df),
            metrics= self.filtered_metrics,
            raise_exceptions=False
        )

        return generator_benchmark

if __name__=="__main__":
    with open('ragKIT/config/sample_config.yaml', 'r') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
 
    save_dir = pd.read_csv("/teamspace/studios/this_studio/ragKIT/Data/syntheticdataset1.csv")
    save_dir['contexts'] = save_dir['contexts'].apply(ast.literal_eval)
    
    gen_bench = Generation_Benchmarking(testset_df=save_dir,config=conf).run_benchmarks()

    print(gen_bench)



