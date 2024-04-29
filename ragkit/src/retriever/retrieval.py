# from langchain.docstore.document import Document
from FlagEmbedding import FlagModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder as ce
from langchain.docstore.document import Document
import cohere
import argparse
import os
import sys
import pickle
from langchain.text_splitter import CharacterTextSplitter
sys.path.append('/teamspace/studios/this_studio/ragKIT/src')
import yaml
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from embedding_creation.embedding_generator import EmbeddingGenerator 
from PyPDF2 import PdfReader
from DataPreprocessing.data_loader import DataProcessor
from Generation.generation_benchmarking import SyntheticDataGenerator
import cohere
import pandas as pd
from cohere.types.rerank_response import RerankResponse
from langchain_community.document_loaders import PyPDFLoader


class Reranking:
    def __init__(self,config):
        self.config=config
        self.client = cohere.Client(os.getenv('COHERE_API_KEY'))
        

    def ret(self,data,top_k,config,dict_db =None):
        df_dict={}
        fpath=""
        if self.config["data"]["benchmark_data"]:
            fpath=self.config["data"]["benchmark_data"]
        else:
            self.config["data"]["benchmark_data"]=os.getcwd()
            documents = [Document(content) for content in data]
            s=SyntheticDataGenerator(documents,self.config)
            fpath=s.generate_testset()
        for d in dict_db:  
            df=pd.read_csv(fpath)     
            df['contexts']=''
            # df['Generated Query']=''
            df_name = f'{d["embeddings"]}-{d["vectorstore"]}'
            for k in range(0,df.shape[0]):
                query=df.iloc[k,0]
                db=d["db"]
                docs=db.similarity_search_with_relevance_scores(query)
                l1=[]
                for doc in docs:
                    l1.append(doc[0].page_content)
                reranked_documents=self.rerank_documents(query,l1,top_k)
                l2=[]
                for doc in reranked_documents:
                    l2.append(doc)
                df.at[k,'contexts']=l2
                df_dict[df_name] = df
                file_path = f'ragKIT/savedireectory/{df_name}.csv'
                df.to_csv(file_path, index=False)
                
        return df_dict
        
    def rerank_documents(self, query, documents, top_n):
        # print("reranking called")
        # api_key = os.getenv('COHERE_API_KEY')
        # client=cohere.Client(api_key)
        results = self.client.rerank(model="rerank-english-v3.0", query=query, documents=documents, top_n=top_n, return_documents=True)
        r1 = results.results
        document_texts = [result.document.text for result in r1]
        r=[]
        for text in document_texts:
            r.append(text)
        return r
  
if __name__ == "__main__":

    config_file='/teamspace/studios/this_studio/ragKIT/config/sample_config.yaml'
    parser = argparse.ArgumentParser(description='Retrieval and Reranking')
    # parser.add_argument("--config", type=str, default="/teamspace/studios/this_studio/ragKIT/config/sample_config.yaml",
    #                     help="Path to the configuration file")
    parser.add_argument('--top_k',help='The number of chunks to be retrieved',required=True)
    args = parser.parse_args()

    with open(config_file, 'r') as file: 
        config = yaml.safe_load(file)

    if args.top_k:
        config["retriever"]["top_k"] = int(args.top_k)
    
    c=DataProcessor(config)
    chunks=c.process_data()
    obj=EmbeddingGenerator(config)
    dict_db=obj.generate_databases(chunks)

    m=Reranking(config)
    q=m.ret(chunks,config["retriever"]["top_k"],config,dict_db=dict_db)
    print(q)