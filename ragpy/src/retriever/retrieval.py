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
import yaml
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from ragpy.src.embeddings_creation.embedding_generator import EmbeddingGenerator
from PyPDF2 import PdfReader
from ragpy.src.dataprocessing.data_loader import DataProcessor
from ragpy.src.generator.generation_benchmarking import SyntheticDataGenerator
import cohere
import pandas as pd
from cohere.types.rerank_response import RerankResponse
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import CrossEncoder


class Reranking:
    def __init__(self,config):
        self.config=config
        self.cross_encoder=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        

    def ret(self,data,top_k,config,dict_db =None):
        df_dict={}
        fpath=""
        if self.config["data"]["benchmark_data"]:
            fpath=self.config["data"]["benchmark_data"]
        else:
            documents = [Document(content) for content in data]
            s=SyntheticDataGenerator(documents,self.config)
            fpath=s.generate_testset()
        df=pd.read_csv(fpath)
        for d in dict_db:    
            df['contexts']=''
            # df['Generated Query']=''
            df_name = f'{d["embeddings"]}_{d["vectorstore"]}'
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
            file_path = self.config["data"]["save_dir"]+"/retrieved_data/{}.csv".format(df_name)
            df.to_csv(file_path,index=False,encoding='utf-8')
            print("Dataframe saved to",file_path)
        retrieved_data_path=config["data"]["save_dir"]+"/retrieved_data/"      
        return retrieved_data_path
        
    def rerank_documents(self, query, documents, top_n):
        # print("reranking called")
        # api_key = os.getenv('COHERE_API_KEY')
        # client=cohere.Client(api_key)
        scores={}
        # results = self.client.rerank(model="rerank-english-v3.0", query=query, documents=documents, top_n=top_n, return_documents=True)
        for doc in documents:
          scores[doc]=self.cross_encoder.predict([query,doc])
        
        r=list(dict(sorted(scores.items(),key= lambda item:item[1],reverse=True)))[:top_n]
        
        return r
  
if __name__ == "__main__":

    config_file='./config.yaml'
    parser = argparse.ArgumentParser(description='Retrieval and Reranking')
    # parser.add_argument("--config", type=str, default="/teamspace/studios/this_studio/ragKIT/config/sample_config.yaml",
    #                     help="Path to the configuration file")
    parser.add_argument('--top_k',help='The number of top documents to be retrieved')
    parser.add_argument('--save_dir',help='Directory to save synthetic data')
    parser.add_argument('--benchmark_data_path',help="Path to the benchmarking dataset in csv")
    

    args = parser.parse_args()

    with open(config_file, 'r') as file: 
        config = yaml.safe_load(file)

    if args.top_k:
        config["retriever"]["top_k"] = int(args.top_k)
    if args.benchmark_data_path:
        config["data"]["benchmark_data"]=args.benchmark_data_path
    if args.save_dir:
        config["data"]["save_dir"]=args.save_dir
    
    c=DataProcessor(config)
    chunks=c.process_data()
    obj=EmbeddingGenerator(config)
    dict_db=obj.generate_databases(chunks)

    m=Reranking(config)
    q=m.ret(chunks,config["retriever"]["top_k"],config,dict_db=dict_db)
    print(q)
