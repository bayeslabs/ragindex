
from langchain.docstore.document import Document
import argparse
import yaml
from ragpy.src.embeddings_creation.embedding_generator import EmbeddingGenerator
import os
from ragpy.src.dataprocessing.data_loader import DataProcessor
from ragpy.src.generator.generation_benchmarking import SyntheticDataGenerator # type: ignore
import pandas as pd
from sentence_transformers import CrossEncoder



class Reranking:
    def __init__(self,config):
        self.config=config
        self.cross_encoder=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        

    def ret(self,data,top_k, config, dict_db =None, num_questions=5):
        # df_dict={}
        fpath=""
        if self.config["data"]["benchmark_data"]:
            fpath=self.config["data"]["benchmark_data"]
        else:
            documents = [Document(content) for content in data]
            s=SyntheticDataGenerator(documents,self.config)
            fpath=s.generate_testset(num_docs=num_questions)
        
        df=pd.read_csv(fpath)
        col_list=['question','ground_truth']
        try:
            if set(col_list).issubset(set(df.columns)):
                save_dir = self.config["data"]["save_dir"] + "/retrieved_data/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

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
                        # df_dict[df_name] = df
                
                    file_path = save_dir + f"{df_name}.csv"
                    df.to_csv(file_path, index=False, encoding='utf-8')
                    print("Dataframe saved to", file_path)
                
                return save_dir
        except:
            raise ValueError("Dataframe should contain tcolumns in the format of 'question','ground_truth','contexts'")

    def rerank_documents(self, query, documents, top_n):
        scores={}
        for doc in documents:
          scores[doc]=self.cross_encoder.predict([query,doc])
        
        r=list(dict(sorted(scores.items(),key= lambda item:item[1],reverse=True)))[:top_n]
        
        return r
  
if __name__ == "__main__":

    config_file='./config.yaml'
    parser = argparse.ArgumentParser(description='Retrieval and Reranking')
    parser.add_argument('--top_k',help='The number of top documents to be retrieved')
    parser.add_argument('--save_dir',help='Directory to save synthetic data')
    parser.add_argument('--benchmark_data_path',help="Path to the benchmarking dataset in csv")
    parser.add_argument('--num_questions',help="Number of questions to be generated in synthetic benchmark dataset",default=5)

    
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
    q=m.ret(chunks,config["retriever"]["top_k"],config,dict_db=dict_db, num_questions=int(args.num_questions))
    print(q)
