from langchain.docstore.document import Document
import argparse
import yaml
from ragindex.src.embeddings_creation.embedding_generator import EmbeddingGenerator
import os
from ragindex.src.dataprocessing.data_loader import DataProcessor
from ragindex.src.generator.generation_benchmarking import SyntheticDataGenerator
import pandas as pd
from sentence_transformers import CrossEncoder
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

class Reranking:
    def __init__(self, config):
        self.config = config
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.flashrank_reranker = FlashrankRerank()

    def ret(self, data, top_k, config, dict_db=None,num_questions=None):
        reranker_methods = self.config["retriever"]["rerankers"]
        if isinstance(reranker_methods, str):
            reranker_methods = [reranker_methods]
        fpath = ""
        if self.config["data"]["benchmark_data"]:
            fpath = self.config["data"]["benchmark_data"]
        else:
            documents = [Document(content) for content in data]
            s = SyntheticDataGenerator(documents, self.config)
            fpath = s.generate_testset(num_docs=len(documents))

        df = pd.read_csv(fpath)
        col_list = ['question', 'ground_truth']

        try:
            if set(col_list).issubset(set(df.columns)):
                save_dir = self.config["data"]["save_dir"] + "/retrieved_data/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                df['contexts'] = ''
                query_docs = []

                for k in range(0, df.shape[0]):
                    query = df.iloc[k, 0]
                    for d in dict_db:
                        db = d["db"]
                        docs = db.similarity_search_with_relevance_scores(query)
                        query_docs.append((query, [doc[0].page_content for doc in docs]))

                reranked_documents_list = []
                for reranker_method in reranker_methods: 
                    logging.info(f"Processing {len(query_docs)} queries with reranker method: {reranker_method}")
                    
                    if reranker_method == 'cross_encoder':
                        reranked_documents = self.rerank_documents_cross_encoder(query_docs, top_k)
                    elif reranker_method == 'flashrank':
                        reranked_documents = self.rerank_documents_flashrank(query_docs, d["db"])
                    else:
                        raise ValueError(f"Invalid reranker method: {reranker_method}")
                    reranked_documents_list.extend(reranked_documents)

                for k, reranked_docs in enumerate(reranked_documents_list):
                    df.at[k, 'contexts'] = reranked_docs

                df_name = f'{dict_db[0]["embeddings"]}_{dict_db[0]["vectorstore"]}_{",".join(reranker_methods)}'
                file_path = save_dir + f"{df_name}.csv"
                df.to_csv(file_path, index=False, encoding='utf-8')
                logging.info("Dataframe saved to", file_path)

                return save_dir

        except Exception as e:
            logging.info("Error:", e)
            raise ValueError("Dataframe should contain columns in the format of 'question','ground_truth','contexts'")
            
    def rerank_documents_cross_encoder(self, query_docs, top_n):
        logging.info("executing cross encoder reranking")
        reranked_documents = []
        for query, documents in query_docs:
            scores = {}
            for doc in documents:   
                scores[doc] = self.cross_encoder.predict([query, doc])
            reranked_docs = list(dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)))[:top_n]
            reranked_documents.append(reranked_docs)
        return reranked_documents 

    def rerank_documents_flashrank(self, query_docs, vector_store):
        logging.info("executing flashrank reranking")
        reranked_documents = []
        for query, documents in query_docs:
            retriever = vector_store.as_retriever(search_kwargs={"k": len(documents)})
            compression_retriever = ContextualCompressionRetriever(base_compressor=self.flashrank_reranker,
                                                                   base_retriever=retriever)
            reranked_docs = [doc.page_content for doc in compression_retriever.invoke(query)]
            reranked_documents.append(reranked_docs)
        return reranked_documents

if __name__ == "__main__":
    config_file = './config.yaml' 
    parser = argparse.ArgumentParser(description='Retrieval and Reranking')
    parser.add_argument('--top_k', help='The number of top documents to be retrieved')
    parser.add_argument('--save_dir', help='Directory to save synthetic data')
    parser.add_argument('--benchmark_data_path', help="Path to the benchmarking dataset in csv")
    parser.add_argument('--num_questions', help="Number of questions to be generated in synthetic benchmark dataset", default=5)
    parser.add_argument('--reranker_methods', nargs='+', choices=['cross_encoder', 'flashrank'], default='cross_encoder',
                        help='The reranker method to use')

    args = parser.parse_args()

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    if args.top_k:
        config["retriever"]["top_k"] = int(args.top_k)

    if args.benchmark_data_path:
        config["data"]["benchmark_data"] = args.benchmark_data_path

    if args.save_dir:
        config["data"]["save_dir"] = args.save_dir
    
    if args.reranker_methods:
        config["retriever"]["rerankers"] = args.reranker_methods

    c = DataProcessor(config)
    chunks = c.process_data()
    obj = EmbeddingGenerator(config)
    dict_db = obj.generate_databases(chunks)
    m = Reranking(config)
    q = m.ret(chunks, config["retriever"]["top_k"], config, dict_db=dict_db)
    logging.info(q)
