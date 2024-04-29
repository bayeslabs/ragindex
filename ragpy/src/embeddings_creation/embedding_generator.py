from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import torch.nn.functional as F
import os
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_openai import OpenAIEmbeddings
import yaml
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from itertools import product
from tqdm import tqdm
import argparse

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config

    
    def huggingface_instruct_embeddings(self, chunks, vectorstore):
        docs = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk
                )
            )
        if not docs:
            return "Chunks list is empty"
        embedding_function = HuggingFaceInstructEmbeddings()
        if vectorstore == "faiss":
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local("huggingface_instruct_embeddings_faiss")
        else:
            db = Chroma(persist_directory='huggingface_instruct_embeddings_chroma', embedding_function=embedding_function)
            db.add_documents(docs)
        return db

    def all_minilm_embeddings(self, chunks, vectorstore):
        docs = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk
                )
            )
        if not docs:
            return "Chunks list is empty"
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        if vectorstore == "faiss":
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local("all_minilm_embeddings_faiss")
        else:
            db = Chroma(persist_directory='all_minilm_embeddings_chroma', embedding_function=embedding_function)
            db.add_documents(docs)
        return db

    def bgem3_embeddings(self, chunks, vectorstore):
        docs = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk
                )
            )
        if not docs:
            return "Chunks list is empty"
        embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        if vectorstore == "faiss":
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local("bgem3_embeddings_faiss")
        else:
            db = Chroma(persist_directory="bgem3_embeddings_chroma", embedding_function=embedding_function)
            db.add_documents(docs)
        return db

    def openai_embeddings(self, chunks, vectorstore, api_key):
        docs = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk
                )
            )
        if not docs:
            return "Chunks list is empty"
        embedding_function = OpenAIEmbeddings()
        if vectorstore == "faiss":
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local("openai_embeddings_faiss")
        else:
            db = Chroma(persist_directory='openai_embeddings_chroma', embedding_function=embedding_function)
            db.add_documents(docs)
        return db
    
    def generate_databases(self,chunks):
        embedding_methods = self.config["retriever"]["vector_store"]["embedding"]
        vectorstore_option = self.config["retriever"]["vector_store"]["database"]
        chunks = self.config["retriever"]["vector_store"]["chunks"]
        databases = []
        for embedding_method in embedding_methods:
            try:
                if embedding_method == "huggingface_instruct_embeddings":
                    db = self.huggingface_instruct_embeddings(chunks, vectorstore_option)
                elif embedding_method == "all_minilm_embeddings":
                    db = self.all_minilm_embeddings(chunks, vectorstore_option)
                elif embedding_method == "bgem3_embeddings":
                    db = self.bgem3_embeddings(chunks, vectorstore_option)
                elif embedding_method == "openai_embeddings":
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if isinstance(os.environ.get("OPENAI_API_KEY"), str):
                        db = self.openai_embeddings(chunks, vectorstore_option, openai_api_key)
                    else:
                        raise ValueError("OpenAI API key is required for OpenAI embeddings.")
                else:
                    raise ValueError("Invalid embedding method specified.")

                di = {
                    "embeddings": embedding_method,
                    "vectorstore": vectorstore_option,
                    "db": db
                }
                databases.append(di)
            except Exception as e:
                print(f"Error generating database for {embedding_method} and {vectorstore_option}: {e}")

        return databases

if __name__ == "__main__":
    config_file = "/home/roopesh_d/Documents/autorag/config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", nargs='+', help="List of embedding options available are: huggingface_instruct_embeddings, all_minilm_embeddings, bgem3_embeddings, openai_embeddings")
    parser.add_argument("--vectorstore", type=str, help="Vector store option (chroma or faiss)")
    parser.add_argument("--chunks", nargs='+', help="List of text chunks")
    args = parser.parse_args()
    # print("Args",args)
    if args.embedding:
        print("embedding",args.embedding)
        config["retriever"]["vector_store"]["embedding"]= args.embedding

    # else:
    #     # If embedding is not provided, check if it exists in the config file
    #     if "embedding" in config:
    #         config["embedding"] = config["embedding"]
    #     else:
    #         # Set a default value if embedding is not in the config file
    #         config["embedding"] = ["huggingface_instruct_embeddings"]

    if args.vectorstore:
        print("db:",args.vectorstore)
        config["retriever"]["vector_store"]["database"] = args.vectorstore
    # else:
    #     # If vectorstore is not provided, check if it exists in the config file
    #     if "vectorstore" in config:
    #         config["vectorstore"] = config["vectorstore"]
    #     else:
    #         # Set a default value if vectorstore is not in the config file
    #         config["vectorstore"] = "chroma"

    
    if args.chunks:
        print("chunks:",args.chunks)
        config["retriever"]["vector_store"]["chunks"] = args.chunks
    # else:
    #     if "chunks" in config:
    #         config["chunks"] = config["chunks"]
    #     else:
    #         config["chunks"] = ["this is my chunk"]
    
    print(config)
    obj = EmbeddingGenerator(config)
    dbs= obj.generate_databases(config["retriever"]["vector_store"]["chunks"])

    for db in dbs:
        print(db)



