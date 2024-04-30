from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import torch.nn.functional as F
import os
import itertools
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
from langchain_openai import OpenAIEmbeddings
import yaml
# from dotenv import load_dotenv
# load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from itertools import product
from tqdm import tqdm
import argparse

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

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
        if vectorstore == "Faiss":
            persist_directory = os.path.join(self.config["retriever"]["vector_store"]["persist_directory"][0], "huggingface_instruct_embeddings_faiss")
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local(persist_directory)
        else:
            persist_directory = os.path.join(self.config["retriever"]["vector_store"]["persist_directory"][0], "huggingface_instruct_embeddings_chroma")
            db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
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
        if vectorstore== "Faiss":
            persist_directory = os.path.join(self.config["retriever"]["vector_store"]["persist_directory"][0], "all_minilm_embeddings_faiss")
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local(persist_directory)
        else:
            persist_directory = os.path.join(self.config["retriever"]["vector_store"]["persist_directory"][0], "all_minilm_embeddings_chroma")
            db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
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
        if vectorstore == "Faiss":
            persist_directory = os.path.join(self.config["retriever"]["vector_store"]["persist_directory"][0], "bgem3_embeddings_faiss")
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local(persist_directory)
        else:
            persist_directory = os.path.join(self.config["retriever"]["vector_store"]["persist_directory"][0], "bgem3_embeddings_chroma")
            db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
            db.add_documents(docs)
        return db


    def openai_embeddings(self, chunks, vectorstore):
        docs = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk
                )
            )
        if not docs:
            return "Chunks list is empty"
        if self.openai_api_key:
            embedding_function = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            if vectorstore == "Faiss":
                persist_directory = os.path.join(self.config["retriever"]["vector_store"]["persist_directory"][0], "openai_embeddings_faiss")
                db = FAISS.from_documents(docs, embedding_function)
                db.save_local(persist_directory)
            else:
                persist_directory = os.path.join(self.config["retriever"]["vector_store"]["persist_directory"][0], "openai_embeddings_chroma")
                db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
                db.add_documents(docs)
            return db
        else:
            print("OpenAI API key is not set in the environment variable.")
            return None
    
    def generate_databases(self, chunks):
        embedding_methods = self.config["retriever"]["vector_store"]["embedding"]
        vectorstore_option = self.config["retriever"]["vector_store"]["database"]
        databases = []
        permutations = list(itertools.product(embedding_methods, vectorstore_option))
        for embedding_method,vectorstore_option in permutations:
            # print("creating db with embeddings:{} and vector store:{}".format(embedding_method,vectorstore_option))
            try:
                if embedding_method == "huggingface_instruct_embeddings":
                    db = self.huggingface_instruct_embeddings(chunks, vectorstore_option)
                elif embedding_method == "all_minilm_embeddings":
                    db = self.all_minilm_embeddings(chunks, vectorstore_option)
                elif embedding_method == "bgem3_embeddings":
                    db = self.bgem3_embeddings(chunks, vectorstore_option)
                elif embedding_method == "openai_embeddings":
                    db = self.openai_embeddings(chunks, vectorstore_option)
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
    config_file = "./config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", nargs='+', help="List of embedding options available are: huggingface_instruct_embeddings, all_minilm_embeddings, bgem3_embeddings, openai_embeddings")
    parser.add_argument("--vectorstore", type=str, help="Vector store option (chroma or faiss)")
    parser.add_argument("--chunks", nargs='+', help="List of text chunks")
    args = parser.parse_args()

    if args.embedding:
        config["retriever"]["vector_store"]["embedding"] = args.embedding

    if args.vectorstore:
        config["retriever"]["vector_store"]["database"] = args.vectorstore

    if args.chunks:
        config["retriever"]["vector_store"]["chunks"] = args.chunks

    obj = EmbeddingGenerator(config)
    dbs = obj.generate_databases(config["retriever"]["vector_store"]["chunks"])

    for db in dbs:
        print(db)

