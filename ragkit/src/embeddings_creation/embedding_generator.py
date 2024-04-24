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
#from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings 
import yaml 
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from itertools import product

class EmbeddingGenerator:
    def huggingface_instruct_embeddings(self, chunks, vectorstore):
        docs = [Document(page_content=chunk) for chunk in chunks]
        embedding_function = HuggingFaceInstructEmbeddings()
        if vectorstore == "faiss":
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local("huggingface_instruct_embeddings_faiss")
        else:
            db = Chroma(persist_directory='huggingface_instruct_embeddings_chroma', embedding_function=embedding_function)
            db.add_documents(docs) 
        return db

    def all_minilm_embeddings(self, chunks, vectorstore):
        docs = [Document(page_content=chunk) for chunk in chunks]
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        if vectorstore == "faiss":
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local("all_minilm_embeddings_faiss")
        else:
            db = Chroma(persist_directory='all_minilm_embeddings_chroma', embedding_function=embedding_function)
            db.add_documents(docs)
        return db

    def bgem3_embeddings(self, chunks, vectorstore):
        docs = [Document(page_content=chunk) for chunk in chunks]
        embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        if vectorstore == "faiss":
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local("bgem3_embeddings_faiss")
        else:
            db = Chroma(persist_directory="bgem3_embeddings_chroma", embedding_function=embedding_function)
            db.add_documents(docs)
        return db

    def openai_embeddings(self, chunks, vectorstore, api_key):
        docs = [Document(page_content=chunk) for chunk in chunks]
        embedding_function = OpenAIEmbeddings()
        if vectorstore == "faiss":
            db = FAISS.from_documents(docs, embedding_function)
            db.save_local("openai_embeddings_faiss") 
        else:
            db = Chroma(persist_directory='openai_embeddings_chroma', embedding_function=embedding_function)
            db.add_documents(docs)
        return db

def load_config(config_file):
    with open(config_file, 'r') as file: 
        config = yaml.safe_load(file)
    return config


def generate_databases(embedding_methods, vectorstore_options):
    databases = []
    embedding_generator = EmbeddingGenerator()

    combinations = product(embedding_methods, vectorstore_options)

    for embedding_method, vectorstore in combinations:
        try:
            if embedding_method == "huggingface_instruct_embeddings":
                db = embedding_generator.huggingface_instruct_embeddings(t, vectorstore)
            elif embedding_method == "all_minilm_embeddings":
                db = embedding_generator.all_minilm_embeddings(t, vectorstore)
            elif embedding_method == "bgem3_embeddings":
                db = embedding_generator.bgem3_embeddings(t, vectorstore)
            elif embedding_method == "openai_embeddings":
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if isinstance(os.environ.get("OPENAI_API_KEY"), str):
                    db = embedding_generator.openai_embeddings(t, vectorstore, openai_api_key)
                else:
                    raise ValueError("OpenAI API key is required for OpenAI embeddings.")
            else:
                raise ValueError("Invalid embedding method specified.")

            databases.append(db)
        except Exception as e:
            print(f"Error generating database for {embedding_method} and {vectorstore}: {e}")

    return databases
# def generate_databases(embedding_methods, vectorstore_options):
#     databases = []
#     embedding_generator = EmbeddingGenerator()

#     for embedding_method in embedding_methods:
#         for vectorstore in vectorstore_options:
#             try:
#                 if embedding_method == "huggingface_instruct_embeddings":
#                     db = embedding_generator.huggingface_instruct_embeddings(t, vectorstore)
#                 elif embedding_method == "all_minilm_embeddings":
#                     db = embedding_generator.all_minilm_embeddings(t, vectorstore)
#                 elif embedding_method == "bgem3_embeddings":
#                     db = embedding_generator.bgem3_embeddings(t, vectorstore)
#                 elif embedding_method == "openai_embeddings":
#                     openai_api_key = os.environ.get("OPENAI_API_KEY")
#                     if isinstance(os.environ.get("OPENAI_API_KEY"), str):
#                         db = embedding_generator.openai_embeddings(t, vectorstore, openai_api_key)
#                     else:
#                         raise ValueError("OpenAI API key is required for OpenAI embeddings.")
#                 else:
#                     raise ValueError("Invalid embedding method specified.")

#                 databases.append(db)
#             except Exception as e:
#                 print(f"Error generating database for {embedding_method} and {vectorstore}: {e}")

#     return databases


if __name__ == "__main__":
    config_file = '/home/roopesh_d/Documents/autorag/config.yaml'
    config = load_config(config_file)
    t = ['The sentence transformers library provides a wide range of pre-trained models that you can use to ',
         'generate sentence embeddings for different use cases. You can explore the available models and choo',
         'se the one that best fits your needs.']

    # Get the embeddings and vector store configurations from the config file
    embedding_methods = config['retriever']['vector_store']['embeddings']
    vectorstore_options = [option.lower() for option in config['retriever']['vector_store']['database']]

    databases = generate_databases(embedding_methods, vectorstore_options)
    for db in databases:
        print(db) 
