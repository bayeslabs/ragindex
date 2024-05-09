import sys
from models_module import models_mod as mm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from prompt import CustomPromptTemplate
import pandas as pd 
import pathlib   
from itertools import product 
import yaml
import argparse     
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma,FAISS
import os 
class Generator_response(): 
    def __init__(self,db_path=None,retriever=None,query=None,
                max_tokens = None,temperature= None,config = None):
        """
        Initializes a new instance of the class.
        Args:
            db (optional): The database object. Defaults to None.
            retriever (optional): The retriever object. Defaults to None.
            query (optional): The query object. Defaults to None.
            model_name (optional): The name of the model. Defaults to None.
        """

        self.retriever = retriever
        self.db_path = db_path
        self.db = None
        self.query = query
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.data = config
    
    
    
    def format_docs(self,docs):
        """
        Format the given list of documents into a single string.
        Args:
            docs (List[Document]): A list of Document objects.
        Returns:
            str: The formatted string containing the page content of all the documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
 
    def retriever_fun(self):
        """
        Retrieves the retriever object and its data type.
        Returns:
            retriever (object): The retriever object.
            datatype (str): The data type of the retriever object.
        """

        if self.retriever == None:
            if isinstance(self.db_path, list):
                self.db_path = self.db_path[0]
            root_path=pathlib.Path(self.db_path).stem
            data_list=root_path.split("_")
            embedding_name="_".join(data_list[:-1])
            embedding_model = self.embedding_fun(embedding_name=embedding_name)
            db_name = data_list[-1]
            if db_name.lower()=="chroma":
                vectordb = Chroma(persist_directory=self.db_path, embedding_function=embedding_model)
                self.db=vectordb
            else:
                vectordb = FAISS.load_local(self.db_path, embeddings=embedding_model)
                self.db=vectordb    
            retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
            return retriever,"document_type"
        else:
            retriever = self.retriever
            return retriever, "string_datatype"

    def embedding_fun(self,embedding_name=None):
        print(embedding_name)
        if embedding_name.lower()=="openai_embeddings":
            embedding_function = OpenAIEmbeddings()
        elif embedding_name.lower()=="huggingface_instruct_embeddings":
            embedding_function = HuggingFaceInstructEmbeddings()
        elif embedding_name.lower()=="all_minilm_embeddings":
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        elif embedding_name.lower()=="bgem3_embeddings":
            embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        else:
            raise ValueError("Invalid embedding name. Please choose from: openai_embeddings, huggingface_instruct_embeddings, all_minilm_embeddings, bgem3_embeddings")
        
        return embedding_function
    def chains(self,retriever, prompt, models):
        if self.data["generator"]["chain_type"]=="simple":
            rag_chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | models
                | StrOutputParser()
                
                )
        else:
            rag_chain = RetrievalQA.from_chain_type(
            llm=models,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt":prompt},
            return_source_documents=True)
        return rag_chain

    def main(self, query):
        domain = self.data['generator']['prompt_template']['domain']
        prompt_type = self.data['generator']['prompt_template']['prompt_type']
        prompt = CustomPromptTemplate(domain)
        prompt = prompt.main(prompt_type)
        try:
             retriever, datatype = self.retriever_fun()
        except Exception as e:
            print(e)
            sys.exit(1)  # Exit with a non-zero status code
            # return "both retriever and db are not provided please provide any one of them"

        objects = mm(config=self.data)
        models = {} # Dictionary to store model instances
        try:
            temp = self.data['generator']['model_config']['temperature']
            model_type = self.data['generator']['models']['model_type']
            if model_type == "openai":
                model_list = self.data['generator']['models']['open_ai_model']
                # model_list = ["gpt-3.5-turbo"]
              

                combinations = product(temp, model_list)
                for i in combinations:
                    model_key = f"openai_{i[1]}_{i[0]}" # Unique key for each model instance
                    models[model_key] = objects.main(model_type, model_name=i[1], temp=i[0])
            else:
                model_list = self.data['generator']['models']['hugging_face_model']
                combinations = product(temp, model_list)
                for i in combinations:
                    model_key = f"hugging_face_{i[1]}_{i[0]}" # Unique key for each model instance
                    models[model_key] = objects.main(model_type, model_name=i[1], temp=i[0])

            if datatype == "string_datatype":
                full_prompt = prompt.format(context=retriever, question=query)
                results = {}
                if model_type == "openai":
                    for model_key, model in models.items():
                        results[model_key] = model.invoke(full_prompt).content
                else:
                    for model_key, model in models.items():
                        results[model_key] = model.invoke(full_prompt)
                return results
            else:
                results = {}
                for model_key, model in models.items():
                    results[model_key] = self.chains(retriever, prompt, model).invoke(query)
                return results
        except Exception as e:
            print(e)
            sys.exit(1)  # Exit with a non-zero status code
           
if __name__=="__main__":
    path = "./config.yaml"
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
        # Argument parser setup
    parser = argparse.ArgumentParser(description='Generator Response for RAG')
    parser.add_argument('--query', default='What is hello?', help='The query for which main logic is executed.')
    parser.add_argument('--context', nargs='+', help='The context for which the query is asked.')
    parser.add_argument('--model_type', type=str, default='hugging_face', help='The type of model to use. Can be "openai", "fireworks", or "hugging_face".')
    parser.add_argument('--chain_type', type=str, default='simple', help='The type of chain to use. Can be "simple", or "retrieval"')
    parser.add_argument('--domain', type=str, default='Healthcare', help='It can be anything.')
    parser.add_argument('--prompt_type', type=str, default='general', help='The type of prompt to use. Can be "general","custom" or "specific",')
    parser.add_argument('--temperature',nargs='+',default=[0.7,0.1],help="Temperature of the model. Default is 0.7.")
    parser.add_argument('--repo_id',type=str,help="Hugging face repo id")
    parser.add_argument('--embeddings', nargs='+', help="List of embedding options available are: huggingface_instruct_embeddings, all_minilm_embeddings, bgem3_embeddings, openai_embeddings")
    parser.add_argument('--db_path',type=str,help="Path of the db")
                        # Parse arguments
    args = parser.parse_args()
    # Check if provided arguments match expected values in the config file
    if args.model_type:
        data["generator"]["models"]["model_type"] = args.model_type
    if args.chain_type:
        data["generator"]["chain_type"] = args.chain_type   
    if args.domain:   
        data["generator"]["prompt_template"]["domain"] = args.domain   
    if args.prompt_type:  
        data["generator"]["prompt_template"]["prompt_type"] = args.prompt_type
    if args.temperature:
        data["generator"]["model_config"]["temperature"]=args.temperature
    if args.repo_id:
        data["generator"]["models"]["hugging_face_model"]=args.repo_id
    if args.embeddings:
        data["retriever"]["vector_store"]["embeddings"] = args.embeddings
    if args.db_path:
        data["retriever"]["vector_store"]["persist_directory"] = args.db_path

    if args.context and args.query:
        rag_object = Generator_response(retriever=args.context,config=data,query=args.query,db=data["retriever"]["vector_store"]["persist_directory"])
    elif args.db_path:
        rag_object = Generator_response(config=data,query=args.query,db_path=args.db_path)
    else:
        raise ValueError("both retriever and db path are not provided")
    results = rag_object.main(args.query)
    print("Results from model:",results) 
