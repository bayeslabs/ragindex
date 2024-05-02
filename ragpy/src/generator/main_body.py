import sys
from ragpy.src.generator.models_module import models_mod as mm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from ragpy.src.generator.prompt import CustomPromptTemplate
import pandas as pd
from itertools import product
import yaml
import argparse


class Generator_response():
    def __init__(self,db=None,retriever=None,query=None,
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
        self.db = db
        self.query = query
        self.max_tokens = 249
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
           retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
           return retriever,"document_type"
        else:
            retriever = self.retriever
            # retriever = self.format_docs(retriever)
            return retriever, "string_datatype"
        
        
    def chains(self,retriever, prompt, models):
        if data["chain_type"]=="simple":
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
            print(f"Error retrieving retriever: {e}")
            return "both retriever and db are not provided."   
        objects = mm(config=self.data)
        models = {} # Dictionary to store model instances
        try:
            temp = self.data['generator']['model_config']['temperature']
            # temp = [0.1,0.7]
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
            return "None"
if __name__=="__main__":
    with open('./config.yaml', 'r') as file:
        data = yaml.safe_load(file)
        # Argument parser setup
    parser = argparse.ArgumentParser(description='Generator Response for RAG')
    parser.add_argument('--query', type=str, default='What are the considerations for treating elderly patients with SCLC?', help='The query for which main logic is executed.')
    parser.add_argument('--context', type=str, default='[Document(page_content=\'viewed in th is light, that is, there must be a toxicity-to-beneﬁting the studies to be stopped early (250, 251).ratio.The optimal management of the elderly with SCLC is anA Phase II study of irinotecan plus cisplatin yielded a CR ofimportant issue as 40% of those who present with the disease29% and an overall response rate of 86% with a median survivalare over 70 years old. Studies that investigated this area suggestof 13.2 months in patients with extensive disease SCLC (269).that a reasonably high initial dose of chemotherapy is importantThis has led to an RCT of irinotecan and cisplatin versus etopo-and that the elderly tolerate radiotherapy well (274–276). In-side and cisplatin in patients with extensive disease SCLC. Thedeed, elderly patients with good performance status and normalstudy was halted early because of a signiﬁcant survival advantageorgan function do as well with optimal chemotherapy dosesfor the patients randomized to irinotecan plus cisplatin (medianas their younger\', metadata={\'page\': 103, \'source\': \'/content/drive/MyDrive/RAG BASED/MergedFiles1.pdf\'})]', help='The context for the query.')
    parser.add_argument('--model_type', type=str, default='openai', help='The type of model to use. Can be "openai", "fireworks", or "hugging_face".')
    parser.add_argument('--chain_type', type=str, default='simple', help='The type of chain to use. Can be "simple", or "retrieval"')
    parser.add_argument('--domain', type=str, default='Healthcare', help='It can be anything.')
    parser.add_argument('--prompt_type', type=str, default='general', help='The type of prompt to use. Can be "general","custom" or "specific",')
    
    # Parse arguments
    args = parser.parse_args()

    # Check if provided arguments match expected values in the config file
    if data["generator"]["models"]:
        data["generator"]["models"]["model_type"] = args.model_type
    if data["generator"]["chain_type"]:
        data["generator"]["chain_type"] = args.chain_type   
    if data["generator"]["prompt_template"]["domain"]:   
        data["generator"]["prompt_template"]["domain"] = args.domain   
    if data["generator"]["prompt_template"]["prompt_type"]:  
        data["generator"]["prompt_template"]["prompt_type"] = args.prompt_type

    # Create an instance of the Generation_Benchmarking class and run the benchmarks
    rag_object = Generator_response(retriever= args.context,config=data)
    results = rag_object.main(args.query) 
   
    print("---------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------")
    print("Results from model:")
    print(results)
