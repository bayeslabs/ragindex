from models_module import models_mod as mm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from prompt import CustomPromptTemplate
import pandas as pd
import yaml

# Open the YAML file
with open('config.yaml', 'r') as file:
    data = yaml.safe_load(file)


class RAG():
    def __init__(self,db=None,retriever=None,query=None,
                max_tokens = None,temperature= None):
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
        self.max_tokesn = max_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature

    def read_df(self,path):
        df = pd.read_csv(path)
        return df 
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


    def main(self,query):
        """
        Executes the main logic of the function using CustomPromptTemplate, retriever, datatype, objects, and llm.
        If the datatype is "string_datatype", formats the full prompt with context and question, then invokes llm.
        Otherwise, constructs a rag_chain pipeline and invokes it with the query.
        
        Parameters:
            query: The query for which main logic is executed.
        
        Returns:
            The result of executing the main logic.
        """
        domain = data['generator']['prompt_template']['domain']
        prompt_type = data['generator']['prompt_template']['prompt_type']
        prompt = CustomPromptTemplate(domain)
        prompt = prompt.main(prompt_type)
        retriever,datatype = self.retriever_fun()
        objects = mm()
        try:
            openai_model = objects.main("openai")
            fireworks_model = objects.main("opne_source")
            hugging_model =  objects.main("hugging_face")
            if datatype=="string_datatype":   
                full_prompt = prompt.format(context=retriever,question=query)
                openai_result = openai_model.invoke(full_prompt)
                fireworks_result = fireworks_model(full_prompt)
                hugging_face_result = hugging_model.invoke(full_prompt)
                return openai_result, fireworks_result, hugging_face_result
            else:
                openai_result = self.chains(retriever,prompt,openai_model).invoke(query)
                fireworks_result =self.chains(retriever,prompt,openai_model).invoke(query)
                hugging_face_result =self.chains(retriever,prompt,openai_model).invoke(query)
            return openai_result, fireworks_result, hugging_face_result
        except Exception as e:
            print(e)
            return openai_result, fireworks_result, "None"
if __name__=="__main__":
    query = "What are the considerations for treating elderly patients with SCLC?"
    context = "[Document(page_content='viewed in th is light, that is, there must be a toxicity-to-beneﬁting the studies to be stopped early (250, 251).ratio.The optimal management of the elderly with SCLC is anA Phase II study of irinotecan plus cisplatin yielded a CR ofimportant issue as 40% of those who present with the disease29% and an overall response rate of 86% with a median survivalare over 70 years old. Studies that investigated this area suggestof 13.2 months in patients with extensive disease SCLC (269).that a reasonably high initial dose of chemotherapy is importantThis has led to an RCT of irinotecan and cisplatin versus etopo-and that the elderly tolerate radiotherapy well (274–276). In-side and cisplatin in patients with extensive disease SCLC. Thedeed, elderly patients with good performance status and normalstudy was halted early because of a signiﬁcant survival advantageorgan function do as well with optimal chemotherapy dosesfor the patients randomized to irinotecan plus cisplatin (medianas their younger', metadata={'page': 103, 'source': '/content/drive/MyDrive/RAG BASED/MergedFiles1.pdf'})]"
    rag_object = RAG(retriever=context)
    # df = rag_object.read_df()
    # df.apply(main_body, axis=1)
    result1,result2,result3 = rag_object.main(query)
    print("---------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------")
    print("Results from OpenAI model:")
    print(result1)
    print("---------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------")
    print("Results from fireworks api model:")
    print(result2)
    print("---------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------")
    print("Result from hugging face model:")
    print(result3)
    #  print("--------------------------------------------------------------------")
    # print(result1,result2,result3,sep="\n")



         

