
from langchain_core.prompts import PromptTemplate

class CustomPromptTemplate:
    def __init__(self,domain):
        """
        Initializes a new instance of the CustomPromptTemplate class.

        Args:
            domain (str): The domain for which the prompt is created.

        Returns:
            None
        """
        self.domain = domain
    def specific_prompt(self):
        """
        Generates a prompt for a specific question related to a domain.

        Returns:
            str: The generated prompt.
        """
        
        prompt = f"""You are a friendly and knowledgeable expert in {self.domain}. Your goal is to provide helpful and informative answers to questions using the provided reference passage when relevant. However, you should adjust your responses based on the following guidelines:
        Always respond in complete, well-structured sentences with proper grammar and punctuation.
        Aim for a conversational and approachable tone, avoiding overly technical jargon or complicated explanations.
        If the reference passage is relevant to the question, use information from it to craft a comprehensive answer that includes necessary background details and context.
        If the passage is not relevant or does not contain enough information to answer the question satisfactorily, feel free to use your own knowledge and expertise to provide a thorough response.
        Break down complex concepts or ideas into simpler terms that a non-technical audience can easily understand.
        Be proactive in identifying and addressing potential follow-up questions or areas where additional explanation may be needed.
        Context: {{context}}

        Question: {{question}}

        Helpful Answer:"""
        return prompt
    
    def general_prompt(self):
        prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """
        return prompt
    
    def custom_prompt(self):
        """
        Generates a prompt for a specific question related to a domain.
        """
        with open('custom_prompt.txt', 'r') as file:
            generation_template = file.read()
        return generation_template

    
    def main(self,prompt_type=None):
        """
        The main function that generates a prompt based on the given prompt type.

        Parameters:
            prompt_type (str): The type of prompt to generate. Can be "specific", "generator", or None.
                If None, the function will generate a general prompt.

        Returns:
            custom_rag_prompt (PromptTemplate): The generated prompt.
        """
        if prompt_type == "specific":
            prompt = self.specific_prompt()
        elif prompt_type == "custom":
            prompt = self.custom_prompt()
            print(prompt)
        else:
            prompt = self.general_prompt()
        custom_rag_prompt = PromptTemplate.from_template(prompt)
        return custom_rag_prompt
    
if __name__=="__main__":
    prompt = CustomPromptTemplate("healthcare")
    prompt = prompt.main("generator")
    # print(prompt)
