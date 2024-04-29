import os
import random
import chardet
import csv
import yaml
import nltk
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langchain.text_splitter import CharacterTextSplitter
import argparse

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

class DataProcessor:
    def __init__(self, config):
        # Initialize class variables
        self.config = config
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=self.config["retriever"]["chunk_size"],
            chunk_overlap=self.config["retriever"]["text_overlap"]
        )

    def process_data(self):
        # Process data from file paths in the configuration file
        corpus_file_paths = self.config["data"]["corpus"]

        chunks = []
        for file_path in corpus_file_paths:
            # Determine file type and read text
            file_type = os.path.splitext(file_path)[1][1:]
            if file_type.lower() not in ['pdf', 'csv', 'txt']:
                print(f"Warning: Unsupported file type: {file_path}")
                continue
            if file_type.lower() == 'pdf':
                with open(file_path, "rb") as f:
                    pdf = PdfReader(f)
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
            elif file_type.lower() == 'csv':
                with open(file_path, "r") as f:
                    reader = csv.reader(f)
                    text = "\n".join([",".join(row) for row in reader])
            else:
                with open(file_path, "rb") as f:
                    rawdata = f.read()
                result = chardet.detect(rawdata)
                encoding = result['encoding']
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
            # Preprocess text and split into chunks
            processed_text = self.process_text(text)
            chunks.extend(self.split_into_chunks(processed_text))

        return chunks

    def process_text(self, text):
        # Preprocess text by tokenizing, removing stopwords, and lemmatizing
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(tokens)

    def split_into_chunks(self, data):
        # Split text into chunks using the CharacterTextSplitter
        chunks = self.text_splitter.split_text(data)
        return chunks


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument("--config", type=str, default="/teamspace/studios/this_studio/ragKIT/config/sample_config.yaml",help="Path to the configuration file")
    parser.add_argument("--user_files", nargs='+', type=str, default=None, help="Path to the user-specified file to be processed")
    parser.add_argument("--chunk_size", type=int, default=400, help="Chunk size for splitting text")
    parser.add_argument("--text_overlap", type=int, default=50, help="Text overlap for splitting text")
    args = parser.parse_args()

    # Load configuration file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update configuration if user provides file path
    if args.user_files:
        config["data"]["corpus"] = args.user_files

    # Update configuration if user provides chunk size
    if args.chunk_size:
        config["retriever"]["chunk_size"] = args.chunk_size

    # Update configuration if user provides text overlap
    if args.text_overlap:
        config["retriever"]["text_overlap"] = args.text_overlap

    # Create instance of DataProcessor class
    processor = DataProcessor(config)

    # Call process_data method to get chunks
    chunks = processor.process_data()

    # Return the chunks
    #for i, chunk in enumerate(chunks):
        #print(f"\n{chunk}")
    if args.user_files:
        print("Updated data section of the configuration file:")
        print(yaml.dump(config["data"], default_flow_style=False))
