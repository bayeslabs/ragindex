from ragpy.src.dataprocessing.data_loader import DataProcessor
from ragpy.src.embeddings_creation.embedding_generator import EmbeddingGenerator
from ragpy.src.retriever.retrieval import Reranking
from ragpy.src.retriever.retrieval_benchmarking import RetrievalBenchmarking
import argparse,yaml
from ragpy.src.generator.main_body import Generator_response
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import pathlib
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument("--config", type=str, default="./config.yaml",help="Path to the configuration file")
    parser.add_argument("--user_files", nargs='+', type=str, default=None, help="Path to the user-specified file to be processed")
    parser.add_argument("--chunk_size", type=int, default=400, help="Chunk size for splitting text")
    parser.add_argument("--text_overlap", type=int, default=50, help="Text overlap for splitting text")

    # embedding generation
    parser.add_argument("--embedding", nargs='+', help="List of embedding options available are: huggingface_instruct_embeddings, all_minilm_embeddings, bgem3_embeddings, openai_embeddings")
    parser.add_argument("--vectorstore", nargs='+', help="Vector store option (Chroma or Faiss)")

    # Retrieval
    parser.add_argument('--top_k',help='The number of top documents to be retrieved')
    parser.add_argument('--benchmark_data_path',help="Path to the benchmarking dataset in csv")
    parser.add_argument('--save_dir',help='Directory to save synthetic data')
    parser.add_argument('--num_questions',help="Number of questions to be generated in synthetic benchmark dataset", default=None)

    # Generation
    parser.add_argument('--query', default='What is RAG?', help='The query for which main logic is executed.')
    parser.add_argument('--context', nargs='+', help='The context for which the query is asked.')
    parser.add_argument('--model_type', type=str, help='The type of model to use. Can be "openai", "fireworks", or "hugging_face".')
    parser.add_argument('--chain_type', type=str, default='simple', help='The type of chain to use. Can be "simple", or "retrieval"')
    parser.add_argument('--domain', type=str, default='Healthcare', help='It can be anything.')
    parser.add_argument('--prompt_type', type=str, default='general', help='The type of prompt to use. Can be "general","custom" or "specific",')
    parser.add_argument('--temperature',nargs='+',default=[0.7,0.1],help="Temperature of the model. Default is 0.7.")
    parser.add_argument('--llm_repo_id',type=str,help="Hugging face repo id for llm")
    parser.add_argument('--db_path',type=str,help="Path of the db")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.user_files:
        config["data"]["corpus"] = args.user_files

    if args.chunk_size:
        config["retriever"]["chunk_size"] = args.chunk_size
    
    if args.text_overlap:
        config["retriever"]["text_overlap"] = args.text_overlap

    # for embedding generation
    if args.embedding:
        config["retriever"]["vector_store"]["embedding"] = args.embedding

    if args.vectorstore:
        config["retriever"]["vector_store"]["database"] = args.vectorstore

    # for retrieving top documents
    if args.top_k:
        config["retriever"]["top_k"] = int(args.top_k)
    if args.benchmark_data_path:
        config["data"]["benchmark_data"]=args.benchmark_data_path
    if args.save_dir:
        config["data"]["save_dir"]=args.save_dir
    # for synthetic data generation
    if args.num_questions:
        args.num_questions = int(args.num_questions)
    if args.model_type:
        config["generator"]["models"]["model_type"] = args.model_type
    if args.chain_type:
        config["generator"]["chain_type"] = args.chain_type   
    if args.domain:   
        config["generator"]["prompt_template"]["domain"] = args.domain   
    if args.prompt_type:  
        config["generator"]["prompt_template"]["prompt_type"] = args.prompt_type
    if args.temperature:
        config["generator"]["model_config"]["temperature"]=args.temperature
    if args.llm_repo_id:
        config["generator"]["models"]["hugging_face_model"]=args.llm_repo_id
    if args.embedding:
        config["retriever"]["vector_store"]["embedding"] = args.embedding
    if args.db_path:
        config["retriever"]["vector_store"]["persist_directory"] = args.db_path
      
    processor = DataProcessor(config)
    
    chunks = processor.process_data()
    config["retriever"]["vector_store"]["chunks"]=chunks

    
    embedding_generator=EmbeddingGenerator(config)
    dbs=embedding_generator.generate_databases(chunks)
    print("dbs in main",dbs)
    reranker=Reranking(config)
    if args.num_questions:
      retrieved_data_path=reranker.ret(chunks,config["retriever"]["top_k"],config,dict_db=dbs, num_questions=int(args.num_questions))
    else:
      retrieved_data_path=reranker.ret(chunks,config["retriever"]["top_k"],config,dict_db=dbs)
    print("path is:",retrieved_data_path)
    df,max_combo=RetrievalBenchmarking(datasets_dir_path=retrieved_data_path,config=config).validate_dataframe()
    print("max dataframe is at {}".format(retrieved_data_path+max_combo))
    db_path=retrieved_data_path+max_combo

    root_path=pathlib.Path(db_path).stem
    data=root_path.split("_")
    embedding_name="_".join(data[:-1])
    print("embedding name",embedding_name)
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

    df=pd.read_csv(db_path)
    objects={}
    print("model type in main",config["generator"]["models"]["model_type"])
    for query,context in zip(df["question"].to_list(),df["contexts"].to_list()):
      l={}
      abc=Generator_response(config=config,query=query,retriever=context)
      l[query]=abc
      l["result"]=l[query].main(query)
      objects[query]=l["result"]
      
    print(objects)
    ndf= pd.DataFrame.from_dict(objects, orient='index')
    generated_data_path="./ragpy/data/generated_data/generated_data.csv"
    ndf.to_csv(generated_data_path,index=False)
    print("results are saved to",generated_data_path)

    # if args.query:
    #   if args.embedding.lower()=="openai_embeddings":
    #     embedding_function = OpenAIEmbeddings()
    #   elif args.embedding.lower()=="huggingface_instruct_embeddings":
    #       embedding_function = HuggingFaceInstructEmbeddings()
    #   elif args.embedding.lower()=="all_minilm_embeddings":
    #       embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #   elif args.embedding.lower()=="bgem3_embeddings":
    #       embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    #   rag_object = Generator_response(config=config,query=args.query,db=args.db_path)
    
    # results = rag_object.main(args.query)




