from ragpy.src.dataprocessing.data_loader import DataProcessor
from ragpy.src.embeddings_creation.embedding_generator import EmbeddingGenerator
from ragpy.src.retriever.retrieval import Reranking
from ragpy.src.retriever.retrieval_benchmarking import RetrievalBenchmarking
import argparse,yaml


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

    if args.num_questions:
        args.num_questions = int(args.num_questions)

    processor = DataProcessor(config)
    
    chunks = processor.process_data()
    config["retriever"]["vector_store"]["chunks"]=chunks

    
    embedding_generator=EmbeddingGenerator(config)
    dbs=embedding_generator.generate_databases(chunks)
    reranker=Reranking(config)
    retrieved_data_path=reranker.ret(chunks,config["retriever"]["top_k"],config,dict_db=dbs, num_questions=args.num_questions)
    
    df,max_combo=retrieval_benchmarker=RetrievalBenchmarking(datasets_dir_path=retrieved_data_path,config=config).validate_dataframe()
    print("max dataframe is at {}".format(retrieved_data_path+max_combo))
