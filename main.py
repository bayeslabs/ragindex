import os, ast, shutil
from ragpy.src.dataprocessing.data_loader import DataProcessor
from ragpy.src.embeddings_creation.embedding_generator import EmbeddingGenerator
from ragpy.src.retriever.retrieval import Reranking
from ragpy.src.retriever.retrieval_benchmarking import RetrievalBenchmarking
import argparse,yaml
from ragpy.src.generator.main_body import Generator_response
from ragpy.src.generator.generation_benchmarking import Generation_Benchmarking
import pandas as pd
import tqdm,json

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ragpy")
    parser.add_argument("--config", type=str, default="./config.yaml",help="Path to the configuration file")
    parser.add_argument("--user_files", nargs='+', type=str, default=None, help="Path to the user-specified file to be processed")
    parser.add_argument("--chunk_size", type=int, default=400, help="Chunk size for splitting text")
    parser.add_argument("--text_overlap", type=int, default=50, help="Text overlap for splitting text")

    # embedding generation
    parser.add_argument("--embedding", nargs='+', help="List of embedding options ,the available options are: huggingface_instruct_embeddings, all_minilm_embeddings, bgem3_embeddings, openai_embeddings")
    parser.add_argument("--vectorstore", nargs='+', help="Vector store option (Chroma or Faiss)")
    parser.add_argument("--persist_dir",type=str,help="path to the vector store persistent directory")


    # Retrieval
    parser.add_argument('--reranker_methods', nargs='+', choices=['cross_encoder', 'flashrank'], default='cross_encoder',
                        help='The reranker method to use')
    parser.add_argument('--top_k',help='The number of top documents to be retrieved')
    parser.add_argument('--benchmark_data_path',help="Path to the benchmarking dataset in csv")
    parser.add_argument('--save_dir',help='Directory to save all the results like synthetic data,generated data and the predicted responses')
    parser.add_argument('--num_questions',help="Number of questions to be generated in synthetic benchmark dataset", default=None)

    # Generation
    parser.add_argument('--query', default='What is RAG?', help='The query for which main logic is executed.')
    parser.add_argument('--context_given', help='The context for which the query is asked.')
    parser.add_argument('--model_type', type=str, help='The type of model to use. Can be "openai", or "hugging_face".')
    parser.add_argument('--chain_type', type=str, default='simple', help='The type of chain to use. Can be "simple", or "retrieval"')
    parser.add_argument('--domain', type=str, default='Healthcare', help='It can be anything.')
    parser.add_argument('--prompt_type', type=str, default='general', help='The type of prompt to use. Can be "general","custom" or "specific",')
    parser.add_argument('--temperature',nargs='+',default=[0.1],help="Temperature of the model. Default is 0.7.")
    parser.add_argument('--llm_repo_id',type=str,help="Hugging face repo id for llm")
    parser.add_argument('--db_path',nargs='+',help="Path of the db")

    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.user_files:
        config["data"]["corpus"] = args.user_files
    
    if args.user_files:
        config["data"]["corpus"] = args.user_files

    if args.chunk_size:
        config["retriever"]["chunk_size"] = args.chunk_size
    
    if args.text_overlap:
        config["retriever"]["text_overlap"] = args.text_overlap
    
    if args.context_given:
        config["generator"]["context_given"] = args.context_given

    # for embedding generation
    if args.embedding:
        config["retriever"]["vector_store"]["embedding"] = args.embedding

    if args.vectorstore:
        config["retriever"]["vector_store"]["database"] = args.vectorstore

    if args.persist_dir:
        config["retriever"]["vector_store"]["persist_directory"][0]=args.persist_dir

    # for retrieving top documents
    if args.reranker_methods:
        config["retriever"]["rerankers"] = args.reranker_methods
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


    #Resetting directories
    folders = [config["retriever"]["vector_store"]["persist_directory"][0], 
               config["data"]["save_dir"] + "/retrieved_data/",
               config["data"]["save_dir"] + "/generated_data/"]

    for folder in folders:
        try:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        except FileNotFoundError:
            continue

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
    db_csv_path=retrieved_data_path+max_combo
    
    df=pd.read_csv(db_csv_path)
    final_response={}
    
    if config["generator"]["context_given"]=="no":
        for query in  tqdm.tqdm(df["question"].to_list(), desc="Processing queries with vector db and chains"):
            temp_result={}
            vdb_path =config["retriever"]["vector_store"]["persist_directory"][0]+config["retriever"]["vector_store"]["embedding"][0]+"_"+config["retriever"]["vector_store"]["database"][0]
            generator_object=Generator_response(config=config,query=query,db_path = vdb_path)
            temp_result[query]=generator_object
            temp_result["result"]=generator_object.main(query)
            final_response[query]=temp_result["result"]
    else:
        for query,context in tqdm.tqdm(zip(df["question"].to_list(), df["contexts"].to_list()), desc="Processing queries with dataframe"):
            temp_result={}
            generator_object=Generator_response(config=config,query=query,retriever = context)
            temp_result[query]=generator_object
            temp_result["result"]=generator_object.main(query)
            final_response[query]=temp_result["result"]

    temp_generated_df= pd.DataFrame.from_dict(final_response, orient='index')
    temp_generated_df = temp_generated_df.reset_index().rename(columns={'index': 'question'})

    # creating a folder for all generated data

    generated_data_dir = config["data"]["save_dir"] + "/generated_data/"

    if not os.path.exists(generated_data_dir):
        os.makedirs(generated_data_dir)
        
    temp_generated_data_path=generated_data_dir+"temp_generated_data.csv"
    temp_generated_df.to_csv(temp_generated_data_path,index=False)

    final_generated_df = pd.merge(df,temp_generated_df, on='question')
    final_generated_data=generated_data_dir+"final_generated_data.csv"
    final_generated_df.to_csv(final_generated_data,index=False)

    col_list = final_generated_df.columns
    if "Unnamed: 2" in col_list:
        final_generated_df.drop("Unnamed: 2",axis=1,inplace=True)

    final_generated_df['contexts'] = final_generated_df['contexts'].apply(ast.literal_eval)

    gen_bench= Generation_Benchmarking(testset_df=final_generated_df, config=config).run_benchmarks()
    output_txt_path=generated_data_dir+"Generation_benchmarking_results.txt"
    with open(output_txt_path,"w")as f:
        f.write(json.dumps(gen_bench))
    print("benchmarked_results are saved to",output_txt_path)
