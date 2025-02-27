# RAGINDEX

# Introduction
Ragindex is a python framework which streamlines the process of developing efficient and effective Retrieval Augmented Generation pipelines. The core idea behind this framework is to provide complete freedom to the user during the selection of components for the RAG pipeline, followed by a series of benchmarking at every crucial stage to provide the best combination of components, backed with quantitative metrics. Ragindex is currently a work in progress. It supports a variety of popular libraries and LLMs with support for other frameworks on the way. Overall, Ragindex enhances the process of developing accurate RAG pipelines while also being time efficient.  

## Prerequisites
Python 3.10

## Installation

1. Clone the repository:  
   ```bash
   ! git clone https://github.com/bayeslabs/ragindex.git
   cd ./Ragindex
   ```

2. Install the required packages:
   ```bash
   ! pip install ragindex
   ```

## Usage

Run the script main.py with  command-line argument
```!python main.py -h```
 By executing the RAG pipeline. Here are the available options that you can customise with the parsers
```
--config: Path to the configuration file (default: ./config.yaml)
--user_files: Path(s) to user-specified file(s) to be processed
--chunk_size: Chunk size for splitting text (default: 400)
--text_overlap: Text overlap for splitting text (default: 50)
--embedding: List of embedding options (e.g., huggingface_instruct_embeddings, all_minilm_embeddings, etc.)
--vectorstore: Vector store option (Chroma or Faiss)
--persist_dir: Path to the vector store persistent directory
--top_k: Number of top documents to be retrieved
--benchmark_data_path: Path to the benchmarking dataset in CSV with query context and ground_truth
--save_dir: Directory to save all the results
--num_questions: Number of questions to be generated in synthetic benchmarking dataset
--query: The query for which the main logic is executed
--context_given: Whether or not the context is given
--model_type: The type of model to use (openai or hugging_face)
--chain_type: The type of chain to use (simple or retrieval)
--domain: The domain for which main logic needs to be executed
--prompt_type: The type of prompt to use (general, custom, or specific)
--temperature: Temperature of the model (default: 0.7)
--llm_repo_id: Hugging face repo ID for language modelling
--db_path: Path of the database
```
 
## Example
```bash
python main.py --config ./config.yaml --user_files /path/to/files --chunk_size 400 --top_k 5
```

## Executing Ragindex

Ensure you have the necessary API keys set up:
- `OPENAI_API_KEY`: Your OpenAI API key
- `HUGGINGFACEHUB_API_TOKEN`: Your Hugging Face Hub API token

Run Ragindex with the desired configuration:
```bash
python main.py --config path/to/config.yaml
```

## Configuration
Customize the behavior of Ragindex using the `config.yaml` file. Refer to the configuration file for detailed options and descriptions.
