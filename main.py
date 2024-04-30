from ragpy.src.dataprocessing.data_loader import DataProcessor
import argparse,yaml
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument("--config", type=str, default="./config.yaml",help="Path to the configuration file")
    parser.add_argument("--user_files", nargs='+', type=str, default=None, help="Path to the user-specified file to be processed")
    parser.add_argument("--chunk_size", type=int, default=400, help="Chunk size for splitting text")
    parser.add_argument("--text_overlap", type=int, default=50, help="Text overlap for splitting text")
    args = parser.parse_args()
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

    processor = DataProcessor(config)
    
    chunks = processor.process_data()
