from setuptools import setup, find_packages

setup(
    name='ragpy',
    version='0.1.0',
    author="BayesLabs",# packages=find_packages(),
    author_email="contact@bayeslabs.co",
    packages=
    [
        "ragpy",
        "ragpy.src.dataprocessing",
        "ragpy.src.embeddings_creation",
        "ragpy.src.retriever",
        "ragpy.src.generator"
    ],
    py_modules=["ragpy.src.dataprocessing.data_loader",
    "ragpy.src.embeddings_creation.embedding_generator",
    "ragpy.src.retriever.retrieval",
    "ragpy.src.retriever.retrieval_benchmarking",
    "ragpy.src.generator.generation_benchmarking"],
    # packages = find_packages()

    # Additional metadata
)
