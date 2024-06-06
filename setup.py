from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='ragindex',
    version='0.1.0',
    author="BayesLabs",
    author_email="contact@bayeslabs.co",
    packages=find_packages(),
    py_modules=["ragindex.src.dataprocessing.data_loader",
    "ragindex.src.embeddings_creation.embedding_generator",
    "ragindex.src.retriever.retrieval",
    "ragindex.src.retriever.retrieval_benchmarking",
    "ragindex.src.generator.generation_benchmarking"],
    install_requires = required_packages
)
