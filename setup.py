from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='ragpy',
    version='0.1.0',
    author="BayesLabs",
    author_email="contact@bayeslabs.co",
    packages=find_packages(),
    py_modules=["ragpy.src.dataprocessing.data_loader",
    "ragpy.src.embeddings_creation.embedding_generator",
    "ragpy.src.retriever.retrieval",
    "ragpy.src.retriever.retrieval_benchmarking",
    "ragpy.src.generator.generation_benchmarking"],
    install_requires = required_packages
)
