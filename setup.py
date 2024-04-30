from setuptools import setup, find_packages

setup(
    name='ragpy',
    version='0.1.0',
    author="BayesLabs",# packages=find_packages(),
    author_email="contact@bayeslabs.co",
    packages=
    [
        "ragpy",
        "ragpy.src.Retriever",
        "ragpy.src.Generation",
        "ragpy.src.DataPreprocessing"
    ],
    py_modules=["ragpy.src.DataPreprocessing.data_loader","ragpy.src.embedding_creation.embedding_generator","ragpy.src.Retriever.retrieval_benchmarking","ragpy.src.Retriever.retrieval","ragpy.src.Generation.generation_benchmarking","ragpy.src.Generation.main_body","ragpy.src.Generation.models_module","ragpy.src.Generation.prompt"],
    # packages = find_packages()

    # Additional metadata
)

