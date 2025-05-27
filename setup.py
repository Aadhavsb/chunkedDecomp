from setuptools import setup, find_packages

setup(
    name="chunked-decomp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "accelerate>=0.20.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    ],
    author="Your Name",
    description="ChunkedDecomp for efficient KV cache compression in transformers",
    python_requires=">=3.8",
)
