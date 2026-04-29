from setuptools import find_packages, setup


setup(
    name="seq2seq-recipe-generation-nlp",
    version="0.1.0",
    description="Portfolio-safe Seq2Seq recipe generation project using LSTM encoder-decoder models with optional attention.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "nltk>=3.8.1",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "torch>=2.1.0",
    ],
    extras_require={"dev": ["matplotlib>=3.8.0", "pytest>=8.0.0"]},
    python_requires=">=3.10",
)
