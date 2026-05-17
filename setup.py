from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bowdet",
    version="0.1.0",
    author="Haotian Yuan",
    description="Bow change detection for bowed string instruments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haotian-yuan/bowdet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "transformers>=4.30.0",
        "huggingface-hub>=0.16.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
)
