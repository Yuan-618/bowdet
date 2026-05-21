from setuptools import setup, find_packages

setup(
    name="bowdet",
    version="0.2.1",
    description="Audio-based bow-change and note-boundary detection for string instruments",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Haotian Yuan",
    author_email="yuanhaotianv@gmail.com",
    url="https://github.com/Yuan-618/bowdet",
    packages=find_packages(),
    package_data={"bowdet": ["assets/*"]},
    install_requires=[
        "numpy",
        "librosa",
        "tensorflow>=2.11",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)