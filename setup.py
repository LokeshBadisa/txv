from setuptools import setup, find_packages
from pathlib import Path    

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:    
    requirements = f.readlines()    

setup(
    name = "txv",
    version = "0.0.1",
    author = "Lokesh Badisa",
    author_email = "lokeshbadisa657@gmail.com",
    description = "A Vision Transformer explainability package",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/LokeshBadisa/txv",
    project_urls = {
        "Bug Tracker": "https://github.com/LokeshBadisa/txv/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = find_packages(),
    python_requires = ">=3.8",
    install_requires=requirements    
)