"""
Setup configuration for the document scanner package
"""

from setuptools import setup, find_packages

# Read the content of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="doc-scanner",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A lightweight document scanner application using your camera",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/doc-scanner",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "doc-scanner=doc_scanner.scanner:main",
        ],
    },
)