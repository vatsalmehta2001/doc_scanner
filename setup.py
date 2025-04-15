#!/usr/bin/env python3
"""
ML-Enhanced Document Scanner setup script
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ml-doc-scanner",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning-powered document scanner with OCR and analysis capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-document-scanner",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "doc-scanner=scripts.advanced_scanner:main",
            "simple-scanner=scripts.simple_scanner:main",
            "doc-scanner-demo=scripts.quick_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.md'],
    },
)