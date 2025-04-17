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
    name="doc-scanner",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pytesseract>=0.3.10",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "pdf2image>=1.16.3",
        "Pillow>=10.0.0",
        "python-docx>=1.0.0",
        "reportlab>=4.0.0",
        "spacy>=3.7.0",
        "imutils>=0.5.4",
        "tqdm>=4.65.0",
        "python-magic>=0.4.27"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "doc-scanner=scripts.advanced_scanner:main",
            "doc-scanner-demo=scripts.demo:main",
            "simple-scanner=scripts.simple_scanner:main",
        ]
    },
    author="Vatsal Mehta",
    author_email="your.email@example.com",
    description="A powerful document scanning application with OCR capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="document scanner, ocr, computer vision, image processing",
    url="https://github.com/YOUR_USERNAME/doc_scanner",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
)