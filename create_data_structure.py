#!/usr/bin/env python3
"""
Create training data directory structure
"""

import os
import shutil
from models.document_classifier import DOCUMENT_TYPES

def create_data_structure():
    """Create the directory structure for training data"""
    # Create main training data directory
    train_dir = "train_data"
    os.makedirs(train_dir, exist_ok=True)
    
    # Create subdirectories for each document type
    for doc_type in DOCUMENT_TYPES:
        doc_dir = os.path.join(train_dir, doc_type)
        os.makedirs(doc_dir, exist_ok=True)
        print(f"Created directory: {doc_dir}")
    
    print(f"\nTraining data structure created in '{train_dir}'")
    print("Place your training images in the appropriate subdirectories.")

if __name__ == "__main__":
    create_data_structure()