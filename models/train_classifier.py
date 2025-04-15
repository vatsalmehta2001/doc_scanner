#!/usr/bin/env python3
"""
Document Classifier Training Script
Trains the document classifier on a set of example images
"""

import os
import cv2
import numpy as np
import argparse
import joblib
from pathlib import Path
from models.document_classifier import DocumentClassifier

def load_training_data(data_dir, image_extensions=('.jpg', '.jpeg', '.png')):
    """
    Load training data from subdirectories, where each subdirectory name 
    corresponds to a document type.
    
    Args:
        data_dir: Path to training data directory
        image_extensions: Valid image file extensions
        
    Returns:
        tuple: (images, labels, label_names)
    """
    print(f"Loading training data from {data_dir}...")
    
    images = []
    labels = []
    label_names = []
    
    # Get all subdirectories (each representing a class)
    subdirs = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if not subdirs:
        print("No subdirectories found. Looking for images in the main directory...")
        subdirs = [""]  # Use empty string to indicate main directory
    
    print(f"Found subdirectories: {subdirs if subdirs[0] else 'None, using main directory'}")
    
    # Create a mapping from class names to integers
    class_to_idx = {subdirs[i]: i for i in range(len(subdirs))}
    
    # Track class distribution
    class_counts = {subdir: 0 for subdir in subdirs}
    
    # Load images from each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        
        # If we're looking at the main directory
        if not subdir:
            subdir_path = data_dir
            class_name = "text_document"  # Default class
        else:
            class_name = subdir
            
        label_names.append(class_name)
        
        # Get all image files in this directory
        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file)
                    
                    # Load and process image
                    try:
                        image = cv2.imread(file_path)
                        if image is None:
                            print(f"Warning: Could not load image {file_path}")
                            continue
                            
                        # Add image and label
                        images.append(image)
                        labels.append(class_name)
                        
                        # Update class count
                        class_counts[subdir] += 1
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    # Print statistics
    print("\nTraining data statistics:")
    total_images = len(images)
    print(f"Total images: {total_images}")
    
    for subdir in subdirs:
        class_name = subdir if subdir else "text_document"
        count = class_counts[subdir]
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    return images, labels, label_names

def apply_augmentations(images, labels, augment_factor=2):
    """
    Apply data augmentation to increase training set size and diversity
    
    Args:
        images: List of training images
        labels: List of corresponding labels
        augment_factor: How many augmented versions to create per image
        
    Returns:
        tuple: (augmented_images, augmented_labels)
    """
    print(f"\nApplying data augmentation (factor: {augment_factor})...")
    
    augmented_images = images.copy()
    augmented_labels = labels.copy()
    
    for i, (image, label) in enumerate(zip(images, labels)):
        # Skip small images
        if image.shape[0] < 100 or image.shape[1] < 100:
            continue
            
        for j in range(augment_factor):
            # Apply different augmentations based on iteration
            augmented = image.copy()
            
            # 1. Add blur (simulate blurry scans)
            if j % 2 == 0:
                blur_amount = np.random.randint(3, 7) * 2 + 1  # Odd numbers: 3, 5, 7, ...
                augmented = cv2.GaussianBlur(augmented, (blur_amount, blur_amount), 0)
            
            # 2. Add noise
            if j % 3 == 0:
                noise = np.random.normal(0, 15, augmented.shape).astype(np.uint8)
                augmented = cv2.add(augmented, noise)
            
            # 3. Adjust brightness/contrast
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-30, 30)    # Brightness
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
            
            # 4. Rotate slightly
            if j % 4 == 0:
                angle = np.random.uniform(-5, 5)
                h, w = augmented.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented = cv2.warpAffine(augmented, M, (w, h), 
                                          borderMode=cv2.BORDER_REPLICATE)
            
            # Add the augmented image and label
            augmented_images.append(augmented)
            augmented_labels.append(label)
    
    print(f"Original dataset size: {len(images)} images")
    print(f"Augmented dataset size: {len(augmented_images)} images")
    
    return augmented_images, augmented_labels

def train_and_evaluate(images, labels, label_names, test_split=0.2):
    """
    Train the document classifier and evaluate performance
    
    Args:
        images: List of training images
        labels: List of corresponding labels
        label_names: List of class names
        test_split: Fraction of data to use for testing
        
    Returns:
        DocumentClassifier: Trained classifier
    """
    print("\nTraining document classifier...")
    
    # Initialize classifier
    classifier = DocumentClassifier()
    
    # Set the class names
    classifier.classes = sorted(set(label_names))
    print(f"Class names: {classifier.classes}")
    
    # Split data into training and test sets
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_split)
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    test_images = [images[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"Training set size: {len(train_images)} images")
    print(f"Test set size: {len(test_images)} images")
    
    # Train the classifier
    try:
        print("Extracting features and training model...")
        accuracy = classifier.train(train_images, train_labels)
        print(f"Training accuracy: {accuracy:.2f}")
        
        # Evaluate on test set
        if test_images:
            print("\nEvaluating on test set...")
            correct = 0
            predictions = []
            
            for i, (image, true_label) in enumerate(zip(test_images, test_labels)):
                predicted_label, confidence = classifier.predict(image)
                predictions.append((true_label, predicted_label, confidence))
                
                if predicted_label == true_label:
                    correct += 1
                    
            test_accuracy = correct / len(test_images)
            print(f"Test accuracy: {test_accuracy:.2f}")
            
            # Print confusion matrix
            print("\nClass-wise performance:")
            class_performance = {c: {"correct": 0, "total": 0} for c in classifier.classes}
            
            for true_label, pred_label, _ in predictions:
                if true_label in class_performance:
                    class_performance[true_label]["total"] += 1
                    if true_label == pred_label:
                        class_performance[true_label]["correct"] += 1
            
            for cls, perf in class_performance.items():
                if perf["total"] > 0:
                    cls_accuracy = perf["correct"] / perf["total"]
                    print(f"  {cls}: {cls_accuracy:.2f} ({perf['correct']}/{perf['total']})")
    
    except Exception as e:
        print(f"Error during training: {e}")
        return None
    
    return classifier

def main():
    """Main function for training document classifier"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train document classifier")
    parser.add_argument("--data", "-d", type=str, default="scans",
                        help="Directory containing training images")
    parser.add_argument("--augment", "-a", type=int, default=2,
                        help="Augmentation factor (0 to disable)")
    parser.add_argument("--output", "-o", type=str, default="models/document_classifier.model",
                        help="Output model file path")
    
    args = parser.parse_args()
    
    # Load training data
    images, labels, label_names = load_training_data(args.data)
    
    if not images:
        print("No training images found. Exiting.")
        return
    
    # Apply augmentations if requested
    if args.augment > 0:
        images, labels = apply_augmentations(images, labels, args.augment)
    
    # Train and evaluate
    classifier = train_and_evaluate(images, labels, label_names)
    
    if classifier:
        # Save the trained model
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if classifier.save_model(output_path):
            print(f"\nModel saved to {output_path}")
        else:
            print("\nError: Failed to save model")

if __name__ == "__main__":
    main()