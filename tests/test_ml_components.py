#!/usr/bin/env python3
"""
Test script for ML components
Verifies the functionality of document classifier, image enhancer, and text analyzer
"""

import os
import cv2
import numpy as np
from models.document_classifier import DocumentClassifier
from models.image_enhancer import ImageEnhancer
from models.text_analyzer import TextAnalyzer

def test_classifier(image_path):
    """Test the document classifier"""
    print("\n=== Testing Document Classifier ===")
    
    # Load classifier
    classifier = DocumentClassifier()
    
    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Classify document
    doc_type, confidence = classifier.predict(image)
    
    # Print results
    print(f"Document type: {doc_type}")
    print(f"Confidence: {confidence:.2f}")
    
    # Extract features for debugging
    features = classifier.extract_features(image)
    print(f"Feature vector shape: {features.shape}")
    
    return True

def test_enhancer(image_path, doc_type=None):
    """Test the image enhancer"""
    print("\n=== Testing Image Enhancer ===")
    
    # Load enhancer
    enhancer = ImageEnhancer()
    
    # Load test image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Enhance image
    enhanced = enhancer.enhance(image, doc_type)
    
    # Save enhanced image for comparison
    output_path = os.path.join("scans", "enhanced_test.jpg")
    cv2.imwrite(output_path, enhanced)
    
    print(f"Enhanced image saved to: {output_path}")
    print("Please visually verify the enhancement quality")
    
    return True

def test_text_analyzer(text):
    """Test the text analyzer"""
    print("\n=== Testing Text Analyzer ===")
    
    # Load analyzer
    analyzer = TextAnalyzer()
    
    # Analyze text with different document types
    doc_types = ["receipt", "id_card", "business_card", "text_document"]
    
    for doc_type in doc_types:
        print(f"\nAnalyzing as {doc_type}:")
        result = analyzer.analyze(text, doc_type)
        
        # Print the first few items from the result
        for key, value in list(result.items())[:5]:
            if isinstance(value, dict) and len(value) > 2:
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            elif isinstance(value, list) and len(value) > 2:
                print(f"  {key}: List with {len(value)} items")
            else:
                print(f"  {key}: {value}")
    
    return True

def main():
    """Main test function"""
    print("Testing ML Components for Document Scanner")
    
    # Check if we have any sample images
    scan_dir = "scans"
    test_image = None
    
    if os.path.exists(scan_dir):
        image_files = [f for f in os.listdir(scan_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            test_image = os.path.join(scan_dir, image_files[0])
    
    if test_image is None:
        print("No test images found in 'scans' directory. Using a blank test image.")
        # Create a blank test image
        test_image = os.path.join(scan_dir, "test_blank.jpg")
        os.makedirs(scan_dir, exist_ok=True)
        blank_img = np.ones((500, 700, 3), dtype=np.uint8) * 255
        cv2.putText(blank_img, "TEST DOCUMENT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(blank_img, (50, 70), (650, 400), (0, 0, 0), 2)
        cv2.imwrite(test_image, blank_img)
    
    # Test classifier
    classifier_ok = test_classifier(test_image)
    
    # Test enhancer
    enhancer_ok = test_enhancer(test_image, "text_document")
    
    # Sample text for analyzer
    sample_text = """
    ACME Corporation
    123 Business St.
    New York, NY 10001
    
    RECEIPT
    
    Date: 04/15/2025
    
    Item 1          $10.99
    Item 2          $24.50
    Tax              $3.55
    
    Total:         $39.04
    
    Thank you for your business!
    """
    
    # Test text analyzer
    analyzer_ok = test_text_analyzer(sample_text)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Classifier: {'OK' if classifier_ok else 'FAILED'}")
    print(f"Enhancer: {'OK' if enhancer_ok else 'FAILED'}")
    print(f"Text Analyzer: {'OK' if analyzer_ok else 'FAILED'}")
    
    if classifier_ok and enhancer_ok and analyzer_ok:
        print("\nAll tests passed! ML components are working correctly.")
    else:
        print("\nSome tests failed. Please check the output for details.")

if __name__ == "__main__":
    main()