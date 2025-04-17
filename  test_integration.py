#!/usr/bin/env python3
"""
Integration test for the full scanner application
Uses a pre-captured image to test all scanner components
"""

import os
import cv2
import numpy as np
import time
import argparse
import json
from pathlib import Path

# Import models
from models.document_classifier import DocumentClassifier
from models.image_enhancer import ImageEnhancer
from models.text_analyzer import TextAnalyzer

# Try to import OCR
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not found. OCR functionality will be limited.")

def test_full_scanner(image_path, output_dir="demo_output"):
    """
    Test the full scanner workflow with a pre-captured image
    
    Args:
        image_path: Path to test image
        output_dir: Directory to save output files
    """
    print(f"Testing full scanner with image: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Create timestamp for output files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Initialize ML components
    classifier = DocumentClassifier()
    enhancer = ImageEnhancer()
    analyzer = TextAnalyzer()
    
    # Step 1: Document classification
    print("\nStep 1: Document Classification")
    start_time = time.time()
    doc_type, confidence = classifier.predict(image)
    classification_time = time.time() - start_time
    
    print(f"Document type: {doc_type}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Classification time: {classification_time:.3f} seconds")
    
    # Save original image
    original_path = os.path.join(output_dir, f"demo_document_{timestamp}.jpg")
    cv2.imwrite(original_path, image)
    print(f"Original image saved: {original_path}")
    
    # Step 2: Image enhancement
    print("\nStep 2: Image Enhancement")
    start_time = time.time()
    enhanced_image = enhancer.enhance(image, doc_type)
    enhancement_time = time.time() - start_time
    
    print(f"Enhancement method: {doc_type}")
    print(f"Enhancement time: {enhancement_time:.3f} seconds")
    
    # Save enhanced image
    enhanced_path = os.path.join(output_dir, f"demo_enhanced_{timestamp}.jpg")
    cv2.imwrite(enhanced_path, enhanced_image)
    print(f"Enhanced image saved: {enhanced_path}")
    
    # Create comparison image
    h, w = image.shape[:2]
    target_h = min(400, h)
    scale = target_h / h
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    enh_resized = cv2.resize(enhanced_image, None, fx=scale, fy=scale)
    comparison = np.hstack([resized, enh_resized])
    comparison_path = os.path.join(output_dir, f"demo_comparison_{timestamp}.jpg")
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison image saved: {comparison_path}")
    
    # Step 3: OCR
    print("\nStep 3: OCR Text Extraction")
    if OCR_AVAILABLE:
        start_time = time.time()
        
        # Use grayscale for OCR
        if len(enhanced_image.shape) == 3:
            gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced_image
        
        # Use binary for better OCR if it's a text document
        if doc_type in ["text_document", "form"]:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_image = binary
        else:
            ocr_image = gray
        
        # Run OCR
        text = pytesseract.image_to_string(ocr_image)
        ocr_time = time.time() - start_time
        
        print(f"OCR time: {ocr_time:.3f} seconds")
        
        # Save extracted text
        text_path = os.path.join(output_dir, f"demo_text_{timestamp}.txt")
        with open(text_path, "w") as f:
            f.write(text)
        print(f"Text saved: {text_path}")
    else:
        print("OCR unavailable - skipping text extraction")
        text = None
    
    # Step 4: Text Analysis
    print("\nStep 4: Text Analysis")
    if text:
        start_time = time.time()
        analysis = analyzer.analyze(text, doc_type)
        analysis_time = time.time() - start_time
        
        print(f"Analysis time: {analysis_time:.3f} seconds")
        print("Key findings:")
        
        # Display key findings based on document type
        if doc_type == "receipt":
            print(f"  Total: {analysis.get('total', 'N/A')}")
            print(f"  Date: {analysis.get('date', 'N/A')}")
            print(f"  Merchant: {analysis.get('merchant', 'N/A')}")
            if 'items' in analysis and analysis['items']:
                print(f"  Items: {len(analysis['items'])}")
        elif doc_type == "id_card":
            print(f"  Name: {analysis.get('name', 'N/A')}")
            print(f"  ID Number: {analysis.get('id_number', 'N/A')}")
            print(f"  Date of Birth: {analysis.get('date_of_birth', 'N/A')}")
        elif doc_type == "business_card":
            print(f"  Name: {analysis.get('name', 'N/A')}")
            print(f"  Company: {analysis.get('company', 'N/A')}")
            print(f"  Email: {analysis.get('email', 'N/A')}")
            print(f"  Phone: {analysis.get('phone', 'N/A')}")
        else:
            print(f"  Word count: {analysis.get('word_count', 'N/A')}")
            print(f"  Paragraph count: {analysis.get('paragraph_count', 'N/A')}")
            print(f"  Keywords: {', '.join(analysis.get('keywords', []))}")
        
        # Save analysis as JSON
        analysis_path = os.path.join(output_dir, f"demo_analysis_{timestamp}.json")
        
        # Ensure JSON-serializable
        analysis_serializable = {}
        for k, v in analysis.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                analysis_serializable[k] = v
            elif isinstance(v, list):
                if all(isinstance(i, (str, int, float, bool, type(None))) for i in v):
                    analysis_serializable[k] = v
                else:
                    analysis_serializable[k] = [str(i) for i in v]
            elif isinstance(v, dict):
                analysis_serializable[k] = {str(dk): str(dv) for dk, dv in v.items()}
            else:
                analysis_serializable[k] = str(v)
        
        with open(analysis_path, "w") as f:
            json.dump(analysis_serializable, f, indent=2)
        print(f"Analysis saved: {analysis_path}")
    else:
        print("No text available for analysis")
    
    print("\nIntegration test completed successfully!")
    print(f"All output files saved to: {output_dir}")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the full scanner application")
    parser.add_argument("--image", "-i", type=str, required=True,
                        help="Path to test image")
    parser.add_argument("--output", "-o", type=str, default="demo_output",
                        help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    test_full_scanner(args.image, args.output)

if __name__ == "__main__":
    main()