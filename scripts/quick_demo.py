#!/usr/bin/env python3
"""
Quick Demo Script for ML-Enhanced Document Scanner
Demonstrates the complete pipeline of document scanning, enhancement, and analysis
"""

import os
import cv2
import argparse
import time
from datetime import datetime
import numpy as np
import json

# Import project modules
from doc_scanner.scanner import EnhancedDocumentScanner
from doc_scanner.utils import display_apple_silicon_tips, ensure_directory
from doc_scanner.document import detect_document, enhance_document

# Import ML models
from models.document_classifier import DocumentClassifier
from models.image_enhancer import ImageEnhancer
from models.text_analyzer import TextAnalyzer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML-Enhanced Document Scanner Demo")
    parser.add_argument("--image", "-i", type=str, help="Path to input image (if not using camera)")
    parser.add_argument("--output", "-o", type=str, default="demo_output", help="Output directory")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index (usually 0 for built-in)")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode (no GUI)")
    parser.add_argument("--demo-all", action="store_true", help="Run all demo components")
    
    return parser.parse_args()

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")

def demo_document_classification(image_path, classifier=None):
    """Demo document classification functionality"""
    print_section("Document Classification Demo")
    
    # Initialize classifier if not provided
    if classifier is None:
        classifier = DocumentClassifier()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    print(f"Loaded image: {image_path} with shape {image.shape}")
    
    # Detect document in image
    start_time = time.time()
    document, corners = detect_document(image)
    detection_time = time.time() - start_time
    
    if document is None:
        print("No document detected in the image")
        return None, None
    
    print(f"Document detected in {detection_time:.3f} seconds")
    
    # Classify document
    start_time = time.time()
    doc_type, confidence = classifier.predict(document)
    classification_time = time.time() - start_time
    
    print(f"Document classified as: {doc_type} with {confidence:.2f} confidence")
    print(f"Classification completed in {classification_time:.3f} seconds")
    
    # Display document features used for classification
    features = classifier.extract_features(document).flatten()
    feature_names = [
        "Aspect Ratio", "Edge Density", "Contour Density", 
        "Texture", "Text Lines", "Brightness", "Contrast"
    ]
    
    print("\nDocument Features:")
    for name, value in zip(feature_names, features[:7]):
        print(f"  {name}: {value:.4f}")
    
    return document, doc_type

def demo_image_enhancement(document, doc_type):
    """Demo image enhancement functionality"""
    print_section("Image Enhancement Demo")
    
    if document is None:
        print("No document to enhance")
        return None
    
    # Initialize enhancer
    enhancer = ImageEnhancer()
    
    print(f"Enhancing document of type: {doc_type}")
    
    # Process with different enhancement modes
    start_time = time.time()
    
    # Apply enhancement based on document type
    enhanced = enhancer.enhance(document, doc_type)
    enhancement_time = time.time() - start_time
    
    print(f"Enhancement completed in {enhancement_time:.3f} seconds")
    
    # Also create different enhancement versions for demo
    enhanced_text = enhancer.enhance_text(document)
    enhanced_bw = enhancer.enhance_receipt(document)  # Higher contrast B&W
    
    # Create a combined visualization
    h, w = document.shape[:2]
    
    # Create a 2x2 grid of images
    grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    
    # Convert grayscale to BGR if needed
    if len(document.shape) == 2:
        document_disp = cv2.cvtColor(document, cv2.COLOR_GRAY2BGR)
    else:
        document_disp = document.copy()
        
    if len(enhanced.shape) == 2:
        enhanced_disp = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        enhanced_disp = enhanced.copy()
        
    if len(enhanced_text.shape) == 2:
        enhanced_text_disp = cv2.cvtColor(enhanced_text, cv2.COLOR_GRAY2BGR)
    else:
        enhanced_text_disp = enhanced_text.copy()
        
    if len(enhanced_bw.shape) == 2:
        enhanced_bw_disp = cv2.cvtColor(enhanced_bw, cv2.COLOR_GRAY2BGR)
    else:
        enhanced_bw_disp = enhanced_bw.copy()
    
    # Place images in grid
    grid[:h, :w] = document_disp
    grid[:h, w:] = enhanced_disp
    grid[h:, :w] = enhanced_text_disp
    grid[h:, w:] = enhanced_bw_disp
    
    # Add labels
    cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(grid, f"Enhanced ({doc_type})", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(grid, "Text Mode", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(grid, "High Contrast", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return grid, enhanced

def demo_text_extraction(enhanced_image, doc_type):
    """Demo text extraction and analysis functionality"""
    print_section("Text Extraction & Analysis Demo")
    
    if enhanced_image is None:
        print("No enhanced image for text extraction")
        return None
    
    # Initialize text extraction
    try:
        import pytesseract
        from PIL import Image
        
        # Convert OpenCV image to PIL format
        if len(enhanced_image.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(enhanced_image)
            
        # Extract text
        start_time = time.time()
        text = pytesseract.image_to_string(pil_img)
        extraction_time = time.time() - start_time
        
        print(f"Text extraction completed in {extraction_time:.3f} seconds")
        print("\nExtracted Text:")
        print("-" * 40)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 40)
        
        # Analyze text
        analyzer = TextAnalyzer()
        
        start_time = time.time()
        analysis_result = analyzer.analyze(text, doc_type)
        analysis_time = time.time() - start_time
        
        print(f"Text analysis completed in {analysis_time:.3f} seconds")
        print("\nText Analysis Results:")
        print("-" * 40)
        print(json.dumps(analysis_result, indent=2, default=str)[:1000] + "..." 
              if len(json.dumps(analysis_result, indent=2)) > 1000 
              else json.dumps(analysis_result, indent=2, default=str))
        
        return text, analysis_result
    
    except ImportError:
        print("Text extraction requires pytesseract. Install with: pip install pytesseract")
        print("And install Tesseract OCR engine from: https://github.com/tesseract-ocr/tesseract")
        return None, None
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return None, None

def demo_live_scanner(args):
    """Demo the live document scanner"""
    print_section("Live Document Scanner Demo")
    
    # Initialize scanner
    scanner = EnhancedDocumentScanner(
        output_dir=args.output,
        camera_id=args.camera,
        resolution=(1280, 720),
        batch_mode=False,
        enhancement_mode="auto"
    )
    
    print("Starting live document scanner...")
    print("Press 's' to save the current scan")
    print("Press 'e' to toggle enhancement preview mode")
    print("Press 'q' to quit")
    
    # Run scanner
    scanner.run()

def save_demo_results(output_dir, document, enhanced, grid, text, analysis_result):
    """Save demo results to output directory"""
    ensure_directory(output_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save images
    if document is not None:
        cv2.imwrite(os.path.join(output_dir, f"demo_document_{timestamp}.jpg"), document)
    
    if enhanced is not None:
        cv2.imwrite(os.path.join(output_dir, f"demo_enhanced_{timestamp}.jpg"), enhanced)
    
    if grid is not None:
        cv2.imwrite(os.path.join(output_dir, f"demo_comparison_{timestamp}.jpg"), grid)
    
    # Save text
    if text:
        with open(os.path.join(output_dir, f"demo_text_{timestamp}.txt"), "w") as f:
            f.write(text)
    
    # Save analysis
    if analysis_result:
        with open(os.path.join(output_dir, f"demo_analysis_{timestamp}.json"), "w") as f:
            json.dump(analysis_result, f, indent=2, default=str)
    
    print(f"\nDemo results saved to {output_dir}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Show Apple Silicon tips if applicable
    display_apple_silicon_tips()
    
    # Create output directory
    ensure_directory(args.output)
    
    # Either use provided image or run live scanner demo
    if args.image:
        print(f"Using image: {args.image}")
        
        # Step 1: Document Classification
        document, doc_type = demo_document_classification(args.image)
        
        if document is not None:
            # Step 2: Image Enhancement
            grid, enhanced = demo_image_enhancement(document, doc_type)
            
            # Step 3: Text Extraction and Analysis
            text, analysis = demo_text_extraction(enhanced, doc_type)
            
            # Save results
            save_demo_results(args.output, document, enhanced, grid, text, analysis)
            
            # Display results if GUI is enabled
            if not args.no_gui:
                cv2.imshow("Document Enhancement Comparison", grid)
                print("\nPress any key to exit...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        # Run live scanner demo
        demo_live_scanner(args)

if __name__ == "__main__":
    main()