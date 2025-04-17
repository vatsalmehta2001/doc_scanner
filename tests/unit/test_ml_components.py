#!/usr/bin/env python3
"""
Enhanced Test Suite for ML Document Scanner
Tests all ML components and their integration
"""

import os
import cv2
import numpy as np
import argparse
import time
import glob
from pathlib import Path

# Import ML components
from models.document_classifier import DocumentClassifier
from models.image_enhancer import ImageEnhancer
from models.text_analyzer import TextAnalyzer

def get_test_images(test_dir="scans", count=5):
    """Get a selection of test images"""
    image_files = []
    
    # Look for image files in the directory
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(test_dir, f"*{ext}")))
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Take the most recent images
    return image_files[:count]

def test_classifier(test_images, verbose=True):
    """Test the document classifier on multiple images"""
    print("\n=== Testing Document Classifier ===")
    
    # Load classifier
    classifier = DocumentClassifier()
    
    results = []
    for image_path in test_images:
        if verbose:
            print(f"\nTesting image: {os.path.basename(image_path)}")
        
        # Load test image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        # Time the classification
        start_time = time.time()
        doc_type, confidence = classifier.predict(image)
        elapsed = time.time() - start_time
        
        # Get feature details
        features = classifier.extract_features(image).flatten()
        
        if verbose:
            print(f"Classification: {doc_type}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Processing time: {elapsed:.3f} seconds")
            print(f"Key features:")
            print(f"  - Aspect ratio: {features[0]:.2f}")
            print(f"  - Edge density: {features[1]:.4f}")
            print(f"  - Contour density: {features[2]:.4f}")
        
        results.append({
            "image": os.path.basename(image_path),
            "type": doc_type,
            "confidence": confidence,
            "processing_time": elapsed
        })
    
    # Summarize results
    print("\nClassifier summary:")
    for doc_type in set(r['type'] for r in results):
        count = sum(1 for r in results if r['type'] == doc_type)
        avg_conf = sum(r['confidence'] for r in results if r['type'] == doc_type) / count
        print(f"  {doc_type}: {count} images, avg confidence: {avg_conf:.2f}")
    
    avg_time = sum(r['processing_time'] for r in results) / len(results)
    print(f"Average processing time: {avg_time:.3f} seconds")
    
    return results

def test_enhancer(test_images, output_dir="test_output", verbose=True):
    """Test the image enhancer on multiple images"""
    print("\n=== Testing Image Enhancer ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load enhancer
    enhancer = ImageEnhancer()
    
    results = []
    for image_path in test_images:
        if verbose:
            print(f"\nTesting image: {os.path.basename(image_path)}")
        
        # Load test image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        # Get document type for enhancement
        classifier = DocumentClassifier()
        doc_type, _ = classifier.predict(image)
        
        # Time the enhancement
        start_time = time.time()
        enhanced = enhancer.enhance(image, doc_type)
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"Document type for enhancement: {doc_type}")
            print(f"Processing time: {elapsed:.3f} seconds")
        
        # Save enhanced image for comparison
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"enhanced_{basename}.jpg")
        cv2.imwrite(output_path, enhanced)
        
        # Create comparison image
        if min(image.shape[0], image.shape[1]) >= 200:  # Only for reasonably sized images
            # Resize if needed
            h, w = image.shape[:2]
            target_h = min(400, h)
            scale = target_h / h
            resized = cv2.resize(image, None, fx=scale, fy=scale)
            enh_resized = cv2.resize(enhanced, None, fx=scale, fy=scale)
            
            # Create side-by-side comparison
            comparison = np.hstack([resized, enh_resized])
            comparison_path = os.path.join(output_dir, f"comparison_{basename}.jpg")
            cv2.imwrite(comparison_path, comparison)
            
            if verbose:
                print(f"Comparison saved to: {comparison_path}")
        
        results.append({
            "image": os.path.basename(image_path),
            "type": doc_type,
            "enhanced_image": output_path,
            "processing_time": elapsed
        })
    
    # Summarize results
    print("\nEnhancer summary:")
    avg_time = sum(r['processing_time'] for r in results) / len(results)
    print(f"Processed {len(results)} images")
    print(f"Enhanced images saved to: {output_dir}")
    print(f"Average processing time: {avg_time:.3f} seconds")
    
    return results

def test_ocr_pipeline(test_images, output_dir="test_output", verbose=True):
    """Test the complete OCR pipeline"""
    print("\n=== Testing Complete OCR Pipeline ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load components
    classifier = DocumentClassifier()
    enhancer = ImageEnhancer()
    analyzer = TextAnalyzer()
    
    # Try to import pytesseract for OCR
    try:
        import pytesseract
        ocr_available = True
    except ImportError:
        print("pytesseract not available - skipping OCR portion of test")
        ocr_available = False
    
    results = []
    for image_path in test_images:
        if verbose:
            print(f"\nProcessing image: {os.path.basename(image_path)}")
        
        # Load test image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        result = {
            "image": os.path.basename(image_path),
            "processing_time": 0,
            "classification_time": 0,
            "enhancement_time": 0,
            "ocr_time": 0,
            "analysis_time": 0
        }
        
        start_total = time.time()
        
        # 1. Classify document
        t_start = time.time()
        doc_type, confidence = classifier.predict(image)
        t_class = time.time() - t_start
        result["classification_time"] = t_class
        result["doc_type"] = doc_type
        result["confidence"] = confidence
        
        # 2. Enhance image
        t_start = time.time()
        enhanced = enhancer.enhance(image, doc_type)
        t_enhance = time.time() - t_start
        result["enhancement_time"] = t_enhance
        
        # Save enhanced image
        basename = os.path.splitext(os.path.basename(image_path))[0]
        enhanced_path = os.path.join(output_dir, f"enhanced_{basename}.jpg")
        cv2.imwrite(enhanced_path, enhanced)
        
        # 3. Extract text (if OCR available)
        text = None
        if ocr_available:
            try:
                t_start = time.time()
                
                # Convert to grayscale if needed
                if len(enhanced.shape) == 3:
                    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                else:
                    gray = enhanced
                
                # Apply thresholding if needed
                if doc_type in ["text_document", "form"]:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    ocr_image = binary
                else:
                    ocr_image = gray
                
                # Run OCR
                text = pytesseract.image_to_string(ocr_image)
                t_ocr = time.time() - t_start
                result["ocr_time"] = t_ocr
                
                # Save text to file
                text_path = os.path.join(output_dir, f"text_{basename}.txt")
                with open(text_path, "w") as f:
                    f.write(text)
                
                result["text_file"] = text_path
                
            except Exception as e:
                print(f"OCR error: {e}")
                text = None
        
        # 4. Analyze text
        if text:
            t_start = time.time()
            analysis = analyzer.analyze(text, doc_type)
            t_analyze = time.time() - t_start
            result["analysis_time"] = t_analyze
            result["analysis"] = analysis
            
            # Save analysis to file
            analysis_path = os.path.join(output_dir, f"analysis_{basename}.json")
            
            # Convert analysis to serializable format
            analysis_serializable = {}
            for k, v in analysis.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    analysis_serializable[k] = v
                elif isinstance(v, list):
                    analysis_serializable[k] = v if all(isinstance(i, (str, int, float, bool, type(None))) for i in v) else f"List with {len(v)} items"
                elif isinstance(v, dict):
                    analysis_serializable[k] = f"Dict with {len(v)} items"
                else:
                    analysis_serializable[k] = str(v)
            
            # Write to file (pretty printed)
            import json
            with open(analysis_path, "w") as f:
                json.dump(analysis_serializable, f, indent=2)
            
            result["analysis_file"] = analysis_path
        
        # Calculate total time
        result["processing_time"] = time.time() - start_total
        
        if verbose:
            print(f"Document type: {doc_type} (confidence: {confidence:.2f})")
            print(f"Processing times:")
            print(f"  - Classification: {t_class:.3f} seconds")
            print(f"  - Enhancement: {t_enhance:.3f} seconds")
            if ocr_available:
                print(f"  - OCR: {result.get('ocr_time', 0):.3f} seconds")
                print(f"  - Analysis: {result.get('analysis_time', 0):.3f} seconds")
            print(f"  - Total: {result['processing_time']:.3f} seconds")
        
        results.append(result)
    
    # Summarize results
    print("\nOCR Pipeline summary:")
    print(f"Processed {len(results)} images")
    print(f"Output saved to: {output_dir}")
    
    avg_times = {
        "classification": sum(r['classification_time'] for r in results) / len(results),
        "enhancement": sum(r['enhancement_time'] for r in results) / len(results),
        "total": sum(r['processing_time'] for r in results) / len(results)
    }
    
    if ocr_available:
        ocr_results = [r for r in results if 'ocr_time' in r]
        if ocr_results:
            avg_times["ocr"] = sum(r['ocr_time'] for r in ocr_results) / len(ocr_results)
            
        analysis_results = [r for r in results if 'analysis_time' in r]
        if analysis_results:
            avg_times["analysis"] = sum(r['analysis_time'] for r in analysis_results) / len(analysis_results)
    
    print("\nAverage processing times:")
    for stage, time_value in avg_times.items():
        print(f"  - {stage.capitalize()}: {time_value:.3f} seconds")
    
    return results

def main():
    """Main test function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test ML components")
    parser.add_argument("--images", "-i", type=str, default="scans",
                        help="Directory containing test images")
    parser.add_argument("--count", "-c", type=int, default=5,
                        help="Number of images to test")
    parser.add_argument("--output", "-o", type=str, default="test_output",
                        help="Output directory for test results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--components", choices=["classifier", "enhancer", "ocr", "all"],
                        default="all", help="Components to test")
    
    args = parser.parse_args()
    
    print("Testing ML Components for Document Scanner")
    
    # Get test images
    test_images = get_test_images(args.images, args.count)
    if not test_images:
        print(f"No test images found in '{args.images}'. Using a blank test image.")
        # Create a blank test image
        test_dir = "test_images"
        os.makedirs(test_dir, exist_ok=True)
        test_image = os.path.join(test_dir, "test_blank.jpg")
        
        # Create blank images with different aspect ratios
        blank_images = [
            # Standard document
            {"name": "test_document.jpg", "size": (800, 1100), "text": "TEST DOCUMENT"},
            # Receipt
            {"name": "test_receipt.jpg", "size": (500, 1200), "text": "TEST RECEIPT"},
            # ID card
            {"name": "test_id_card.jpg", "size": (400, 250), "text": "TEST ID CARD"},
        ]
        
        test_images = []
        for img_info in blank_images:
            img_path = os.path.join(test_dir, img_info["name"])
            blank_img = np.ones((img_info["size"][1], img_info["size"][0], 3), dtype=np.uint8) * 255
            
            # Add text
            cv2.putText(blank_img, img_info["text"], 
                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Add border
            cv2.rectangle(blank_img, (20, 20), 
                         (img_info["size"][0]-20, img_info["size"][1]-20), 
                         (0, 0, 0), 2)
            
            cv2.imwrite(img_path, blank_img)
            test_images.append(img_path)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run tests based on requested components
    results = {}
    
    if args.components in ["classifier", "all"]:
        results["classifier"] = test_classifier(test_images, args.verbose)
    
    if args.components in ["enhancer", "all"]:
        results["enhancer"] = test_enhancer(test_images, args.output, args.verbose)
    
    if args.components in ["ocr", "all"]:
        results["ocr"] = test_ocr_pipeline(test_images, args.output, args.verbose)
    
    # Summary
    print("\n=== Test Summary ===")
    for component, result in results.items():
        print(f"{component.capitalize()}: Processed {len(result)} images")
    
    print(f"\nAll test results saved to: {args.output}")

if __name__ == "__main__":
    main()