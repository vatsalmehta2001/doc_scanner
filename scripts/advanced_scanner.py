#!/usr/bin/env python3
"""
Advanced document scanner with specialized OCR for difficult texts
Features:
- Enhanced dark background text detection
- Multiple color space transformations for better text isolation
- AI-assisted contrast enhancement
- Specialized processing for inverted text (light on dark)
- Multi-scale character recognition
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import subprocess
import platform
import tempfile
import shutil
import argparse
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json

# Make magic import optional
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("Warning: python-magic not available. File type detection will be limited.")

from doc_scanner.document import DocumentProcessor
from doc_scanner.utils import ImageEnhancer, Timer
from doc_scanner.text_processor import TextProcessor

# Create output directory
os.makedirs("scans", exist_ok=True)

# Function to check and install required packages
def ensure_dependencies():
    try:
        # Try importing pytesseract and pyperclip
        import pytesseract
        import pyperclip
        
        # Check if tesseract is properly installed
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {tesseract_version}")
        
        return True
    except ImportError:
        print("Installing required dependencies...")
        try:
            subprocess.check_call(["pip", "install", "pytesseract", "pyperclip", "scikit-image"])
            print("Dependencies installed successfully.")
            
            # Check if tesseract is installed
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                print("Tesseract is installed and working.")
            except Exception:
                print("\nIMPORTANT: Tesseract OCR engine needs to be installed separately.")
                print("Please install Tesseract from: https://github.com/tesseract-ocr/tesseract")
                if platform.system() == "Darwin":  # macOS
                    print("\nOn macOS with Homebrew, run:")
                    print("  brew install tesseract")
                elif platform.system() == "Windows":
                    print("\nOn Windows, download installer from:")
                    print("  https://github.com/UB-Mannheim/tesseract/wiki")
                else:  # Linux
                    print("\nOn Ubuntu/Debian, run:")
                    print("  sudo apt install tesseract-ocr")
                
                input("\nPress Enter to continue without OCR functionality...")
                
            return True
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            print("Continuing without OCR functionality.")
            return False


def detect_document(image, sensitivity=0.7):
    """
    Detect a document in the given image with improved sensitivity for dark backgrounds
    
    Args:
        image: Input image (numpy array)
        sensitivity: Detection sensitivity (0.0-1.0, higher = more sensitive)
    
    Returns:
        tuple: (perspective_corrected_document, corners)
    """
    # Create multiple processing paths for better detection
    # This helps with both light and dark documents
    
    # Path 1: Standard grayscale processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Path 2: HSV processing (better for colored documents)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1]
    
    # Path 3: Inverted processing (better for dark documents)
    inverted = cv2.bitwise_not(gray)
    blurred_inv = cv2.GaussianBlur(inverted, (5, 5), 0)
    
    # Combine different edge detection methods
    edges1 = cv2.Canny(blurred, 50, 150)
    edges2 = cv2.Canny(saturation, 50, 150)
    edges3 = cv2.Canny(blurred_inv, 50, 150)
    
    # Combine edges from different methods
    combined_edges = cv2.bitwise_or(edges1, edges2)
    combined_edges = cv2.bitwise_or(combined_edges, edges3)
    
    # Dilate the edges to close gaps - more dilation with higher sensitivity
    kernel_size = int(3 + (sensitivity * 4))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(combined_edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Initialize document and corners
    document = None
    corners = None
    
    # Get image area for filtering small contours
    image_area = image.shape[0] * image.shape[1]
    min_area_ratio = 0.03 * (1.0 - sensitivity)  # Lower threshold with higher sensitivity
    
    # Loop through contours
    for contour in contours[:10]:  # Check more contours with higher sensitivity
        # Skip small contours
        if cv2.contourArea(contour) < (image_area * min_area_ratio):
            continue
            
        # Get perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Approximate the contour shape - more permissive with higher sensitivity
        epsilon = (0.04 - (0.02 * sensitivity)) * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Accept shapes with 4 points (rectangles) or shapes that are close (3-6 points)
        if 3 <= len(approx) <= 6:
            # If not exactly 4 points but sensitivity is high, approximate to 4 points
            if len(approx) != 4 and sensitivity > 0.5:
                # Find the minimum area rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                approx = box
            
            # If we have 4 points (or approximated to 4), we likely have found our document
            if len(approx) == 4:
                corners = approx
                document = four_point_transform(image, corners.reshape(4, 2))
                break
    
    return document, corners


def enhance_document_for_ocr(image, mode="auto"):
    """
    Apply specialized enhancements optimized for OCR text extraction
    
    Args:
        image: Input document image
        mode: Enhancement mode (auto, text, dark, light)
        
    Returns:
        List of processed images for OCR
    """
    if image is None:
        return []
    
    result_images = []
    
    # Get grayscale version
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Determine if the image has dark background using histogram analysis
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dark_pixels = np.sum(hist[:128])
    light_pixels = np.sum(hist[128:])
    has_dark_background = dark_pixels > light_pixels
    
    # Process differently based on auto-detection or specified mode
    if mode == "auto":
        if has_dark_background:
            mode = "dark"
        else:
            mode = "text"
    
    # Create basic binarized version
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result_images.append(binary_otsu)
    
    # Process based on detected/specified mode
    if mode == "text" or mode == "light":
        # Standard text processing (dark text on light background)
        
        # Adaptive thresholding with different parameters
        adaptive1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 15, 5)
        result_images.extend([adaptive1, adaptive2])
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        _, clahe_binary = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result_images.append(clahe_binary)
        
    elif mode == "dark" or mode == "inverted":
        # Specialized processing for light text on dark background
        
        # Invert the image first
        inverted = cv2.bitwise_not(gray)
        result_images.append(inverted)
        
        # Apply thresholding to inverted image
        _, inv_binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result_images.append(inv_binary)
        
        # Apply adaptive thresholding to inverted image
        inv_adaptive = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        result_images.append(inv_adaptive)
        
        # Enhanced contrast on inverted image
        alpha = 1.5  # Contrast control
        beta = 10    # Brightness control
        enhanced_inv = cv2.convertScaleAbs(inverted, alpha=alpha, beta=beta)
        _, enh_binary = cv2.threshold(enhanced_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result_images.append(enh_binary)
    
    # Add specialized filters for all modes
    
    # Bilateral filter for noise reduction while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    _, bilateral_binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result_images.append(bilateral_binary)
    
    # Edge enhancement
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    edge_enhanced = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    _, edge_binary = cv2.threshold(edge_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result_images.append(edge_binary)
    
    # Apply morphological operations for cleaning
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)
    result_images.append(cleaned)
    
    return result_images


def extract_text_advanced(image):
    """
    Extract text from image using advanced multi-stage OCR
    with special handling for difficult text conditions
    
    Args:
        image: Input document image
        
    Returns:
        str: Best extracted text result
    """
    try:
        import pytesseract
        from PIL import Image, ImageEnhance
        
        # Skip processing if image is None
        if image is None:
            return "No document detected"
        
        # Create temp directory for working files
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Auto-detect if image has dark background
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            dark_pixels = np.sum(hist[:128])
            light_pixels = np.sum(hist[128:])
            has_dark_background = dark_pixels > light_pixels
            
            # Process document for OCR with specialized enhancements
            enhanced_images = []
            
            # Process with standard enhancement
            enhanced_images.extend(enhance_document_for_ocr(image, "auto"))
            
            # Always also process with opposite mode to what was detected
            if has_dark_background:
                enhanced_images.extend(enhance_document_for_ocr(image, "light"))
            else:
                enhanced_images.extend(enhance_document_for_ocr(image, "dark"))
            
            # Create grayscale and color versions of the original at multiple scales
            scales = [1.0, 1.5, 2.0, 3.0]
            for scale in scales:
                if scale != 1.0:
                    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    if len(image.shape) == 3:
                        scaled_gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
                    else:
                        scaled_gray = scaled
                    enhanced_images.append(scaled_gray)
                    
                    # For dark background, add inverted version
                    if has_dark_background:
                        enhanced_images.append(cv2.bitwise_not(scaled_gray))
            
            # Convert to PIL Image format for additional processing and save to temp files
            pil_images = []
            image_paths = []
            
            for i, img in enumerate(enhanced_images):
                # Convert OpenCV image to PIL format
                if len(img.shape) == 2:  # Grayscale
                    pil_img = Image.fromarray(img)
                else:  # Color
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                
                # Additional PIL-based enhancements
                if i % 2 == 0:  # Apply to every other image for variety
                    enhancer = ImageEnhance.Contrast(pil_img)
                    pil_img = enhancer.enhance(1.5)
                
                pil_images.append(pil_img)
                
                # Save to temp file
                img_path = os.path.join(temp_dir, f"temp_ocr_{i}.png")
                pil_img.save(img_path)
                image_paths.append(img_path)
            
            # Run OCR with different configurations
            results = []
            
            # PSM modes to try (Page Segmentation Modes)
            psm_modes = [6, 4, 11, 3]  # Single block, Column, Single line, Auto
            
            # Process all images with multiple configurations
            for img_path in image_paths:
                for psm in psm_modes:
                    config = f'--oem 3 --psm {psm}'
                    try:
                        # Use lang=eng+osd for better text orientation detection
                        text = pytesseract.image_to_string(img_path, config=config, lang='eng')
                        if text.strip():
                            results.append(text)
                    except Exception as e:
                        print(f"OCR error with PSM {psm}: {e}")
            
            # If no results, try direct OpenCV processing
            if not results:
                for img in enhanced_images:
                    try:
                        text = pytesseract.image_to_string(img)
                        if text.strip():
                            results.append(text)
                    except Exception as e:
                        print(f"Direct OCR error: {e}")
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # If still no results
            if not results:
                return "No text detected"
            
            # Return the best result (longest text, most likely to be complete)
            # First clean up results by removing empty lines and excess whitespace
            cleaned_results = []
            for text in results:
                # Remove excess whitespace and empty lines
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                cleaned_text = "\n".join(lines)
                cleaned_results.append(cleaned_text)
            
            # Filter out empty results and find the longest
            cleaned_results = [r for r in cleaned_results if r]
            if not cleaned_results:
                return "No text detected"
                
            # Find the result with the most characters (not counting whitespace)
            best_result = max(cleaned_results, key=lambda x: len(x.replace(" ", "").replace("\n", "")))
            return best_result
            
        finally:
            # Make sure temp directory is removed
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    except Exception as e:
        print(f"OCR processing error: {e}")
        return f"Error extracting text: {str(e)}"


def enhance_document_for_display(image, mode="auto"):
    """
    Enhance document for display and saving
    
    Args:
        image: Input document image
        mode: Enhancement mode (auto, text, photo, bw, color)
        
    Returns:
        Enhanced image
    """
    if image is None:
        return None
    
    # Auto-detect the best enhancement mode if set to "auto"
    if mode == "auto":
        # Get grayscale version
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Determine if the image has dark background using histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dark_pixels = np.sum(hist[:128])
        light_pixels = np.sum(hist[128:])
        has_dark_background = dark_pixels > light_pixels
        
        # Calculate color variance
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1]
            color_variance = np.std(saturation)
        else:
            color_variance = 0
        
        # Decide enhancement mode based on image properties
        if has_dark_background:
            mode = "inverted"
        elif color_variance > 50:  # High color variance indicates a colorful image
            mode = "color"
        else:  # Default to text for light images with low color variance
            mode = "text"
    
    # Apply enhancement based on mode
    if mode == "text":
        # Optimize for text documents
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding with better parameters for text clarity
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )
        
        # Apply mild denoising to clean up the image
        denoised = cv2.fastNlMeansDenoising(adaptive, None, 10, 7, 21)
        
        result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return result
        
    elif mode == "bw":
        # Convert to high-contrast black and white
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        return result
        
    elif mode == "inverted":
        # Specialized processing for inverted colors (light text on dark background)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Invert colors to make processing more consistent
        inverted = cv2.bitwise_not(gray)
        
        # Apply thresholding
        _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply mild denoising
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Invert back to original color scheme
        result = cv2.bitwise_not(denoised)
        
        # Convert to BGR
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result
        
    elif mode == "color":
        # Enhance colors
        if len(image.shape) != 3:
            # Convert grayscale to BGR if needed
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # Convert to HSV for better color processing
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Increase saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)
        
        # Increase value (brightness) slightly
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Apply mild sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return sharpened
        
    else:
        # Default to original image
        if len(image.shape) == 2:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            return image.copy()


def copy_to_clipboard(text):
    """Copy text to clipboard"""
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception as e:
        print(f"Clipboard error: {e}")
        return False


def order_points(pts):
    """Order points in a consistent order: top-left, top-right, bottom-right, bottom-left"""
    # Initialize a list of coordinates
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference
    # The bottom-left point will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # Return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    """Apply perspective transform to get a top-down view"""
    # Order the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Define destination points for transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


class AdvancedDocumentScanner:
    def __init__(self, output_dir: str = "scanned_documents"):
        """Initialize the advanced document scanner"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.image_enhancer = ImageEnhancer()
        self.text_processor = TextProcessor()
        
        # Initialize camera
        self.camera = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.processing_queue = queue.Queue(maxsize=10)
        self.is_running = False
        
        # Processing settings
        self.auto_capture = False
        self.show_preview = True
        self.current_mode = "normal"  # normal, hdr, text
        
        # Initialize timer
        self.timer = Timer()
        
    def start(self):
        """Start the document scanner"""
        self.is_running = True
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        # Start UI loop
        self._ui_loop()
        
    def stop(self):
        """Stop the document scanner"""
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.join()
        if self.processing_thread:
            self.processing_thread.join()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
    def _camera_loop(self):
        """Camera capture loop"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # Put frame in queue, skip if full
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                continue
                
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Process frame
            processed_frame = self._process_frame(frame)
            
            # Put in processing queue for UI
            try:
                self.processing_queue.put(processed_frame, block=False)
            except queue.Full:
                continue
                
    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame"""
        self.timer.start()
        
        # Detect document
        corners = self.doc_processor.detect_document_corners(frame)
        
        result = {
            'frame': frame,
            'corners': corners,
            'processing_time': self.timer.stop()
        }
        
        if corners is not None:
            # Transform document
            warped = self.doc_processor.four_point_transform(frame, corners)
            
            # Enhance based on mode
            if self.current_mode == "hdr":
                enhanced = self._apply_hdr_enhancement(warped)
            elif self.current_mode == "text":
                enhanced = self._optimize_for_text(warped)
            else:
                enhanced = self.image_enhancer.enhance(warped)
                
            result['warped'] = warped
            result['enhanced'] = enhanced
            
            # Auto-capture if enabled and good quality
            if self.auto_capture and self._check_quality(enhanced):
                self._save_document(enhanced)
                
        return result
    
    def _apply_hdr_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply HDR-like enhancement"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply additional contrast enhancement
        enhanced = cv2.addWeighted(enhanced, 1.2, enhanced, 0, 0)
        
        return enhanced
    
    def _optimize_for_text(self, image: np.ndarray) -> np.ndarray:
        """Optimize image for text recognition"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)
        
        # Apply adaptive threshold with optimized parameters
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Remove noise while preserving text
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Ensure black text on white background
        if np.mean(denoised) < 127:
            denoised = cv2.bitwise_not(denoised)
        
        return denoised
    
    def _check_quality(self, image: np.ndarray) -> bool:
        """Check if image quality is good enough for capture"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image quality metrics
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Define quality thresholds
        return (blur > 100 and  # Not too blurry
                20 < brightness < 235 and  # Good brightness
                contrast > 40)  # Good contrast
    
    def _save_document(self, image: np.ndarray):
        """Save processed document"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"doc_{timestamp}.png"
            
            # Save image with high quality
            cv2.imwrite(str(filename), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"\nSaved document image: {filename}")
            
            # Optimize image for text extraction
            ocr_image = self._optimize_for_text(image.copy())
            
            # Extract text with higher quality settings
            text_data = self.text_processor.extract_text(ocr_image)
            
            # Save text if we got meaningful content
            if text_data['full_text'].strip():
                text_file = filename.with_suffix('.txt')
                with open(text_file, 'w') as f:
                    f.write(text_data['full_text'])
                print(f"Saved extracted text: {text_file}")
                
                # Save structured data if we found any
                has_data = any(len(v) > 0 for v in text_data['structured_data'].values())
                if has_data:
                    struct_file = filename.with_suffix('.json')
                    with open(struct_file, 'w') as f:
                        json.dump(text_data['structured_data'], f, indent=2)
                    print(f"Saved structured data: {struct_file}")
                
                print(f"\nExtracted text preview:")
                preview = text_data['full_text'][:200] + "..." if len(text_data['full_text']) > 200 else text_data['full_text']
                print(preview)
            else:
                print("No text could be extracted from the document")
                
        except Exception as e:
            print(f"Error saving document: {str(e)}")
    
    def _ui_loop(self):
        """Main UI loop"""
        window_name = "Advanced Document Scanner"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while self.is_running:
            try:
                result = self.processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Prepare display frame
            display_frame = result['frame'].copy()
            
            # Draw document corners if detected
            if result['corners'] is not None:
                corners = result['corners'].astype(np.int32)
                cv2.polylines(
                    display_frame,
                    [corners],
                    True,
                    (0, 255, 0),
                    2
                )
                
                # Show processing time
                cv2.putText(
                    display_frame,
                    f"Processing: {result['processing_time']:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Show enhanced view if available
                if 'enhanced' in result:
                    cv2.imshow("Enhanced View", result['enhanced'])
            
            # Show main frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('c'):
                if 'enhanced' in result:
                    self._save_document(result['enhanced'])
            elif key == ord('m'):
                self._cycle_mode()
            elif key == ord('a'):
                self.auto_capture = not self.auto_capture
            elif key == ord('p'):
                self.show_preview = not self.show_preview
    
    def _cycle_mode(self):
        """Cycle through different processing modes"""
        modes = ["normal", "hdr", "text"]
        current_idx = modes.index(self.current_mode)
        self.current_mode = modes[(current_idx + 1) % len(modes)]
        print(f"Switched to {self.current_mode} mode")

def main():
    parser = argparse.ArgumentParser(description="Advanced Document Scanner")
    parser.add_argument(
        "--output",
        default="scanned_documents",
        help="Output directory for scanned documents"
    )
    args = parser.parse_args()
    
    scanner = AdvancedDocumentScanner(args.output)
    try:
        scanner.start()
    except KeyboardInterrupt:
        pass
    finally:
        scanner.stop()

if __name__ == "__main__":
    main()