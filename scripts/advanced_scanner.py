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
                box = np.int0(box)
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


def main():
    """Main function to run the document scanner"""
    # Check and install dependencies
    ocr_available = ensure_dependencies()
    
    print("\nAdvanced Document Scanner with OCR")
    print("--------------------------------")
    print("Press 's' to save the current scan and copy text to clipboard")
    print("Press 'e' to change enhancement mode (text, bw, inverted, color)")
    print("Press 'm' to toggle manual capture mode")
    print("Press '+'/'-' to adjust detection sensitivity")
    print("Press 'q' to quit")
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Variables
    current_document = None
    current_corners = None
    enhancement_mode = "auto"  # Default mode is auto-detection
    manual_mode = False  # Auto detection by default
    detection_sensitivity = 0.7  # Default sensitivity (0.0-1.0)
    status_message = ""
    last_message_time = 0
    current_ocr_text = ""
    
    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Detect document in the frame (skip if in manual mode)
            if not manual_mode:
                document, corners = detect_document(frame, detection_sensitivity)
                current_document = document
                current_corners = corners
            
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # If in manual mode, provide guidance for capture
            if manual_mode:
                # Draw capture area in the center
                h, w = display_frame.shape[:2]
                margin_x = int(w * 0.1)
                margin_y = int(h * 0.1)
                cv2.rectangle(display_frame, 
                            (margin_x, margin_y), 
                            (w - margin_x, h - margin_y), 
                            (0, 255, 255), 2)
                
                # Add instructions
                cv2.putText(
                    display_frame, 
                    "Manual Mode - Press 's' to capture area", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 255), 
                    2
                )
            # Otherwise show detected document
            elif current_corners is not None:
                cv2.polylines(display_frame, [current_corners], True, (0, 255, 0), 2)
                
                # Add a text indicator
                cv2.putText(
                    display_frame, 
                    "Document Detected - Press 's' to save & extract text", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
            
            # Show current modes and settings
            cv2.putText(
                display_frame,
                f"Enhancement: {enhancement_mode} | {'Manual' if manual_mode else 'Auto'} | Sens: {detection_sensitivity:.1f}",
                (10, display_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Show OCR text preview if available (truncated)
            if current_ocr_text and not manual_mode:
                # Get first line or truncate
                if '\n' in current_ocr_text:
                    preview = current_ocr_text.split('\n')[0] + " ..."
                else:
                    preview = (current_ocr_text[:40] + "...") if len(current_ocr_text) > 40 else current_ocr_text
                    
                cv2.putText(
                    display_frame,
                    f"OCR: {preview}",
                    (10, display_frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
            
            # Show status message for 3 seconds
            if status_message and time.time() - last_message_time < 3.0:
                cv2.putText(
                    display_frame,
                    status_message,
                    (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Show the frame
            cv2.imshow("Advanced Document Scanner", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # 's' to save current document or capture manual area
            if key == ord('s'):
                if manual_mode:
                    # In manual mode, capture the center area
                    h, w = frame.shape[:2]
                    margin_x = int(w * 0.1)
                    margin_y = int(h * 0.1)
                    
                    current_document = frame[margin_y:h-margin_y, margin_x:w-margin_x].copy()
                    
                    # Update status
                    status_message = "Manual area captured"
                    last_message_time = time.time()
                    print(status_message)
                
                # Save and process the captured document
                if current_document is not None:
                    # Start status message
                    status_message = "Processing document..."
                    print(status_message)
                    
                    # Update status in the UI
                    cv2.putText(
                        display_frame,
                        status_message,
                        (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    cv2.imshow("Advanced Document Scanner", display_frame)
                    cv2.waitKey(1)  # Update display
                    
                    # Enhance document for display and saving
                    enhanced = enhance_document_for_display(current_document, enhancement_mode)
                    
                    # Extract text if OCR is available
                    text = None
                    if ocr_available:
                        status_message = "Running OCR..."
                        print(status_message)
                        
                        # Update status in the UI
                        cv2.putText(
                            display_frame,
                            status_message,
                            (10, display_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )
                        cv2.imshow("Advanced Document Scanner", display_frame)
                        cv2.waitKey(1)  # Update display
                        
                        text = extract_text_advanced(current_document)
                        current_ocr_text = text
                        
                        if text and text != "No text detected":
                            # Copy text to clipboard
                            if copy_to_clipboard(text):
                                clipboard_status = "Text copied to clipboard"
                            else:
                                clipboard_status = "Failed to copy text"
                                
                            # Print extracted text
                            print("\nExtracted Text:")
                            print("--------------")
                            print(text)
                            print("--------------")
                        else:
                            clipboard_status = "No text detected"
                            text = "No text detected"
                    else:
                        clipboard_status = "OCR not available"
                        text = None
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"scans/scan_{timestamp}.jpg"
                    
                    # Save the document
                    cv2.imwrite(filename, enhanced)
                    
                    # Also save a text file with the OCR results if text was detected
                    if text and text != "No text detected" and ocr_available:
                        text_filename = f"scans/scan_{timestamp}.txt"
                        with open(text_filename, "w") as f:
                            f.write(text)
                    
                    # Update status message
                    status_message = f"Saved: {os.path.basename(filename)} - {clipboard_status}"
                    last_message_time = time.time()
                    print(status_message)
                else:
                    status_message = "No document detected to save"
                    last_message_time = time.time()
                    print(status_message)
            
            # 'e' to change enhancement mode
            elif key == ord('e'):
                modes = ["auto", "text", "bw", "inverted", "color"]
                current_index = modes.index(enhancement_mode) if enhancement_mode in modes else 0
                enhancement_mode = modes[(current_index + 1) % len(modes)]
                status_message = f"Enhancement mode changed to: {enhancement_mode}"
                last_message_time = time.time()
                print(status_message)
            
            # 'm' to toggle manual mode
            elif key == ord('m'):
                manual_mode = not manual_mode
                status_message = f"Manual mode: {'ON' if manual_mode else 'OFF'}"
                last_message_time = time.time()
                print(status_message)
            
            # '+' to increase detection sensitivity
            elif key == ord('+') or key == ord('='):
                detection_sensitivity = min(1.0, detection_sensitivity + 0.1)
                status_message = f"Detection sensitivity: {detection_sensitivity:.1f}"
                last_message_time = time.time()
                print(status_message)
            
            # '-' to decrease detection sensitivity
            elif key == ord('-'):
                detection_sensitivity = max(0.1, detection_sensitivity - 0.1)
                status_message = f"Detection sensitivity: {detection_sensitivity:.1f}"
                last_message_time = time.time()
                print(status_message)
            
            # 'q' to quit
            elif key == ord('q'):
                break
                
    finally:
        # Clean up resources
        camera.release()
        cv2.destroyAllWindows()
        print("Document Scanner closed")


if __name__ == "__main__":
    main()