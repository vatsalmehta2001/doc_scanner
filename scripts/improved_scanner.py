#!/usr/bin/env python3
"""
Improved document scanner with enhanced detection and OCR capabilities.
Features:
- Improved document detection sensitivity
- Enhanced OCR preprocessing for better text recognition
- Adaptive thresholding for different text styles
- Manual capture mode when automatic detection fails
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import subprocess
import platform

# Create output directory
os.makedirs("scans", exist_ok=True)

# Function to check and install required packages
def ensure_dependencies():
    try:
        # Try importing pytesseract
        import pytesseract
        import pyperclip
        return True
    except ImportError:
        print("Installing required dependencies...")
        try:
            subprocess.check_call(["pip", "install", "pytesseract", "pyperclip"])
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


def detect_document(image, sensitivity=0.5):
    """
    Detect a document in the given image with improved sensitivity
    
    Args:
        image: Input image (numpy array)
        sensitivity: Detection sensitivity (0.0-1.0, higher = more sensitive)
    
    Returns:
        tuple: (perspective_corrected_document, corners)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Dynamic edge detection thresholds based on sensitivity
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = high_thresh * (0.3 + (0.2 * sensitivity))  # Adjust based on sensitivity
    
    # Edge detection using Canny with dynamic thresholds
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    # Dilate the edges to close gaps - more dilation with higher sensitivity
    kernel_size = int(3 + (sensitivity * 4))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
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
            if len(approx) != 4 and sensitivity > 0.7:
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


def enhance_document(image, mode="auto"):
    """Enhanced document processing with improved text clarity"""
    if image is None:
        return None
    
    # Auto-detect the best enhancement mode if set to "auto"
    if mode == "auto":
        # Calculate average brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Calculate color variance
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        color_variance = np.std(saturation)
        
        # Decide enhancement mode based on image properties
        if color_variance > 50:  # High color variance indicates a colorful image
            mode = "color"
        elif brightness < 150:  # Darker images might be photos
            mode = "photo"
        else:  # Default to text for light images with low color variance
            mode = "text"
    
    # Apply enhancement based on mode
    if mode == "text":
        # Optimize for text documents
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
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
        
    elif mode == "color":
        # Enhance colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Apply mild sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return sharpened
        
    else:
        # Default: just return the original
        return image


def extract_text(image, ocr_mode="advanced"):
    """
    Extract text from image using advanced OCR techniques
    
    Args:
        image: Input document image
        ocr_mode: OCR processing mode (advanced, fast, decorative)
    
    Returns:
        str: Extracted text
    """
    try:
        import pytesseract
        
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        if ocr_mode == "advanced":
            # Multi-stage OCR with different preprocessing methods
            
            # 1. Basic preprocessing - adaptive threshold
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 2. CLAHE preprocessing for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)
            _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 3. Edge enhancement preprocessing
            blurred = cv2.GaussianBlur(gray, (0, 0), 3)
            edge_enhanced = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            _, edge_binary = cv2.threshold(edge_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Create multiple versions at different scales for better OCR
            scales = [1.0, 1.5, 2.0]
            results = []
            
            for scale in scales:
                # Resize each preprocessed image
                if scale != 1.0:
                    scaled_adaptive = cv2.resize(adaptive, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    scaled_otsu = cv2.resize(otsu, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    scaled_edge = cv2.resize(edge_binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                else:
                    scaled_adaptive = adaptive
                    scaled_otsu = otsu
                    scaled_edge = edge_binary
                
                # Try OCR with different configurations
                config = r'--oem 3 --psm 6'  # Assume a single uniform block of text
                
                # Process each preprocessed image
                text1 = pytesseract.image_to_string(scaled_adaptive, config=config)
                text2 = pytesseract.image_to_string(scaled_otsu, config=config)
                text3 = pytesseract.image_to_string(scaled_edge, config=config)
                
                # Also try with page segmentation mode 4 (single column of text)
                config = r'--oem 3 --psm 4'
                text4 = pytesseract.image_to_string(scaled_adaptive, config=config)
                
                # Collect all results
                results.extend([text1, text2, text3, text4])
            
            # Choose the longest text result (often the most complete)
            results = [r for r in results if r.strip()]  # Remove empty results
            if not results:
                return "No text detected"
                
            # Sort by length and return the longest result
            return max(results, key=len)
            
        elif ocr_mode == "decorative":
            # Special processing for decorative or stylized text
            
            # Apply multiple thresholds to handle different text styles
            methods = [
                # Regular binary threshold
                lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
                # Inverted threshold for light text on dark background
                lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1],
                # Adaptive threshold with different parameters
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
            ]
            
            # Scale factors to try
            scales = [1.0, 1.5, 2.0, 3.0]
            
            results = []
            
            # Try each preprocessing method with each scale
            for method in methods:
                processed = method(gray)
                
                for scale in scales:
                    if scale != 1.0:
                        scaled = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    else:
                        scaled = processed
                    
                    # Try with different PSM modes
                    for psm in [6, 7, 8, 11, 13]:
                        config = f'--oem 3 --psm {psm}'
                        text = pytesseract.image_to_string(scaled, config=config)
                        if text.strip():
                            results.append(text)
            
            if not results:
                return "No text detected"
            
            # Return the longest result
            return max(results, key=len)
            
        else:  # "fast" mode - quicker but less accurate
            # Simple preprocessing
            # Increase image size for better OCR results
            resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            
            # Run OCR
            text = pytesseract.image_to_string(resized)
            return text if text.strip() else "No text detected"
            
    except Exception as e:
        print(f"OCR error: {e}")
        return "OCR error: " + str(e)


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
    
    print("\nImproved Document Scanner with OCR")
    print("--------------------------------")
    print("Press 's' to save the current scan and copy text to clipboard")
    print("Press 'e' to change enhancement mode (text, bw, color)")
    print("Press 'd' to toggle OCR mode (advanced, decorative)")
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
    enhancement_mode = "bw"  # Default mode
    ocr_mode = "advanced"  # Default OCR mode
    manual_mode = False  # Auto detection by default
    detection_sensitivity = 0.7  # Default sensitivity (0.0-1.0)
    status_message = ""
    last_message_time = 0
    
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
                f"Mode: {enhancement_mode} | OCR: {ocr_mode} | Sensitivity: {detection_sensitivity:.1f}",
                (10, display_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
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
            cv2.imshow("Improved Document Scanner", display_frame)
            
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
                    # Enhance document
                    enhanced = enhance_document(current_document, enhancement_mode)
                    
                    # Extract text if OCR is available
                    if ocr_available:
                        text = extract_text(enhanced, ocr_mode)
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
                modes = ["text", "bw", "color"]
                current_index = modes.index(enhancement_mode) if enhancement_mode in modes else 0
                enhancement_mode = modes[(current_index + 1) % len(modes)]
                status_message = f"Enhancement mode changed to: {enhancement_mode}"
                last_message_time = time.time()
                print(status_message)
            
            # 'd' to toggle OCR mode
            elif key == ord('d'):
                modes = ["advanced", "decorative"]
                current_index = modes.index(ocr_mode) if ocr_mode in modes else 0
                ocr_mode = modes[(current_index + 1) % len(modes)]
                status_message = f"OCR mode changed to: {ocr_mode}"
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