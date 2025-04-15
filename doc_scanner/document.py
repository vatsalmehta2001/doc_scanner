"""
Document processing module
Handles document detection, enhancement, and classification
"""

import cv2
import numpy as np
import os

# Import ML models if available
try:
    from models.document_classifier import DocumentClassifier
    from models.image_enhancer import ImageEnhancer
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    print("Warning: ML models not available. Some features will be disabled.")

# Singleton instances for ML models
_document_classifier = None
_image_enhancer = None

def get_document_classifier():
    """Get or create document classifier instance"""
    global _document_classifier
    if _document_classifier is None and ML_MODELS_AVAILABLE:
        _document_classifier = DocumentClassifier()
    return _document_classifier

def get_image_enhancer():
    """Get or create image enhancer instance"""
    global _image_enhancer
    if _image_enhancer is None and ML_MODELS_AVAILABLE:
        _image_enhancer = ImageEnhancer()
    return _image_enhancer

def detect_document(image, sensitivity=0.7):
    """
    Detect document in image using edge detection and contour analysis
    
    Args:
        image: Input BGR image
        sensitivity: Detection sensitivity (0.0-1.0, higher = more sensitive)
    
    Returns:
        tuple: (document_image, document_corners) if detected, else (None, None)
    """
    if image is None:
        return None, None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Blur and apply edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    
    # Dilate edges to close gaps
    dilated = cv2.dilate(edges, np.ones((3, 3)), iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Get image dimensions
    image_height, image_width = gray.shape[:2]
    image_area = image_height * image_width
    
    # Find the document contour
    document_contour = None
    corners = None
    
    # Minimum area ratio (0.03 = 3% of image)
    # Adjust based on sensitivity
    min_area_ratio = 0.03 * (1.0 - sensitivity)  # Lower threshold with higher sensitivity
    
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < (image_area * min_area_ratio):
            continue
        
        # Approximate contour shape
        epsilon = 0.02 * cv2.arcLength(contour, True) * (2.0 - sensitivity)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If approximated contour has 4 points, it might be our document
        if len(approx) == 4:
            document_contour = approx
            corners = approx
            break
    
    # If we found a contour, extract and transform the document
    if document_contour is not None:
        # Order the corners consistently
        corners = _order_points(corners.reshape(4, 2))
        
        # Apply perspective transform
        document = _four_point_transform(image, corners)
        
        return document, corners.reshape(-1, 1, 2).astype(np.int32)
    
    return None, None

def enhance_document(image, enhancement_mode="auto"):
    """
    Enhance document image for better readability and OCR
    
    Args:
        image: Input document image
        enhancement_mode: Enhancement mode (auto, text, photo, bw, color)
        
    Returns:
        Enhanced document image
    """
    if image is None:
        return None
    
    # Try to use ML-based enhancer if available
    enhancer = get_image_enhancer()
    
    if enhancer is not None:
        # Use the ML-based enhancer
        doc_type = estimate_document_type(image)
        
        if enhancement_mode == "bw":
            return _enhance_black_white(image)
        elif enhancement_mode == "text":
            return enhancer.enhance_text(image)
        elif enhancement_mode == "color":
            # For color enhancement, we want to preserve color
            if len(image.shape) == 2:
                # Convert grayscale to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            return enhancer.enhance_id_card(image)  # Uses color preservation
        else:  # auto
            return enhancer.enhance(image, doc_type)
    else:
        # Fall back to basic enhancement without ML
        if enhancement_mode == "bw":
            return _enhance_black_white(image)
        elif enhancement_mode == "text":
            return _enhance_text_document(image)
        elif enhancement_mode == "color":
            return _enhance_color_document(image)
        else:  # auto
            # Determine document properties
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Check if document has dark background
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            dark_pixels = np.sum(hist[:128])
            light_pixels = np.sum(hist[128:])
            has_dark_background = dark_pixels > light_pixels
            
            if has_dark_background:
                return _invert_document(image)
            else:
                return _enhance_text_document(image)

def estimate_document_type(image):
    """
    Estimate document type using ML classifier
    
    Args:
        image: Input document image
        
    Returns:
        str: Document type (text_document, id_card, etc.)
    """
    # Try to use ML-based classifier if available
    classifier = get_document_classifier()
    
    if classifier is not None:
        # Use the ML-based classifier
        doc_type, confidence = classifier.predict(image)
        return doc_type
    else:
        # Fall back to basic rule-based classification
        return _rule_based_classification(image)

def _order_points(pts):
    """Order points in consistent order: tl, tr, br, bl"""
    # Create array for ordered points
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left point has smallest sum
    # Bottom-right point has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right point has smallest difference
    # Bottom-left point has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def _four_point_transform(image, pts):
    """Apply perspective transform to get top-down view"""
    # Order the points
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate width of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Calculate height of new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Destination points for transform (top-down view)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    return warped

def _enhance_black_white(image):
    """Basic black and white enhancement"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to BGR if input was BGR
    if len(image.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        return binary

def _enhance_text_document(image):
    """Basic text document enhancement"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply bilateral filter (preserves edges better)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Convert back to BGR if input was BGR
    if len(image.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        return binary

def _enhance_color_document(image):
    """Basic color document enhancement"""
    # Must be color image
    if len(image.shape) == 2:
        # Convert grayscale to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert to HSV for better color processing
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Enhance the brightness and contrast of value channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Merge back and convert to BGR
    hsv = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return enhanced

def _invert_document(image):
    """Handle inverted document (light text on dark background)"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Invert the image
    inverted = cv2.bitwise_not(gray)
    
    # Apply thresholding to make text more readable
    _, binary = cv2.threshold(
        inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Convert back to BGR if input was BGR
    if len(image.shape) == 3:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        return binary

def _rule_based_classification(image):
    """Simple rule-based document classification"""
    # Get image dimensions
    h, w = image.shape[:2]
    aspect_ratio = w / h
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate edge density (indicates text vs images)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # Calculate brightness and standard deviation
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 255.0
    
    # Apply simple rules
    if aspect_ratio < 0.8:
        # Tall and narrow - likely a receipt
        return "receipt"
    elif 0.8 <= aspect_ratio <= 1.2:
        # Nearly square
        if edge_density < 0.1:
            return "id_card"
        else:
            return "business_card"
    else:
        # Wide format
        if edge_density > 0.15:
            return "form"
        else:
            return "text_document"