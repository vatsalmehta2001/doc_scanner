"""
Document processing module
Handles document detection, enhancement, and classification
"""

import cv2
import numpy as np
import os
from typing import List, Optional, Tuple, Dict
import math

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

class DocumentProcessor:
    def __init__(self):
        self.min_contour_area = 5000  # Minimum contour area to be considered a document
        self.angle_threshold = 1.0     # Maximum angle deviation for alignment
        
    def detect_document_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect document corners in the image using advanced techniques
        
        Args:
            image: Input image
            
        Returns:
            Array of corner points or None if no document detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Find edges
        edges = cv2.Canny(thresh, 50, 200, apertureSize=3)
        
        # Dilate edges to connect components
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None
            
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we have a quadrilateral
        if len(approx) == 4:
            corners = self._order_points(approx.reshape(4, 2))
            return self._refine_corners(corners, edges)
        
        return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting from top-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Top-left will have smallest sum
        # Bottom-right will have largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right will have smallest difference
        # Bottom-left will have largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _refine_corners(self, corners: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Refine corner positions using edge information"""
        refined_corners = corners.copy()
        
        for i, corner in enumerate(corners):
            x, y = int(corner[0]), int(corner[1])
            
            # Define search region
            region_size = 10
            x_start = max(0, x - region_size)
            x_end = min(edges.shape[1], x + region_size)
            y_start = max(0, y - region_size)
            y_end = min(edges.shape[0], y + region_size)
            
            # Extract region around corner
            region = edges[y_start:y_end, x_start:x_end]
            
            # Find strongest edge point
            y_coords, x_coords = np.nonzero(region)
            
            if len(x_coords) > 0:
                # Calculate distances to current corner
                distances = np.sqrt(
                    (x_coords - region_size) ** 2 +
                    (y_coords - region_size) ** 2
                )
                
                # Find closest edge point
                closest_idx = np.argmin(distances)
                refined_corners[i] = [
                    x_start + x_coords[closest_idx],
                    y_start + y_coords[closest_idx]
                ]
        
        return refined_corners
    
    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform to obtain top-down view"""
        rect = pts.astype(np.float32)
        
        # Compute width of new image
        widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Compute height of new image
        heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Construct destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def estimate_document_type(self, image: np.ndarray) -> Dict:
        """
        Estimate document type based on visual features
        
        Args:
            image: Input document image
            
        Returns:
            Dictionary with document type and confidence score
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate features
        aspect_ratio = image.shape[1] / image.shape[0]
        text_density = self._calculate_text_density(gray)
        edge_density = self._calculate_edge_density(gray)
        
        # Simple rule-based classification
        if aspect_ratio > 1.4:  # Wide format
            if text_density > 0.15:
                doc_type = 'letter'
                confidence = 0.8
            else:
                doc_type = 'receipt'
                confidence = 0.7
        else:  # Portrait format
            if edge_density > 0.1:
                if text_density > 0.2:
                    doc_type = 'form'
                    confidence = 0.75
                else:
                    doc_type = 'id_card'
                    confidence = 0.85
            else:
                doc_type = 'photo'
                confidence = 0.9
        
        return {
            'type': doc_type,
            'confidence': confidence,
            'features': {
                'aspect_ratio': aspect_ratio,
                'text_density': text_density,
                'edge_density': edge_density
            }
        }
    
    def _calculate_text_density(self, gray: np.ndarray) -> float:
        """Calculate approximate text density in image"""
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Count potential text pixels
        text_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        
        return text_pixels / total_pixels
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density in image"""
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate density
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        
        return edge_pixels / total_pixels