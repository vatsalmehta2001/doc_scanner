"""
Enhanced document detection and processing module
"""

import cv2
import numpy as np
import math


def detect_document(image):
    """
    Detect a document in the given image with improved edge detection
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        tuple: (perspective_corrected_document, corners)
            - perspective_corrected_document: The document image with perspective correction
            - corners: The coordinates of document corners
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate optimal threshold using Otsu's method
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = high_thresh * 0.5
    
    # Edge detection using Canny with dynamic thresholds
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    # Dilate the edges to close gaps
    kernel = np.ones((5, 5), np.uint8)
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
    min_area_ratio = 0.05  # Minimum 5% of the image
    
    # Loop through contours
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < (image_area * min_area_ratio):
            continue
            
        # Get perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Approximate the contour shape
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If we have a contour with 4 points, we likely have found our document
        if len(approx) == 4:
            # Get the corners
            corners = approx
            
            # Apply perspective transform to get a top-down view
            document = four_point_transform(image, corners.reshape(4, 2))
            break
    
    return document, corners


def enhance_document(image, enhancement_mode="auto"):
    """
    Enhanced document processing with multiple enhancement modes
    
    Args:
        image: Document image
        enhancement_mode: Enhancement mode ("auto", "text", "photo", "bw", "color")
        
    Returns:
        Enhanced image
    """
    if image is None:
        return None
    
    # Auto-detect the best enhancement mode if set to "auto"
    if enhancement_mode == "auto":
        # Calculate average brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Calculate color variance
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1]
        color_variance = np.std(saturation)
        
        # Decide enhancement mode based on image properties
        if color_variance > 50:  # High color variance indicates a colorful image
            enhancement_mode = "color"
        elif brightness < 150:  # Darker images might be photos
            enhancement_mode = "photo"
        else:  # Default to text for light images with low color variance
            enhancement_mode = "text"
    
    # Apply enhancement based on mode
    if enhancement_mode == "text":
        # Optimize for text documents
        return enhance_text_document(image)
    elif enhancement_mode == "photo":
        # Optimize for photo documents
        return enhance_photo_document(image)
    elif enhancement_mode == "bw":
        # Convert to high contrast black and white
        return enhance_bw_document(image)
    elif enhancement_mode == "color":
        # Enhance colors
        return enhance_color_document(image)
    else:
        # Default enhancement
        return image


def enhance_text_document(image):
    """Enhance text documents for better readability"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to improve text clarity
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply mild bilateral filter to smooth while preserving edges
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Create a 3-channel output
    result = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    
    return result


def enhance_photo_document(image):
    """Enhance photo documents for better visual quality"""
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Increase contrast and brightness
    alpha = 1.1  # Contrast control (1.0-3.0)
    beta = 10    # Brightness control (0-100)
    
    adjusted = cv2.convertScaleAbs(filtered, alpha=alpha, beta=beta)
    
    # Apply mild sharpening
    kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)
    
    return sharpened


def enhance_bw_document(image):
    """Convert to high-contrast black and white"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply global thresholding using Otsu's method
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create a 3-channel output
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return result


def enhance_color_document(image):
    """Enhance colors in document"""
    # Convert to HSV to enhance saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Increase saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255).astype(np.uint8)
    
    # Apply mild sharpening
    kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply sharpening
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def four_point_transform(image, pts):
    """
    Apply perspective transform to obtain a top-down view of a document
    
    Args:
        image: Input image
        pts: Four corner points of the document
        
    Returns:
        Transformed (top-down view) image
    """
    # Order the points - top-left, top-right, bottom-right, bottom-left
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
    
    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Return the warped image
    return warped


def order_points(pts):
    """
    Order the 4 points in a consistent order: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: Four corner points of the document
        
    Returns:
        Ordered rectangle coordinates
    """
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


def estimate_document_type(document):
    """
    Estimate the type of document (text, photo, mixed)
    
    Args:
        document: Document image
        
    Returns:
        str: Document type ('text', 'photo', 'mixed')
    """
    if document is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
    
    # Calculate statistical features
    std_dev = np.std(gray)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (document.shape[0] * document.shape[1])
    
    # Calculate color variance
    hsv = cv2.cvtColor(document, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1]
    color_variance = np.std(saturation)
    
    # Simple decision rule
    if edge_density > 0.1 and std_dev < 50:
        return "text"
    elif color_variance > 40 and edge_density < 0.05:
        return "photo"
    else:
        return "mixed"