"""
Document detection module
Contains functions for detecting, extracting, and processing documents in images
"""

import cv2
import numpy as np

def detect_document(image):
    """
    Detect a document in the given image
    
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
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Initialize document and corners
    document = None
    corners = None
    
    # Loop through contours
    for contour in contours:
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