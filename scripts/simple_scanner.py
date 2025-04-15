#!/usr/bin/env python3
"""
Simple document scanner that runs directly without any imports.
All code is contained in this single file for easy execution.
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime


# Create output directory
os.makedirs("scans", exist_ok=True)


def detect_document(image):
    """Detect a document in the given image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 75, 200)
    
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


def enhance_document(image, mode="text"):
    """Enhance the document image"""
    if image is None:
        return None
        
    if mode == "text":
        # Optimize for text documents
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        result = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        return result
        
    elif mode == "bw":
        # Convert to high-contrast black and white
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
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
    print("\nSimple Document Scanner")
    print("----------------------")
    print("Press 's' to save the current scan")
    print("Press 'e' to change enhancement mode")
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
    enhancement_mode = "text"  # Default mode
    status_message = ""
    last_message_time = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Detect document in the frame
            document, corners = detect_document(frame)
            current_document = document
            current_corners = corners
            
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Draw green rectangle around detected document
            if current_corners is not None:
                cv2.polylines(display_frame, [current_corners], True, (0, 255, 0), 2)
                
                # Add a text indicator
                cv2.putText(
                    display_frame, 
                    "Document Detected - Press 's' to save", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
            
            # Show enhancement mode
            cv2.putText(
                display_frame,
                f"Mode: {enhancement_mode}",
                (display_frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
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
            cv2.imshow("Document Scanner", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # 's' to save current frame
            if key == ord('s'):
                if current_document is not None:
                    # Enhance document
                    enhanced = enhance_document(current_document, enhancement_mode)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"scans/scan_{timestamp}.jpg"
                    
                    # Save the document
                    cv2.imwrite(filename, enhanced)
                    
                    # Update status message
                    status_message = f"Document saved: {os.path.basename(filename)}"
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