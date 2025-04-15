#!/usr/bin/env python3
"""
Quick demo script to show the enhanced document scanner in action.
This is a simplified version for demonstration purposes.
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime


def enhance_document(image, mode="text"):
    """Simple document enhancement for demo"""
    if mode == "bw":
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    elif mode == "text":
        # Optimize for text
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    elif mode == "color":
        # Enhance colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        return image


def detect_document(frame):
    """Basic document detection for demo"""
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Check top contours
    for contour in contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Check if it's a quadrilateral of sufficient size
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:
            return approx
    
    return None


def main():
    # Create output directory
    os.makedirs("scans", exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Enhanced Document Scanner Demo")
    print("------------------------------")
    print("s: Save document")
    print("1-3: Change enhancement mode (1=Normal, 2=Text, 3=B&W)")
    print("q: Quit")
    
    # Initial settings
    enhancement_mode = "normal"
    status_message = ""
    status_time = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create a copy for display
        display = frame.copy()
        
        # Detect document
        document_corners = detect_document(frame)
        
        if document_corners is not None:
            # Draw outline
            cv2.polylines(display, [document_corners], True, (0, 255, 0), 2)
            
            # Add indicator text
            cv2.putText(display, "Document Detected - Press 's' to save", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show status message
        if status_message:
            cv2.putText(display, status_message, (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show current mode
        cv2.putText(display, f"Mode: {enhancement_mode}", 
                   (display.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Show the frame
        cv2.imshow("Document Scanner Demo", display)
        
        # Get key press
        key = cv2.waitKey(1) & 0xFF
        
        # Save document
        if key == ord('s') and document_corners is not None:
            # Create a mask for the document
            mask = np.zeros_like(frame)
            cv2.drawContours(mask, [document_corners], 0, (255, 255, 255), -1)
            
            # Extract document region
            document = frame.copy()
            document[mask == 0] = 0
            
            # Enhance based on mode
            enhanced = enhance_document(document, enhancement_mode)
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scans/demo_scan_{timestamp}.jpg"
            cv2.imwrite(filename, enhanced)
            
            status_message = f"Saved: {filename}"
            print(status_message)
        
        # Change enhancement mode
        elif key == ord('1'):
            enhancement_mode = "normal"
            status_message = "Mode: Normal"
        elif key == ord('2'):
            enhancement_mode = "text"
            status_message = "Mode: Text Enhancement"
        elif key == ord('3'):
            enhancement_mode = "bw"
            status_message = "Mode: Black & White"
        elif key == ord('4'):
            enhancement_mode = "color"
            status_message = "Mode: Color Enhancement"
        
        # Quit
        elif key == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()