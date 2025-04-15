#!/usr/bin/env python3
"""
Document Scanner - Main application module
Uses lite-camera for camera access and OpenCV for image processing
Optimized for Apple Silicon (M3) and macOS Sonoma
"""

import os
import time
import argparse
from datetime import datetime
import cv2
import lite_camera  # Import the lite-camera library

from .document import detect_document
from .utils import ensure_directory

class DocumentScanner:
    """Main document scanner application class"""
    
    def __init__(self, output_dir="scans", camera_id=0, resolution=(1280, 720)):
        """
        Initialize the document scanner
        
        Args:
            output_dir (str): Directory to save scanned documents
            camera_id (int): Camera index (usually 0 for built-in)
            resolution (tuple): Desired camera resolution (width, height)
        """
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.resolution = resolution
        self.camera = None
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
    def initialize_camera(self):
        """Initialize the camera using lite-camera library"""
        try:
            # Initialize camera with lite-camera
            self.camera = lite_camera.Camera(self.camera_id)
            
            # Set the camera resolution
            self.camera.set_resolution(self.resolution[0], self.resolution[1])
            
            # Allow camera to warm up
            time.sleep(0.5)
            
            print(f"Camera initialized successfully: {self.camera.get_device_name()}")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure VS Code has camera permissions")
            print("2. Go to System Settings > Privacy & Security > Camera")
            print("3. Try a different camera_id if multiple cameras are connected")
            return False
    
    def run(self):
        """Run the document scanner application"""
        if not self.initialize_camera():
            return
        
        print("\nDocument Scanner Running")
        print("------------------------")
        print("Press 's' to save the current scan")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Capture frame
                frame = self.camera.read()
                if frame is None:
                    print("Error: Could not read frame from camera")
                    break
                
                # Make a copy for drawing
                display_frame = frame.copy()
                
                # Detect document in the frame
                document, corners = detect_document(frame)
                
                # If document detected, draw green rectangle
                if document is not None and corners is not None:
                    # Draw the outline on the display frame
                    cv2.polylines(display_frame, [corners], True, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow("Document Scanner", display_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                # 's' to save current frame
                if key == ord('s'):
                    if document is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(self.output_dir, f"scan_{timestamp}.jpg")
                        cv2.imwrite(filename, document)
                        print(f"Document saved: {filename}")
                    else:
                        print("No document detected to save")
                
                # 'q' to quit
                elif key == ord('q'):
                    break
                    
        finally:
            # Clean up resources
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            print("Document Scanner closed")

def main():
    """Entry point function when script is run directly"""
    parser = argparse.ArgumentParser(description="Document Scanner Application")
    parser.add_argument("--output", "-o", type=str, default="scans",
                        help="Directory to save scanned documents")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera index (usually 0 for built-in)")
    parser.add_argument("--width", "-W", type=int, default=1280,
                        help="Camera width resolution")
    parser.add_argument("--height", "-H", type=int, default=720,
                        help="Camera height resolution")
    
    args = parser.parse_args()
    
    scanner = DocumentScanner(
        output_dir=args.output,
        camera_id=args.camera,
        resolution=(args.width, args.height)
    )
    
    scanner.run()

if __name__ == "__main__":
    main()