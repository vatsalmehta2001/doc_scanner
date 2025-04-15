#!/usr/bin/env python3
"""
Enhanced Document Scanner - Main application module
Uses OpenCV for camera access and image processing
Optimized for Apple Silicon (M3) and macOS Sonoma
"""

import os
import time
import argparse
from datetime import datetime
import cv2
import numpy as np
import threading
import queue
import platform

from .document import (
    detect_document, 
    enhance_document, 
    estimate_document_type
)
from .utils import (
    ensure_directory, 
    display_macos_camera_permission_help, 
    display_apple_silicon_tips
)


class EnhancedDocumentScanner:
    """Enhanced document scanner application class"""
    
    def __init__(self, output_dir="scans", camera_id=0, resolution=(1280, 720), 
                 batch_mode=False, enhancement_mode="auto"):
        """
        Initialize the document scanner
        
        Args:
            output_dir (str): Directory to save scanned documents
            camera_id (int): Camera index (usually 0 for built-in)
            resolution (tuple): Desired camera resolution (width, height)
            batch_mode (bool): Batch mode automatically saves detected documents
            enhancement_mode (str): Enhancement mode for saved documents
        """
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.resolution = resolution
        self.camera = None
        self.batch_mode = batch_mode
        self.enhancement_mode = enhancement_mode
        self.current_document = None
        self.current_corners = None
        self.last_save_time = 0
        self.frame_queue = queue.Queue(maxsize=1)
        self.processing_thread = None
        self.running = False
        self.preview_mode = "normal"  # normal, enhanced, bw
        self.status_message = ""
        self.document_detected = False
        self.auto_save_counter = 0
        
        # Ensure output directory exists
        ensure_directory(output_dir)
        
    def initialize_camera(self):
        """Initialize camera with OpenCV"""
        try:
            # Initialize camera with OpenCV
            self.camera = cv2.VideoCapture(self.camera_id)
            
            # Set the camera resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Check if camera opened successfully
            if not self.camera.isOpened():
                print("Error: Could not open camera")
                display_macos_camera_permission_help()
                return False
                
            # Allow camera to warm up
            time.sleep(0.5)
            
            # Get actual camera resolution (might be different from requested)
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"Camera initialized successfully with index: {self.camera_id}")
            print(f"Resolution: {actual_width}x{actual_height}")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure VS Code has camera permissions")
            print("2. Go to System Settings > Privacy & Security > Camera")
            print("3. Try a different camera_id if multiple cameras are connected")
            display_macos_camera_permission_help()
            return False
    
    def process_frames(self):
        """Process frames in a separate thread"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Detect document
                document, corners = detect_document(frame)
                
                # Update current document and corners
                self.current_document = document
                self.current_corners = corners
                self.document_detected = document is not None
                
                # Auto-save logic for batch mode
                if self.batch_mode and self.document_detected:
                    current_time = time.time()
                    # Only save every 2 seconds to avoid duplicates
                    if current_time - self.last_save_time > 2.0:
                        self.save_document()
                        self.last_save_time = current_time
                
                # Mark task as done
                self.frame_queue.task_done()
            except queue.Empty:
                # Just continue if queue is empty
                pass
            except Exception as e:
                print(f"Error in processing thread: {e}")
    
    def save_document(self):
        """Save the current document"""
        if self.current_document is not None:
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Estimate document type
            doc_type = estimate_document_type(self.current_document)
            
            # Choose enhancement based on document type
            if doc_type == "text":
                enhancement_mode = "text" if self.enhancement_mode == "auto" else self.enhancement_mode
            elif doc_type == "photo":
                enhancement_mode = "photo" if self.enhancement_mode == "auto" else self.enhancement_mode
            else:
                enhancement_mode = self.enhancement_mode
            
            # Enhance document
            enhanced = enhance_document(self.current_document, enhancement_mode)
            if enhanced is None:
                enhanced = self.current_document
            
            # Save the document
            filename = os.path.join(self.output_dir, f"scan_{timestamp}.jpg")
            cv2.imwrite(filename, enhanced)
            
            # Update status message
            self.status_message = f"Document saved: {os.path.basename(filename)} ({doc_type})"
            print(self.status_message)
            
            # Increment counter for batch mode
            self.auto_save_counter += 1
            
            return filename
        else:
            self.status_message = "No document detected to save"
            print(self.status_message)
            return None
    
    def toggle_preview_mode(self):
        """Toggle between preview modes"""
        modes = ["normal", "enhanced", "bw"]
        current_index = modes.index(self.preview_mode)
        self.preview_mode = modes[(current_index + 1) % len(modes)]
        self.status_message = f"Preview mode: {self.preview_mode}"
    
    def run(self):
        """Run the document scanner application"""
        if not self.initialize_camera():
            return
        
        print("\nEnhanced Document Scanner Running")
        print("--------------------------------")
        print("Press 's' to save the current scan")
        print("Press 'e' to toggle enhancement preview mode")
        print("Press 'q' to quit")
        
        if self.batch_mode:
            print("\nBatch Mode Enabled: Automatically saving detected documents")
        
        # Set running flag
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("Error: Could not read frame from camera")
                    break
                
                # Put frame in queue for processing (non-blocking)
                try:
                    # Update the queue with the latest frame (skip if full)
                    if self.frame_queue.empty():
                        self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    # Queue is full, just continue with display
                    pass
                
                # Make a copy for display
                display_frame = frame.copy()
                
                # Draw document outline if detected
                if self.current_corners is not None:
                    # Draw the outline on the display frame
                    cv2.polylines(display_frame, [self.current_corners], True, (0, 255, 0), 2)
                    
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
                
                # Apply preview enhancement if selected
                if self.preview_mode != "normal" and self.current_document is not None:
                    # Create a smaller preview of the enhanced document
                    preview_height = 300
                    doc_aspect = self.current_document.shape[1] / self.current_document.shape[0]
                    preview_width = int(preview_height * doc_aspect)
                    
                    if self.preview_mode == "enhanced":
                        preview = enhance_document(self.current_document, "auto")
                    elif self.preview_mode == "bw":
                        preview = enhance_document(self.current_document, "bw")
                    else:
                        preview = self.current_document
                    
                    # Resize preview
                    preview = cv2.resize(preview, (preview_width, preview_height))
                    
                    # Position in bottom-right corner
                    h, w = display_frame.shape[:2]
                    x_offset = w - preview_width - 10
                    y_offset = h - preview_height - 10
                    
                    # Create a background for the preview
                    cv2.rectangle(
                        display_frame,
                        (x_offset - 5, y_offset - 5),
                        (x_offset + preview_width + 5, y_offset + preview_height + 5),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Overlay the preview
                    display_frame[y_offset:y_offset+preview_height, 
                                  x_offset:x_offset+preview_width] = preview
                
                # Show status message
                if self.status_message:
                    # Display for 3 seconds
                    if time.time() - self.last_save_time < 3.0:
                        cv2.putText(
                            display_frame,
                            self.status_message,
                            (10, display_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )
                
                # Display batch mode counter if enabled
                if self.batch_mode:
                    counter_text = f"Auto-saved: {self.auto_save_counter}"
                    cv2.putText(
                        display_frame,
                        counter_text,
                        (display_frame.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )
                
                # Show the frame
                cv2.imshow("Enhanced Document Scanner", display_frame)
                
                # Check for key presses (1ms wait)
                key = cv2.waitKey(1) & 0xFF
                
                # 's' to save current frame
                if key == ord('s'):
                    self.save_document()
                    self.last_save_time = time.time()
                
                # 'e' to toggle enhancement preview
                elif key == ord('e'):
                    self.toggle_preview_mode()
                
                # 'q' to quit
                elif key == ord('q'):
                    self.running = False
                    break
                    
        finally:
            # Clean up resources
            self.running = False
            
            if self.processing_thread is not None:
                self.processing_thread.join(timeout=1.0)
            
            if self.camera is not None:
                self.camera.release()
                
            cv2.destroyAllWindows()
            print("Document Scanner closed")


def main():
    """Entry point function when script is run directly"""
    parser = argparse.ArgumentParser(description="Enhanced Document Scanner Application")
    parser.add_argument("--output", "-o", type=str, default="scans",
                        help="Directory to save scanned documents")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera index (usually 0 for built-in)")
    parser.add_argument("--width", "-W", type=int, default=1280,
                        help="Camera width resolution")
    parser.add_argument("--height", "-H", type=int, default=720,
                        help="Camera height resolution")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Enable batch mode (auto-save documents)")
    parser.add_argument("--enhance", "-e", type=str, default="auto",
                        choices=["auto", "text", "photo", "bw", "color"],
                        help="Enhancement mode for saved documents")
    
    args = parser.parse_args()
    
    # Show Apple Silicon tips if applicable
    display_apple_silicon_tips()
    
    scanner = EnhancedDocumentScanner(
        output_dir=args.output,
        camera_id=args.camera,
        resolution=(args.width, args.height),
        batch_mode=args.batch,
        enhancement_mode=args.enhance
    )
    
    scanner.run()


if __name__ == "__main__":
    main()