#!/usr/bin/env python3
"""
High-Performance Document Scanner and Analyzer

A professional-grade document scanner that:
1. Uses a multi-threaded architecture for smooth UI
2. Processes documents with ML in a separate thread
3. Provides a responsive, almost-live experience
4. Supports multiple enhancement modes and document types
"""

import cv2
import numpy as np
import os
import time
import queue
import threading
import argparse
from datetime import datetime
from pathlib import Path

# Import ML components (with error handling)
try:
    from models.document_classifier import DocumentClassifier
    from models.image_enhancer import ImageEnhancer
    from models.text_analyzer import TextAnalyzer
    ML_AVAILABLE = True
except ImportError:
    print("Warning: ML components not found. Running in basic mode.")
    ML_AVAILABLE = False

# Try to import OCR components
try:
    import pytesseract
    import pyperclip
    OCR_AVAILABLE = True
except ImportError:
    print("Warning: OCR components not available. Text extraction disabled.")
    OCR_AVAILABLE = False


class DocumentScannerApp:
    """High-performance document scanner application with ML capabilities"""
    
    def __init__(self, model_path=None, camera_id=0, resolution=(1280, 720)):
        """Initialize the scanner application"""
        self.camera_id = camera_id
        self.resolution = resolution
        self.model_path = model_path
        
        # Status variables
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # UI and control variables
        self.detection_sensitivity = 0.7
        self.enhancement_mode = "auto"
        self.manual_mode = False
        self.debug_mode = False
        self.status_message = ""
        self.message_expiry = 0
        
        # Processing variables
        self.current_frame = None
        self.current_document = None
        self.current_corners = None
        self.document_detected = False
        self.last_scan_time = 0
        
        # Create output directory
        self.output_dir = "scans"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # UI settings
        self.window_name = "Smart Document Scanner"
        
        # Initialize ML components if available
        if ML_AVAILABLE:
            print("Initializing ML components...")
            self.classifier = DocumentClassifier(model_path)
            self.enhancer = ImageEnhancer()
            self.analyzer = TextAnalyzer()
        
        # Set up processing queues
        self.frame_queue = queue.Queue(maxsize=1)
        self.document_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Create processing threads
        self.detection_thread = None
        self.processing_thread = None
    
    def start_camera(self):
        """Initialize and start the camera"""
        print(f"Initializing camera {self.camera_id}...")
        self.camera = cv2.VideoCapture(self.camera_id)
        
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Check if camera opened successfully
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Allow camera to initialize
        time.sleep(0.5)
        
        # Get actual camera resolution
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"Camera initialized at {actual_width}x{actual_height}")
        return True
    
    def start_processing_threads(self):
        """Start the background processing threads"""
        self.running = True
        
        # Create and start detection thread
        self.detection_thread = threading.Thread(
            target=self.detection_worker,
            name="DocumentDetectionThread"
        )
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # Create and start ML processing thread
        if ML_AVAILABLE:
            self.processing_thread = threading.Thread(
                target=self.ml_processing_worker,
                name="MLProcessingThread"
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
        
        print("Processing threads started")
    
    def detection_worker(self):
        """Worker thread for document detection"""
        print("Document detection thread started")
        while self.running:
            try:
                # Get latest frame from queue (non-blocking)
                try:
                    frame = self.frame_queue.get(block=False)
                    self.frame_queue.task_done()
                except queue.Empty:
                    # No new frame, sleep briefly
                    time.sleep(0.01)
                    continue
                
                # Skip detection in manual mode
                if self.manual_mode:
                    continue
                
                # Detect document with optimized method
                document, corners = self.detect_document_optimized(frame)
                
                # Update current document info (thread-safe)
                if corners is not None:
                    self.current_document = document
                    self.current_corners = corners
                    self.document_detected = True
            
            except Exception as e:
                print(f"Error in detection thread: {e}")
                time.sleep(0.1)  # Prevent tight error loop
    
    def ml_processing_worker(self):
        """Worker thread for ML processing"""
        print("ML processing thread started")
        while self.running:
            try:
                # Get document from queue (blocking with timeout)
                try:
                    task = self.document_queue.get(timeout=0.5)
                    document, timestamp = task
                except queue.Empty:
                    continue
                
                # Start processing
                start_time = time.time()
                
                # Process the document with ML pipeline
                results = self.process_document_ml(document, self.enhancement_mode)
                results['timestamp'] = timestamp
                results['processing_time'] = time.time() - start_time
                
                # Put results in result queue
                self.result_queue.put(results)
                
                # Mark task as done
                self.document_queue.task_done()
                
            except Exception as e:
                print(f"Error in ML processing thread: {e}")
                time.sleep(0.1)  # Prevent tight error loop
    
    def detect_document_optimized(self, image, small_scale=0.5):
        """
        Optimized document detection for better performance
        
        Args:
            image: Input image
            small_scale: Scale factor for detection (smaller = faster)
            
        Returns:
            tuple: (document, corners)
        """
        # Work with smaller image for speed
        small_image = cv2.resize(image, None, fx=small_scale, fy=small_scale)
        h, w = small_image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        
        # Apply mild blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use both edge detection and thresholding for better results
        # Dynamic threshold value based on image brightness
        mean_value = np.mean(blurred)
        threshold_value = max(0, min(255, mean_value * 0.7))
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Edge detection (simplified)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Combine methods
        combined = cv2.bitwise_or(adaptive, edges)
        
        # Dilate to close gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(combined, kernel, iterations=1)
        
        # Show debug images if enabled
        if self.debug_mode:
            # Create debug mosaic
            debug_images = np.hstack([
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            ])
            cv2.imshow("Detection Steps", debug_images)
        
        # Find contours (optimized method)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Initialize document and corners
        document = None
        corners = None
        
        # Minimum contour area based on sensitivity
        min_area = (w * h) * (0.05 * (1.0 - self.detection_sensitivity * 0.5))
        
        # Check only the largest contours for performance
        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if it's approximately a quadrilateral
            if len(approx) == 4:
                # Scale corners back to original image size
                corners = (approx / small_scale).astype(np.int32)
                
                # Get the document via perspective transform
                try:
                    document = self.four_point_transform(image, corners.reshape(4, 2))
                    
                    # If document is too small or malformed, reject it
                    if document.shape[0] < 100 or document.shape[1] < 100:
                        continue
                    
                    break
                except:
                    # If transform fails, continue to next contour
                    continue
        
        return document, corners
    
    def four_point_transform(self, image, pts):
        """Apply perspective transform to get a top-down view"""
        # Order points clockwise starting from top-left
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute width of new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Compute height of new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Set destination points for transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply transform
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def order_points(self, pts):
        """Order points clockwise starting from top-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left: smallest sum of coordinates
        # Bottom-right: largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right: smallest difference
        # Bottom-left: largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def display_extracted_text(self, text, doc_type):
        """Display extracted text in a separate window"""
        if not text or not text.strip():
            return
        
        # Create a blank image for the text display
        # White background with black text
        lines = text.split('\n')
        line_height = 30
        padding = 20
        width = 800
        height = max(400, (len(lines) + 2) * line_height + 2 * padding)
        
        text_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add a title
        title = f"Extracted Text - {doc_type.capitalize()}"
        cv2.putText(text_image, title, 
                   (padding, padding + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add a separator line
        cv2.line(text_image, 
                (padding, padding + line_height * 2 - 10), 
                (width - padding, padding + line_height * 2 - 10), 
                (0, 0, 0), 1)
        
        # Add the text lines
        y_pos = padding + line_height * 3
        for line in lines:
            if line.strip():  # Skip empty lines
                cv2.putText(text_image, line, 
                          (padding, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += line_height
        
        # Show the window
        window_name = "Extracted Text"
        cv2.imshow(window_name, text_image)
        
        # Also save this text visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_path = os.path.join(self.output_dir, f"text_preview_{timestamp}.jpg")
        cv2.imwrite(vis_path, text_image)
        
        print(f"Text preview saved to: {vis_path}")
    
    def process_document_ml(self, document, enhancement_mode="auto"):
        """
        Process document with ML pipeline
        
        Args:
            document: Document image
            enhancement_mode: Enhancement mode to use
        
        Returns:
            dict: Processing results
        """
        results = {"success": False}
        
        if document is None or not ML_AVAILABLE:
            return results
        
        try:
            # 1. Classify document
            doc_type, confidence = self.classifier.predict(document)
            results["doc_type"] = doc_type
            results["confidence"] = confidence
            
            # 2. Enhance document
            enhanced = self.enhancer.enhance(document, 
                                           doc_type if enhancement_mode == "auto" else enhancement_mode)
            results["enhanced_image"] = enhanced
            
            # 3. Extract text if OCR is available
            if OCR_AVAILABLE:
                # Process image for OCR
                if len(enhanced.shape) == 3:
                    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                else:
                    gray = enhanced
                
                # Choose best preprocessing based on document type
                if doc_type in ["text_document", "form"]:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    ocr_image = binary
                else:
                    ocr_image = gray
                
                # Extract text
                text = pytesseract.image_to_string(ocr_image)
                results["text"] = text
                
                # 4. Analyze text
                if text and text.strip():
                    analysis = self.analyzer.analyze(text, doc_type)
                    results["analysis"] = analysis
            
            results["success"] = True
            
        except Exception as e:
            print(f"Error processing document: {e}")
            results["error"] = str(e)
        
        return results
    
    def save_processed_document(self, results):
        """Save processed document and results to files"""
        if not results.get("success", False):
            print("No results to save")
            return None
        
        try:
            # Generate timestamp
            timestamp = results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            
            # Create filenames
            original_path = os.path.join(self.output_dir, f"scan_{timestamp}.jpg")
            enhanced_path = os.path.join(self.output_dir, f"scan_{timestamp}_enhanced.jpg")
            
            # Save original and enhanced images
            if self.current_document is not None:
                cv2.imwrite(original_path, self.current_document)
            
            if "enhanced_image" in results:
                cv2.imwrite(enhanced_path, results["enhanced_image"])
            
            # Save text if available
            if "text" in results and results["text"]:
                text_path = os.path.join(self.output_dir, f"scan_{timestamp}_text.txt")
                with open(text_path, "w") as f:
                    f.write(results["text"])
                print(f"Text saved to: {text_path}")
                
                # Try to copy to clipboard
                if "text" in results:
                    try:
                        pyperclip.copy(results["text"])
                        print("Text copied to clipboard")
                    except:
                        print("Could not copy to clipboard")
            
            # Save analysis if available
            if "analysis" in results:
                try:
                    import json
                    analysis_path = os.path.join(self.output_dir, f"scan_{timestamp}_analysis.json")
                    
                    # Convert to JSON-serializable format
                    serializable = {}
                    for k, v in results["analysis"].items():
                        if isinstance(v, (str, int, float, bool, type(None))):
                            serializable[k] = v
                        elif isinstance(v, list):
                            serializable[k] = [str(i) for i in v]
                        elif isinstance(v, dict):
                            serializable[k] = {str(dk): str(dv) for dk, dv in v.items()}
                        else:
                            serializable[k] = str(v)
                    
                    with open(analysis_path, "w") as f:
                        json.dump(serializable, f, indent=2)
                except Exception as e:
                    print(f"Error saving analysis: {e}")
            
            print(f"Document saved to {original_path}")
            return original_path
            
        except Exception as e:
            print(f"Error saving document: {e}")
            return None
    
    def process_key_press(self, key):
        """Process keyboard input"""
        if key == ord('q'):
            # Quit
            self.running = False
            return True
            
        elif key == ord('s'):
            # Scan document
            if self.manual_mode:
                # In manual mode, capture center of frame
                h, w = self.current_frame.shape[:2]
                margin_x = int(w * 0.15)
                margin_y = int(h * 0.15)
                document = self.current_frame[margin_y:h-margin_y, margin_x:w-margin_x].copy()
                
                # Update status
                self.status_message = "Manual capture - Processing..."
                self.message_expiry = time.time() + 3.0
                
                # Queue document for processing
                self.document_queue.put((document, datetime.now().strftime("%Y%m%d_%H%M%S")))
                
            elif self.current_document is not None:
                # Ensure we don't scan too frequently
                if time.time() - self.last_scan_time < 1.0:
                    return False
                
                self.last_scan_time = time.time()
                self.status_message = "Processing document..."
                self.message_expiry = time.time() + 3.0
                
                # Queue document for processing
                self.document_queue.put((self.current_document, 
                                       datetime.now().strftime("%Y%m%d_%H%M%S")))
            else:
                self.status_message = "No document detected"
                self.message_expiry = time.time() + 2.0
                
        elif key == ord('e'):
            # Cycle enhancement modes
            modes = ["auto", "text", "bw", "color"]
            idx = modes.index(self.enhancement_mode) if self.enhancement_mode in modes else 0
            self.enhancement_mode = modes[(idx + 1) % len(modes)]
            self.status_message = f"Enhancement mode: {self.enhancement_mode}"
            self.message_expiry = time.time() + 2.0
            
        elif key == ord('m'):
            # Toggle manual mode
            self.manual_mode = not self.manual_mode
            self.status_message = f"Manual mode: {'ON' if self.manual_mode else 'OFF'}"
            self.message_expiry = time.time() + 2.0
            
        elif key == ord('+') or key == ord('='):
            # Increase detection sensitivity
            self.detection_sensitivity = min(1.0, self.detection_sensitivity + 0.1)
            self.status_message = f"Sensitivity: {self.detection_sensitivity:.1f}"
            self.message_expiry = time.time() + 2.0
            
        elif key == ord('-'):
            # Decrease detection sensitivity
            self.detection_sensitivity = max(0.1, self.detection_sensitivity - 0.1)
            self.status_message = f"Sensitivity: {self.detection_sensitivity:.1f}"
            self.message_expiry = time.time() + 2.0
            
        elif key == ord('d'):
            # Toggle debug mode
            self.debug_mode = not self.debug_mode
            self.status_message = f"Debug mode: {'ON' if self.debug_mode else 'OFF'}"
            self.message_expiry = time.time() + 2.0
            
            # Close debug windows if turning off debug
            if not self.debug_mode:
                for window in ["Detection Steps"]:
                    try:
                        cv2.destroyWindow(window)
                    except:
                        pass
        
        return False
    
    def run(self):
        """Main application loop"""
        print("\nStarting Document Scanner")
        print("-------------------------")
        print("Press 's' to scan document")
        print("Press 'e' to toggle enhancement mode")
        print("Press 'm' to toggle manual capture mode")
        print("Press '+'/'-' to adjust detection sensitivity")
        print("Press 'd' to toggle debug mode")
        print("Press 'q' to quit")
        
        try:
            # Start camera
            self.start_camera()
            
            # Start processing threads
            self.start_processing_threads()
            
            # Main loop
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Error reading from camera")
                    break
                
                # Store the current frame
                self.current_frame = frame
                
                # Update frame count and FPS
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                
                # Push frame to detection queue (non-blocking)
                try:
                    if self.frame_queue.empty():
                        self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Skip this frame if queue is full
                
                # Check for ML processing results
                try:
                    while not self.result_queue.empty():
                        results = self.result_queue.get_nowait()
                        
                        # Save results
                        self.save_processed_document(results)
                        
                        # Update status message
                        if results.get("success", False):
                            doc_type = results.get("doc_type", "unknown")
                            confidence = results.get("confidence", 0.0)
                            self.status_message = f"Document type: {doc_type} ({confidence:.2f})"
                            
                            if "text" in results:
                                self.status_message += " - Text extracted"
                                
                            proc_time = results.get("processing_time", 0)
                            print(f"Document processed in {proc_time:.2f} seconds")
                            
                            # Display extracted text if available
                            if "text" in results and results["text"]:
                                self.display_extracted_text(results["text"], doc_type)
                                # Also print to console
                                print("\nExtracted Text:")
                                print("--------------")
                                print(results["text"])
                                print("--------------")
                        else:
                            self.status_message = f"Processing failed: {results.get('error', 'Unknown error')}"
                            
                        self.message_expiry = time.time() + 3.0
                        self.result_queue.task_done()
                        
                except queue.Empty:
                    pass
                
                # Create display frame
                display_frame = frame.copy()
                
                # Draw document outline or manual capture area
                if self.manual_mode:
                    # Draw manual capture area
                    h, w = display_frame.shape[:2]
                    margin_x = int(w * 0.15)
                    margin_y = int(h * 0.15)
                    cv2.rectangle(display_frame, 
                                 (margin_x, margin_y), 
                                 (w - margin_x, h - margin_y), 
                                 (0, 255, 255), 2)
                    
                    # Show manual mode message
                    cv2.putText(display_frame, "Manual Mode - Press 's' to capture",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                elif self.current_corners is not None:
                    # Draw document outline
                    cv2.polylines(display_frame, [self.current_corners], True, (0, 255, 0), 2)
                    cv2.putText(display_frame, "Document Detected - Press 's' to scan",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show status message
                if self.status_message and time.time() < self.message_expiry:
                    cv2.putText(display_frame, self.status_message,
                               (10, display_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show current settings
                settings_text = (f"Mode: {self.enhancement_mode} | "
                               f"{'Manual' if self.manual_mode else 'Auto'} | "
                               f"Sens: {self.detection_sensitivity:.1f} | "
                               f"FPS: {self.fps}")
                               
                cv2.putText(display_frame, settings_text,
                           (10, display_frame.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow(self.window_name, display_frame)
                
                # Process key press
                key = cv2.waitKey(1) & 0xFF
                if self.process_key_press(key):
                    break
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Clean up
            self.running = False
            
            # Wait for threads to finish
            if self.detection_thread:
                self.detection_thread.join(timeout=1.0)
                
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
                
            # Release camera and close windows
            if hasattr(self, 'camera'):
                self.camera.release()
                
            cv2.destroyAllWindows()
            print("Scanner closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="High-Performance Document Scanner")
    parser.add_argument("--model", "-m", type=str, default="models/test_classifier.model",
                        help="Path to trained classifier model")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--width", "-W", type=int, default=1280,
                        help="Camera width resolution")
    parser.add_argument("--height", "-H", type=int, default=720,
                        help="Camera height resolution")
    
    args = parser.parse_args()
    
    # Check if model exists
    if args.model and not os.path.exists(args.model):
        print(f"Warning: Model file not found: {args.model}")
        print("Will use rule-based classification instead")
    
    # Create and run scanner
    scanner = DocumentScannerApp(
        model_path=args.model,
        camera_id=args.camera,
        resolution=(args.width, args.height)
    )
    
    scanner.run()


if __name__ == "__main__":
    main()