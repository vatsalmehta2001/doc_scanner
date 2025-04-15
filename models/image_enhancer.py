"""
Image Enhancement Module
Uses ML techniques to improve document image quality before OCR
"""

import cv2
import numpy as np
from skimage import exposure, restoration

class ImageEnhancer:
    """Document image enhancement with ML techniques"""
    
    def __init__(self):
        """Initialize the image enhancer"""
        # Dictionary of enhancement methods for different document types
        self.enhancement_methods = {
            "text_document": self.enhance_text,
            "id_card": self.enhance_id_card,
            "receipt": self.enhance_receipt,
            "form": self.enhance_form,
            "business_card": self.enhance_business_card,
            "default": self.enhance_default
        }
    
    def enhance(self, image, doc_type=None):
        """
        Enhance document image based on document type
        
        Args:
            image: Input document image
            doc_type: Document type (if known)
            
        Returns:
            Enhanced image
        """
        if image is None:
            return None
            
        # Auto-detect document properties
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Check if document has dark background
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        dark_pixels = np.sum(hist[:128])
        light_pixels = np.sum(hist[128:])
        has_dark_background = dark_pixels > light_pixels
        
        # Apply appropriate enhancement method
        if doc_type in self.enhancement_methods:
            return self.enhancement_methods[doc_type](image, has_dark_background)
        else:
            return self.enhancement_methods["default"](image, has_dark_background)
    
    def enhance_text(self, image, has_dark_background=False):
        """
        Enhance text document for better OCR
        
        Args:
            image: Input document image
            has_dark_background: Whether document has dark background
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # If dark background, invert image
        if has_dark_background:
            gray = cv2.bitwise_not(gray)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 5
        )
        
        # Apply morphological operations to clean noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR if input was BGR
        if len(image.shape) == 3:
            return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        else:
            return cleaned
    
    def enhance_id_card(self, image, has_dark_background=False):
        """Enhance ID card images"""
        # Special processing for ID cards
        if len(image.shape) == 3:
            # Preserve color for ID cards
            # Convert to HSV for easier color manipulation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Enhance brightness and contrast of value channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            
            # Merge back and convert to BGR
            hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 5.0
            return cv2.filter2D(enhanced, -1, kernel)
        else:
            # For grayscale, use text document enhancement
            return self.enhance_text(image, has_dark_background)
    
    def enhance_receipt(self, image, has_dark_background=False):
        """Enhance receipt images"""
        # Receipts often have thin text on thermal paper
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Increase contrast
        alpha = 1.5  # Contrast control
        beta = 10    # Brightness control
        contrast_enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Apply Wiener filter for denoising (keeps details better than Gaussian)
        denoised = restoration.wiener(contrast_enhanced, 
                                      psf=np.ones((5, 5)) / 25,
                                      balance=0.3)
        
        # Normalize to uint8
        denoised = (denoised * 255).astype(np.uint8)
        
        # Binarize with Otsu's method
        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Convert back to BGR if input was BGR
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            return binary
    
    def enhance_form(self, image, has_dark_background=False):
        """Enhance form images with fields and boxes"""
        # Forms need special handling to preserve lines and boxes
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply bilateral filter (preserves edges better)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding with larger block size
        binary = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 25, 5
        )
        
        # Use morphology to clean up and preserve lines
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR if input was BGR
        if len(image.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            return result
    
    def enhance_business_card(self, image, has_dark_background=False):
        """Enhance business card images"""
        # Business cards may have colorful backgrounds and logos
        if len(image.shape) == 3:
            # For color images
            # Convert to LAB color space for better color enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance lightness channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back and convert to BGR
            lab = cv2.merge([l, a, b])
            color_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply mild sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 5.0
            return cv2.filter2D(color_enhanced, -1, kernel)
        else:
            # For grayscale
            return self.enhance_text(image, has_dark_background)
    
    def enhance_default(self, image, has_dark_background=False):
        """Default enhancement for unknown document types"""
        if has_dark_background:
            return self.enhance_inverted(image)
        else:
            return self.enhance_text(image, False)
    
    def enhance_inverted(self, image):
        """Enhancement for inverted documents (light text on dark background)"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Invert the image
        inverted = cv2.bitwise_not(gray)
        
        # Apply CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inverted)
        
        # Apply thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Invert back to original color scheme (light on dark)
        result = cv2.bitwise_not(cleaned)
        
        # Convert back to BGR if input was BGR
        if len(image.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            return result