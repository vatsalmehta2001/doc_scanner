import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Optional, Tuple
import re
import os

class TextProcessor:
    def __init__(self):
        self.lang = 'eng'  # Default language
        self.config = r'--oem 3 --psm 3 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%&*()+-=:;/ \'"'
        
    def extract_text(self, image: np.ndarray) -> Dict:
        """
        Extract text and structured information from document image
        
        Args:
            image: Input document image
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Print debug info
            print("\nStarting text extraction...")
            print(f"Image shape: {image.shape}")
            print(f"Image type: {image.dtype}")
            
            # Prepare image for OCR
            prepared = self._prepare_for_ocr(image)
            
            # Try multiple PSM modes for better results
            psm_modes = [3, 6, 4]  # Auto, Uniform block, Single column
            best_text = ""
            best_confidence = 0
            best_data = None
            
            for psm in psm_modes:
                try:
                    config = f'--oem 3 --psm {psm} {self.config}'
                    print(f"\nTrying PSM mode {psm}...")
                    
                    # Get detailed OCR data
                    ocr_data = pytesseract.image_to_data(
                        prepared,
                        lang=self.lang,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Calculate confidence for this attempt
                    conf = self._calculate_confidence(ocr_data)
                    text = ' '.join([word for word in ocr_data['text'] if word.strip()])
                    
                    print(f"Confidence: {conf:.2f}%")
                    print(f"Characters found: {len(text)}")
                    
                    if conf > best_confidence and text.strip():
                        best_confidence = conf
                        best_text = text
                        best_data = ocr_data
                        
                except Exception as e:
                    print(f"Error in PSM mode {psm}: {str(e)}")
                    continue
            
            if not best_text:
                print("No text was successfully extracted.")
                return {
                    'full_text': '',
                    'text_blocks': [],
                    'structured_data': self._get_empty_structured_data(),
                    'confidence': 0
                }
            
            # Extract structured information from best result
            structured_data = self._extract_structured_info(best_data)
            
            # Get text with layout preservation
            text_blocks = self._extract_text_blocks(best_data)
            
            result = {
                'full_text': best_text,
                'text_blocks': text_blocks,
                'structured_data': structured_data,
                'confidence': best_confidence
            }
            
            # Print extraction summary
            print("\nExtraction Summary:")
            print(f"Final confidence: {best_confidence:.2f}%")
            print(f"Text blocks found: {len(text_blocks)}")
            print(f"Structured data items found: {sum(len(v) for v in structured_data.values())}")
            
            return result
            
        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
            return {
                'full_text': '',
                'text_blocks': [],
                'structured_data': self._get_empty_structured_data(),
                'confidence': 0
            }
    
    def _prepare_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for optimal OCR performance"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Increase contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # Remove noise
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # Ensure black text on white background
            if np.mean(denoised) < 127:
                denoised = cv2.bitwise_not(denoised)
            
            # Scale up image if too small
            min_height = 1000
            scale = max(1, min_height / denoised.shape[0])
            if scale > 1:
                denoised = cv2.resize(
                    denoised, 
                    None, 
                    fx=scale, 
                    fy=scale, 
                    interpolation=cv2.INTER_CUBIC
                )
            
            return denoised
            
        except Exception as e:
            print(f"Error in image preparation: {str(e)}")
            return image
    
    def _get_empty_structured_data(self) -> Dict:
        """Get empty structured data dictionary"""
        return {
            'dates': [],
            'emails': [],
            'phones': [],
            'addresses': [],
            'amounts': []
        }
    
    def _extract_structured_info(self, ocr_data: Dict) -> Dict:
        """Extract structured information from OCR data"""
        try:
            structured_info = self._get_empty_structured_data()
            
            text = ' '.join(ocr_data['text'])
            
            # Extract dates (various formats)
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD/MM/YYYY
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',    # YYYY/MM/DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
            ]
            for pattern in date_patterns:
                structured_info['dates'].extend(re.findall(pattern, text))
            
            # Extract emails
            structured_info['emails'] = re.findall(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                text
            )
            
            # Extract phone numbers
            structured_info['phones'] = re.findall(
                r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
                text
            )
            
            # Extract monetary amounts
            structured_info['amounts'] = re.findall(
                r'\$\s*\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP)',
                text
            )
            
            # Extract addresses (basic)
            address_pattern = r'\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b'
            structured_info['addresses'] = re.findall(address_pattern, text, re.IGNORECASE)
            
            return structured_info
            
        except Exception as e:
            print(f"Error in structured info extraction: {str(e)}")
            return self._get_empty_structured_data()
    
    def _extract_text_blocks(self, ocr_data: Dict) -> List[str]:
        """Extract text while preserving layout structure"""
        try:
            blocks = []
            current_block = []
            last_top = None
            margin = 5  # Pixels margin for line grouping
            
            for i in range(len(ocr_data['text'])):
                if ocr_data['conf'][i] < 0:  # Skip low confidence or empty
                    continue
                    
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                    
                top = ocr_data['top'][i]
                
                # Check if new line based on vertical position
                if last_top is not None and abs(top - last_top) > margin:
                    if current_block:
                        blocks.append(' '.join(current_block))
                        current_block = []
                
                current_block.append(text)
                last_top = top
            
            # Add last block
            if current_block:
                blocks.append(' '.join(current_block))
            
            return blocks
            
        except Exception as e:
            print(f"Error in text block extraction: {str(e)}")
            return []
    
    def _calculate_confidence(self, ocr_data: Dict) -> float:
        """Calculate overall OCR confidence score"""
        try:
            confidences = [conf for conf in ocr_data['conf'] if conf > 0]
            return sum(confidences) / len(confidences) if confidences else 0.0
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def set_language(self, lang: str):
        """Set OCR language"""
        self.lang = lang
        
    def get_available_languages(self) -> List[str]:
        """Get list of available OCR languages"""
        try:
            return pytesseract.get_languages()
        except Exception as e:
            print(f"Error getting languages: {str(e)}")
            return ['eng'] 