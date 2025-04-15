"""
Text Analyzer Module
Uses NLP techniques to extract structured information from document text
Identifies key fields based on document type classification
"""

import re
import datetime
import spacy
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union

# Load spaCy model (if available)
try:
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    nlp = None
    print("Warning: spaCy model not available. Install with: python -m spacy download en_core_web_sm")


class TextAnalyzer:
    """Analyzes text from scanned documents to extract structured information"""
    
    def __init__(self):
        """Initialize the text analyzer"""
        # Regular expressions for various data types
        self.regex_patterns = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone": r'(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}',
            "date": r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4})\b',
            "url": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!$&\'()*+,;=:~]+)*(?:\?[-\w%!$&\'()*+,;=:~/?]+)?',
            "price": r'\$\s*\d+(?:\.\d{2})?|\d+\s*(?:USD|dollars|EUR|euros)',
            "address": r'\d+\s+[A-Za-z0-9\s,.-]+(?:Avenue|Ave|Street|St|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Plaza|Plz|Terrace|Way)\b[A-Za-z0-9\s,.-]+',
            "time": r'\b(?:1[0-2]|0?[1-9]):[0-5][0-9](?:\s?[AP]M)?|\b(?:2[0-3]|[01]?[0-9]):[0-5][0-9](?:\s?hrs)?',
            "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "zip_code": r'\b\d{5}(?:-\d{4})?\b'
        }
        
        # Keywords for different document types
        self.keywords = {
            "receipt": ["total", "subtotal", "tax", "amount", "payment", "change", "cash", 
                       "credit card", "item", "qty", "quantity", "price", "receipt", "store",
                       "transaction", "order", "purchase", "customer", "cashier", "date", "time"],
            "business_card": ["name", "phone", "email", "website", "address", "company", 
                             "title", "position", "cell", "mobile", "tel", "fax", "linkedin"],
            "id_card": ["name", "birth", "issue", "expiration", "identification", "id number", 
                       "gender", "height", "weight", "eyes", "hair", "address", "class", "restrictions"],
            "form": ["form", "name", "address", "date", "signature", "please", "print", 
                    "check", "select", "complete", "sign", "submit", "office", "use", "only"],
            "text_document": ["chapter", "section", "page", "paragraph", "introduction", 
                             "conclusion", "references", "figure", "table", "appendix"]
        }
        
    def analyze(self, text: str, doc_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze document text and extract structured information
        
        Args:
            text: The text extracted from a document
            doc_type: Type of document (if known from classification)
            
        Returns:
            Dictionary with structured information extracted from text
        """
        if not text or text.strip() == "":
            return {"error": "Empty text provided for analysis"}
            
        # Determine document type if not provided
        if doc_type is None:
            doc_type = self._determine_doc_type(text)
            
        # Process text differently based on document type
        if doc_type == "receipt":
            return self._analyze_receipt(text)
        elif doc_type == "business_card":
            return self._analyze_business_card(text)
        elif doc_type == "id_card":
            return self._analyze_id_card(text)
        elif doc_type == "form":
            return self._analyze_form(text)
        else:  # text_document or unknown
            return self._analyze_general_text(text)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy if available
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = defaultdict(list)
        
        # Attempt to use spaCy if available
        if nlp:
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    entities[ent.label_].append(ent.text)
            except Exception as e:
                print(f"Error in spaCy processing: {e}")
                
        # Extract entities using regex patterns
        for entity_type, pattern in self.regex_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities[entity_type].append(match.group())
        
        return dict(entities)
    
    def _determine_doc_type(self, text: str) -> str:
        """
        Determine document type from text content
        
        Args:
            text: Document text
            
        Returns:
            Predicted document type
        """
        # Count keywords for each document type
        scores = {doc_type: 0 for doc_type in self.keywords}
        
        for doc_type, keywords in self.keywords.items():
            for keyword in keywords:
                # Count case-insensitive occurrences
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
                scores[doc_type] += count
        
        # Get document type with highest score
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return "text_document"  # Default
    
    def _analyze_receipt(self, text: str) -> Dict[str, Any]:
        """Analyze receipt-specific information"""
        result = {
            "type": "receipt",
            "merchant": None,
            "date": None,
            "time": None,
            "total": None,
            "subtotal": None,
            "tax": None,
            "payment_method": None,
            "items": [],
            "metadata": {}
        }
        
        # Extract merchant name (usually at the top)
        lines = text.strip().split('\n')
        if lines:
            result["merchant"] = lines[0].strip()
        
        # Extract date and time
        date_matches = re.findall(self.regex_patterns["date"], text)
        if date_matches:
            result["date"] = date_matches[0]
            
        time_matches = re.findall(self.regex_patterns["time"], text)
        if time_matches:
            result["time"] = time_matches[0]
        
        # Extract total amount
        total_patterns = [
            r'total\s*(?:amount)?(?:\s*:|\s*=)?\s*\$?\s*(\d+\.\d{2})',
            r'total\s*(?:amount)?(?:\s*:|\s*=)?\s*(\d+\.\d{2})',
            r'(?:total|amount)(?:\s*:|\s*=)?\s*\$?\s*(\d+\.\d{2})'
        ]
        
        for pattern in total_patterns:
            total_match = re.search(pattern, text, re.IGNORECASE)
            if total_match:
                result["total"] = float(total_match.group(1))
                break
        
        # Extract subtotal
        subtotal_match = re.search(r'subtotal\s*(?::\s*|\s*=\s*)?\$?(\d+\.\d{2})', text, re.IGNORECASE)
        if subtotal_match:
            result["subtotal"] = float(subtotal_match.group(1))
        
        # Extract tax
        tax_match = re.search(r'tax\s*(?::\s*|\s*=\s*)?\$?(\d+\.\d{2})', text, re.IGNORECASE)
        if tax_match:
            result["tax"] = float(tax_match.group(1))
        
        # Extract payment method
        payment_methods = ["cash", "credit", "debit", "visa", "mastercard", "amex", "american express", "discover", "paypal", "check"]
        for method in payment_methods:
            if re.search(r'\b' + re.escape(method) + r'\b', text, re.IGNORECASE):
                result["payment_method"] = method
                break
        
        # Try to extract items (this is complex and might need refinement)
        item_pattern = r'(\d+)\s*(?:x)?\s*(.+?)\s*\$?(\d+\.\d{2})'
        item_matches = re.finditer(item_pattern, text, re.IGNORECASE)
        
        for match in item_matches:
            try:
                quantity = int(match.group(1))
                description = match.group(2).strip()
                price = float(match.group(3))
                
                result["items"].append({
                    "quantity": quantity,
                    "description": description,
                    "price": price,
                    "subtotal": quantity * price
                })
            except (ValueError, IndexError):
                continue
        
        # Add all extracted entities as metadata
        result["metadata"] = self.extract_entities(text)
        
        return result
    
    def _analyze_business_card(self, text: str) -> Dict[str, Any]:
        """Analyze business card information"""
        result = {
            "type": "business_card",
            "name": None,
            "title": None,
            "company": None,
            "phone": None,
            "mobile": None,
            "email": None,
            "website": None,
            "address": None,
            "metadata": {}
        }
        
        # Extract name (usually the most prominent text at the top)
        lines = text.strip().split('\n')
        if lines:
            result["name"] = lines[0].strip()
            
            # Title is often on the second line
            if len(lines) > 1:
                result["title"] = lines[1].strip()
                
            # Company might be on the third line
            if len(lines) > 2:
                result["company"] = lines[2].strip()
        
        # Extract email
        email_matches = re.findall(self.regex_patterns["email"], text)
        if email_matches:
            result["email"] = email_matches[0]
        
        # Extract phone numbers
        phone_matches = re.findall(self.regex_patterns["phone"], text)
        if phone_matches:
            # Try to differentiate between phone and mobile
            for i, phone in enumerate(phone_matches):
                if i == 0:
                    result["phone"] = phone
                elif i == 1:
                    result["mobile"] = phone
        
        # Extract website
        url_matches = re.findall(self.regex_patterns["url"], text)
        if url_matches:
            result["website"] = url_matches[0]
        else:
            # Try simpler website pattern
            website_match = re.search(r'www\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(?:\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+', text)
            if website_match:
                result["website"] = website_match.group(0)
        
        # Extract address
        address_matches = re.findall(self.regex_patterns["address"], text)
        if address_matches:
            result["address"] = address_matches[0]
        
        # Add all extracted entities as metadata
        result["metadata"] = self.extract_entities(text)
        
        return result
    
    def _analyze_id_card(self, text: str) -> Dict[str, Any]:
        """Analyze ID card information"""
        result = {
            "type": "id_card",
            "id_number": None,
            "name": None,
            "date_of_birth": None,
            "issue_date": None,
            "expiration_date": None,
            "address": None,
            "metadata": {}
        }
        
        # Extract ID number
        id_number_match = re.search(r'(?:id|license|number|#)\s*(?::\s*|\s*=\s*|\s+)?([A-Z0-9-]+)', text, re.IGNORECASE)
        if id_number_match:
            result["id_number"] = id_number_match.group(1)
        
        # Extract name
        name_match = re.search(r'name\s*(?::\s*|\s*=\s*)?(.*?)(?:\n|$)', text, re.IGNORECASE)
        if name_match:
            result["name"] = name_match.group(1).strip()
        
        # Extract dates
        # Look for date of birth
        dob_match = re.search(r'(?:date of birth|birth date|dob|born)\s*(?::\s*|\s*=\s*)?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
        if dob_match:
            result["date_of_birth"] = dob_match.group(1)
        
        # Look for issue date
        issue_match = re.search(r'(?:issue|issued)\s*(?:date|on)?\s*(?::\s*|\s*=\s*)?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
        if issue_match:
            result["issue_date"] = issue_match.group(1)
        
        # Look for expiration date
        expiration_match = re.search(r'(?:expiration|expires|exp)\s*(?:date|on)?\s*(?::\s*|\s*=\s*)?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
        if expiration_match:
            result["expiration_date"] = expiration_match.group(1)
        
        # Extract address
        address_matches = re.findall(self.regex_patterns["address"], text)
        if address_matches:
            result["address"] = address_matches[0]
        
        # Add all extracted entities as metadata
        result["metadata"] = self.extract_entities(text)
        
        return result
    
    def _analyze_form(self, text: str) -> Dict[str, Any]:
        """Analyze form information"""
        result = {
            "type": "form",
            "title": None,
            "fields": {},
            "signature": False,
            "metadata": {}
        }
        
        # Extract form title (usually at the top)
        lines = text.strip().split('\n')
        if lines:
            result["title"] = lines[0].strip()
        
        # Try to extract field-value pairs
        field_pattern = r'([\w\s]+)(?:\:|\s*\=)\s*([^\n]+)'
        field_matches = re.finditer(field_pattern, text)
        
        for match in field_matches:
            field_name = match.group(1).strip().lower()
            field_value = match.group(2).strip()
            
            # Skip if field name or value is too long (likely a false positive)
            if len(field_name) > 30 or len(field_value) > 100:
                continue
                
            result["fields"][field_name] = field_value
        
        # Check if the form appears to have a signature
        signature_keywords = ["signature", "sign here", "signed", "authorized signature"]
        for keyword in signature_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                result["signature"] = True
                break
        
        # Add all extracted entities as metadata
        result["metadata"] = self.extract_entities(text)
        
        return result
    
    def _analyze_general_text(self, text: str) -> Dict[str, Any]:
        """Analyze general text document"""
        lines = text.strip().split('\n')
        
        result = {
            "type": "text_document",
            "title": lines[0].strip() if lines else None,
            "length": {
                "characters": len(text),
                "words": len(text.split()),
                "lines": len(lines)
            },
            "language": "en",  # Default to English
            "summary": None,
            "metadata": {}
        }
        
        # Generate a simple summary (first line or first 100 characters)
        if result["title"] and len(lines) > 1:
            # Use the second line if available, otherwise use beginning of text
            result["summary"] = lines[1].strip() if len(lines) > 1 else text[:100].strip()
        else:
            result["summary"] = text[:100].strip()
        
        # Add all extracted entities as metadata
        result["metadata"] = self.extract_entities(text)
        
        return result
    
    def get_key_value_pairs(self, text: str) -> Dict[str, str]:
        """
        Extract key-value pairs from text
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of key-value pairs
        """
        pairs = {}
        
        # Try to extract field-value pairs with different patterns
        patterns = [
            r'([\w\s]+?):\s*([^\n]+)',  # Key: Value
            r'([\w\s]+?)\s*=\s*([^\n]+)',  # Key = Value
            r'([\w\s]+?)\s{2,}([^\n]+)'  # Key    Value (multiple spaces)
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                key = match.group(1).strip().lower()
                value = match.group(2).strip()
                
                # Skip if key or value is too long (likely a false positive)
                if len(key) > 30 or len(value) > 100:
                    continue
                    
                pairs[key] = value
        
        return pairs
    
    def redact_sensitive_information(self, text: str) -> str:
        """
        Redact sensitive information from text
        
        Args:
            text: Document text
            
        Returns:
            Redacted text
        """
        redacted_text = text
        
        # Redact credit card numbers
        redacted_text = re.sub(self.regex_patterns["credit_card"], "[REDACTED CARD]", redacted_text)
        
        # Redact SSNs
        redacted_text = re.sub(self.regex_patterns["ssn"], "[REDACTED SSN]", redacted_text)
        
        # Redact full phone numbers but keep area code
        phone_matches = re.finditer(self.regex_patterns["phone"], redacted_text)
        for match in reversed(list(phone_matches)):
            phone = match.group(0)
            if len(phone) >= 10:  # Only redact if it's a full phone number
                # Keep area code or first 3 digits
                area_code = phone[:4]  # This may include a '+' or '(' character
                redacted_text = redacted_text[:match.start()] + area_code + "XXX-XXXX" + redacted_text[match.end():]
        
        return redacted_text