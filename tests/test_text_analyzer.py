#!/usr/bin/env python3
"""
Unit tests for the TextAnalyzer component
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from models.text_analyzer import TextAnalyzer


class TestTextAnalyzer(unittest.TestCase):
    """Test cases for TextAnalyzer component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = TextAnalyzer()
        
        # Sample texts for different document types
        self.receipt_text = """
        GROCERY STORE
        123 Main Street
        City, State 12345
        
        RECEIPT
        
        Date: 04/15/2025
        Time: 14:30
        
        1 x Milk          $3.99
        2 x Bread         $5.98
        1 x Eggs          $2.49
        
        Subtotal:         $12.46
        Tax (8%):         $1.00
        Total:            $13.46
        
        Payment: VISA ****1234
        
        Thank you for shopping with us!
        """
        
        self.business_card_text = """
        John Smith
        Software Engineer
        
        TECH COMPANY INC.
        
        Phone: (555) 123-4567
        Email: john.smith@techcompany.com
        Website: www.techcompany.com
        
        123 Tech Street
        San Francisco, CA 94103
        """
        
        self.id_card_text = """
        DRIVER LICENSE
        
        ID: DL123456789
        
        Name: JANE DOE
        Address: 456 Oak Avenue, Anytown, CA 90210
        DOB: 01/15/1985
        Issue Date: 02/20/2023
        Expiration: 01/15/2028
        
        Class: C
        Sex: F
        Height: 5'6"
        Eyes: BRN
        """
        
        self.form_text = """
        EMPLOYMENT APPLICATION FORM
        
        Name: Sarah Johnson
        Address: 789 Pine Street, Somewhere, NY 10001
        Phone: 555-987-6543
        Email: sarah.j@email.com
        
        Position applied for: Marketing Manager
        
        Education: Bachelor's Degree
        Experience: 5 years
        
        References: Available upon request
        
        Signature: _Sarah Johnson_
        Date: 04/10/2025
        """
    
    def test_determine_doc_type(self):
        """Test document type determination"""
        # Test receipt detection
        doc_type = self.analyzer._determine_doc_type(self.receipt_text)
        self.assertEqual(doc_type, "receipt")
        
        # Test business card detection
        doc_type = self.analyzer._determine_doc_type(self.business_card_text)
        self.assertEqual(doc_type, "business_card")
        
        # Test ID card detection
        doc_type = self.analyzer._determine_doc_type(self.id_card_text)
        self.assertEqual(doc_type, "id_card")
        
        # Test form detection
        doc_type = self.analyzer._determine_doc_type(self.form_text)
        self.assertEqual(doc_type, "form")
    
    def test_analyze_receipt(self):
        """Test receipt analysis"""
        result = self.analyzer._analyze_receipt(self.receipt_text)
        
        # Verify basic fields
        self.assertEqual(result["type"], "receipt")
        self.assertEqual(result["merchant"], "GROCERY STORE")
        self.assertEqual(result["date"], "04/15/2025")
        self.assertEqual(result["time"], "14:30")
        self.assertEqual(result["total"], 13.46)
        self.assertEqual(result["subtotal"], 12.46)
        self.assertEqual(result["tax"], 1.00)
        self.assertEqual(result["payment_method"], "visa")
        
        # Check metadata
        self.assertIn("date", result["metadata"])
        self.assertIn("price", result["metadata"])
    
    def test_analyze_business_card(self):
        """Test business card analysis"""
        result = self.analyzer._analyze_business_card(self.business_card_text)
        
        # Verify basic fields
        self.assertEqual(result["type"], "business_card")
        self.assertEqual(result["name"], "John Smith")
        self.assertEqual(result["title"], "Software Engineer")
        self.assertEqual(result["company"], "TECH COMPANY INC.")
        self.assertEqual(result["phone"], "(555) 123-4567")
        self.assertEqual(result["email"], "john.smith@techcompany.com")
        self.assertEqual(result["website"], "www.techcompany.com")
        
        # Check metadata
        self.assertIn("email", result["metadata"])
        self.assertIn("phone", result["metadata"])
    
    def test_analyze_id_card(self):
        """Test ID card analysis"""
        result = self.analyzer._analyze_id_card(self.id_card_text)
        
        # Verify basic fields
        self.assertEqual(result["type"], "id_card")
        self.assertEqual(result["id_number"], "DL123456789")
        self.assertEqual(result["name"], "JANE DOE")
        self.assertEqual(result["date_of_birth"], "01/15/1985")
        self.assertEqual(result["issue_date"], "02/20/2023")
        self.assertEqual(result["expiration_date"], "01/15/2028")
        
        # Check metadata
        self.assertIn("date", result["metadata"])
    
    def test_analyze_form(self):
        """Test form analysis"""
        result = self.analyzer._analyze_form(self.form_text)
        
        # Verify basic fields
        self.assertEqual(result["type"], "form")
        self.assertEqual(result["title"], "EMPLOYMENT APPLICATION FORM")
        self.assertTrue(result["signature"])
        
        # Check fields
        self.assertIn("name", result["fields"])
        self.assertEqual(result["fields"]["name"], "Sarah Johnson")
        self.assertIn("email", result["fields"])
        self.assertEqual(result["fields"]["email"], "sarah.j@email.com")
        
        # Check metadata
        self.assertIn("email", result["metadata"])
        self.assertIn("phone", result["metadata"])
        self.assertIn("date", result["metadata"])
    
    def test_extract_entities(self):
        """Test entity extraction"""
        # Test email extraction
        text = "Contact me at test@example.com or call 555-123-4567"
        entities = self.analyzer.extract_entities(text)
        
        self.assertIn("email", entities)
        self.assertIn("phone", entities)
        self.assertEqual(entities["email"][0], "test@example.com")
        self.assertEqual(entities["phone"][0], "555-123-4567")
        
        # Test date extraction
        text = "The meeting is on 04/15/2025 at 3:30 PM"
        entities = self.analyzer.extract_entities(text)
        
        self.assertIn("date", entities)
        self.assertIn("time", entities)
        self.assertEqual(entities["date"][0], "04/15/2025")
        self.assertEqual(entities["time"][0], "3:30 PM")
        
        # Test multiple entities
        text = "Send payment of $123.45 to 123 Main St, Springfield, IL 62701"
        entities = self.analyzer.extract_entities(text)
        
        self.assertIn("price", entities)
        self.assertIn("address", entities)
        self.assertTrue(any("123.45" in p for p in entities["price"]))
        self.assertTrue(any("123 Main St" in a for a in entities["address"]))
    
    def test_get_key_value_pairs(self):
        """Test key-value pair extraction"""
        text = """
        Name: John Doe
        Email = john.doe@example.com
        Phone Number: 555-678-9012
        Address    123 Elm Street
        """
        
        pairs = self.analyzer.get_key_value_pairs(text)
        
        self.assertIn("name", pairs)
        self.assertIn("email", pairs)
        self.assertIn("phone number", pairs)
        self.assertIn("address", pairs)
        
        self.assertEqual(pairs["name"], "John Doe")
        self.assertEqual(pairs["email"], "john.doe@example.com")
        self.assertEqual(pairs["phone number"], "555-678-9012")
        self.assertEqual(pairs["address"], "123 Elm Street")
    
    def test_redact_sensitive_information(self):
        """Test sensitive information redaction"""
        text = """
        Credit Card: 4111-1111-1111-1111
        SSN: 123-45-6789
        Phone: (555) 987-6543
        """
        
        redacted = self.analyzer.redact_sensitive_information(text)
        
        # Check credit card redaction
        self.assertNotIn("4111-1111-1111-1111", redacted)
        self.assertIn("[REDACTED CARD]", redacted)
        
        # Check SSN redaction
        self.assertNotIn("123-45-6789", redacted)
        self.assertIn("[REDACTED SSN]", redacted)
        
        # Check phone redaction (area code should be preserved)
        self.assertNotIn("(555) 987-6543", redacted)
        self.assertIn("(555", redacted)
        self.assertIn("XXX-XXXX", redacted)
    
    @patch('spacy.load')
    def test_spacy_integration(self, mock_spacy_load):
        """Test spaCy integration if available"""
        # Mock spaCy
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_ent = MagicMock()
        
        # Setup mock entities
        mock_ent.text = "John Smith"
        mock_ent.label_ = "PERSON"
        mock_doc.ents = [mock_ent]
        
        # Setup nlp return value
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        # Temporarily replace the nlp attribute
        original_nlp = self.analyzer.nlp
        self.analyzer.nlp = mock_nlp
        
        # Test entity extraction with mocked spaCy
        text = "My name is John Smith"
        entities = self.analyzer.extract_entities(text)
        
        # Restore original
        self.analyzer.nlp = original_nlp
        
        # Verify the result
        self.assertIn("PERSON", entities)
        self.assertEqual(entities["PERSON"][0], "John Smith")
    
    def test_analyze_empty_text(self):
        """Test handling of empty text"""
        result = self.analyzer.analyze("")
        self.assertIn("error", result)
        
        result = self.analyzer.analyze(None)
        self.assertIn("error", result)
    
    def test_general_document_analysis(self):
        """Test analysis of general text documents"""
        text = """
        PROJECT PROPOSAL
        
        This document outlines the project scope and timeline.
        We will implement the feature in three phases.
        
        Phase 1: Requirements gathering
        Phase 2: Implementation
        Phase 3: Testing
        
        Budget: $50,000
        Timeline: 3 months
        """
        
        result = self.analyzer._analyze_general_text(text)
        
        # Check basic fields
        self.assertEqual(result["type"], "text_document")
        self.assertEqual(result["title"], "PROJECT PROPOSAL")
        
        # Check length calculations
        self.assertIn("characters", result["length"])
        self.assertIn("words", result["length"])
        self.assertIn("lines", result["length"])
        
        # Check metadata
        self.assertIn("price", result["metadata"])


if __name__ == "__main__":
    unittest.main()