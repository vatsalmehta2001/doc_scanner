"""
Tests for the document scanner application
"""

import unittest
import os
import tempfile
from doc_scanner.utils import ensure_directory, get_system_info

class TestScanner(unittest.TestCase):
    """Test cases for the document scanner application"""
    
    def test_ensure_directory(self):
        """Test the ensure_directory function"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "test_dir")
            
            # Directory should not exist yet
            self.assertFalse(os.path.exists(test_dir))
            
            # Create the directory
            ensure_directory(test_dir)
            
            # Directory should now exist
            self.assertTrue(os.path.exists(test_dir))
            
            # Calling again should not raise an error
            ensure_directory(test_dir)
    
    def test_get_system_info(self):
        """Test the get_system_info function"""
        info = get_system_info()
        
        # Verify that the required keys are present
        self.assertIn("platform", info)
        self.assertIn("python_version", info)
        self.assertIn("architecture", info)
        self.assertIn("is_apple_silicon", info)
        
        # Values should be non-empty
        self.assertTrue(info["platform"])
        self.assertTrue(info["python_version"])
        self.assertTrue(info["architecture"])

if __name__ == '__main__':
    unittest.main()