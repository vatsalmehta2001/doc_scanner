"""
Tests for the document scanner application
"""

import unittest
import os
import tempfile
import numpy as np
import cv2

from doc_scanner.utils import ensure_directory, get_system_info
from doc_scanner.document import order_points, four_point_transform


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""
    
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


class TestDocument(unittest.TestCase):
    """Test cases for document processing functions"""
    
    def test_order_points(self):
        """Test the order_points function"""
        # Create test points in random order
        pts = np.array([
            [100, 100],  # top-left
            [300, 100],  # top-right
            [300, 300],  # bottom-right
            [100, 300],  # bottom-left
        ], dtype="float32")
        
        # Shuffle the points
        np.random.shuffle(pts)
        
        # Order the points
        rect = order_points(pts)
        
        # Check the ordering
        self.assertEqual(rect[0][0], 100)  # top-left x
        self.assertEqual(rect[0][1], 100)  # top-left y
        self.assertEqual(rect[1][0], 300)  # top-right x
        self.assertEqual(rect[1][1], 100)  # top-right y
        self.assertEqual(rect[2][0], 300)  # bottom-right x
        self.assertEqual(rect[2][1], 300)  # bottom-right y
        self.assertEqual(rect[3][0], 100)  # bottom-left x
        self.assertEqual(rect[3][1], 300)  # bottom-left y
    
    def test_four_point_transform(self):
        """Test the four_point_transform function"""
        # Create a simple test image
        image = np.zeros((400, 400, 3), dtype="uint8")
        cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), -1)
        
        # Define the points
        pts = np.array([
            [100, 100],  # top-left
            [300, 100],  # top-right
            [300, 300],  # bottom-right
            [100, 300],  # bottom-left
        ], dtype="float32")
        
        # Apply the transform
        warped = four_point_transform(image, pts)
        
        # Check the shape
        self.assertEqual(warped.shape[0], 200)  # height
        self.assertEqual(warped.shape[1], 200)  # width


if __name__ == '__main__':
    unittest.main()