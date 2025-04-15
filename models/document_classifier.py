"""
Document Classifier Module
Uses machine learning to classify document types from images
"""

import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Define document types
DOCUMENT_TYPES = [
    "text_document",  # Regular text documents/pages
    "id_card",        # ID cards, driver's licenses
    "receipt",        # Shopping receipts
    "form",           # Forms with fields and boxes
    "business_card"   # Business cards
]

class DocumentClassifier:
    """ML-based document type classifier"""
    
    def __init__(self, model_path=None):
        """
        Initialize the document classifier
        
        Args:
            model_path: Path to a saved model file (optional)
        """
        self.model = None
        self.scaler = StandardScaler()
        self.classes = DOCUMENT_TYPES
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            # Initialize a new model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def extract_features(self, image):
        """
        Extract meaningful features from a document image
        
        Args:
            image: Input document image (BGR format)
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Document dimensions and aspect ratio
        height, width = gray.shape
        aspect_ratio = width / height
        
        # Calculate histogram features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten() / (height * width)  # Normalize
        
        # Calculate edge density (indicates text vs images)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Calculate text-like regions
        # Use adaptive thresholding and count contours
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_density = len(contours) / (height * width)
        
        # Calculate texture feature using GLCM (Gray-Level Co-occurrence Matrix)
        texture_feature = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        
        # Find potential text regions
        # Count regions that might be text lines
        horizontal_projection = np.sum(thresh, axis=1) / width
        text_line_feature = np.std(horizontal_projection)
        
        # Combine all features
        features = np.array([
            aspect_ratio,
            edge_density,
            contour_density,
            texture_feature,
            text_line_feature,
            np.mean(gray) / 255.0,  # Brightness
            np.std(gray) / 255.0,   # Contrast
        ])
        
        # Add histogram features
        features = np.concatenate([features, hist])
        
        return features.reshape(1, -1)
    
    def predict(self, image):
        """
        Predict the document type
        
        Args:
            image: Input document image
            
        Returns:
            str: Predicted document type
            float: Confidence score (probability)
        """
        # If model is not trained, fall back to rule-based classification
        if self.model is None or not hasattr(self.model, 'classes_'):
            # Fallback to rule-based classification if no model
            return self._rule_based_classification(image), 0.6
        
        # Extract features
        features = self.extract_features(image)
        
        # Scale features if scaler is fitted
        if hasattr(self.scaler, 'mean_'):
            features = self.scaler.transform(features)
        
        try:
            # Check if the model is trained
            if not hasattr(self.model, 'predict_proba'):
                return self._rule_based_classification(image), 0.6
                
            # Predict class and probability
            probabilities = self.model.predict_proba(features)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            return self.classes[predicted_class_idx], float(confidence)
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback to rule-based classification
            return self._rule_based_classification(image), 0.5
    
    def train(self, images, labels):
        """
        Train the classifier on a dataset of document images
        
        Args:
            images: List of document images
            labels: List of corresponding document type labels
            
        Returns:
            float: Training accuracy
        """
        # Extract features from all images
        features = []
        for img in images:
            features.append(self.extract_features(img).flatten())
        
        features = np.array(features)
        
        # Fit the scaler
        self.scaler.fit(features)
        
        # Scale the features
        scaled_features = self.scaler.transform(features)
        
        # Train the model
        self.model.fit(scaled_features, labels)
        
        # Return training accuracy
        accuracy = self.model.score(scaled_features, labels)
        return accuracy
    
    def save_model(self, model_path):
        """Save the trained model to a file"""
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and scaler
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'classes': self.classes
            }, model_path)
            
            return True
        return False
    
    def _load_model(self, model_path):
        """Load a model from a file"""
        try:
            # Load the saved model
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.classes = data['classes']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _rule_based_classification(self, image):
        """
        Simple rule-based classification when no ML model is available
        
        Args:
            image: Input document image
            
        Returns:
            str: Predicted document type
        """
        # Get image dimensions
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Extract features for rules
        features = self.extract_features(image).flatten()
        edge_density = features[1]
        brightness = features[5]
        
        # Apply simple rules
        if aspect_ratio < 0.8:
            # Tall and narrow
            return "receipt"
        elif 0.8 <= aspect_ratio <= 1.2:
            # Nearly square
            return "id_card" if edge_density < 0.1 else "business_card"
        else:
            # Wide format
            return "form" if edge_density > 0.15 else "text_document"
    
   

    def evaluate_image(self, image, true_label=None):
        """
        Evaluate classifier on a single image with detailed diagnostic info
        
        Args:
            image: Input document image
            true_label: Actual document type (optional)
            
        Returns:
            dict: Evaluation results and diagnostics
        """
        results = {
            "image_shape": image.shape,
            "features": {},
            "prediction": None,
            "confidence": 0.0,
            "correct": None
        }
        
        # Extract features
        feature_vector = self.extract_features(image).flatten()
        
        # Store feature values for diagnostics
        features_info = {
            "aspect_ratio": feature_vector[0],
            "edge_density": feature_vector[1],
            "contour_density": feature_vector[2],
            "texture": feature_vector[3],
            "text_lines": feature_vector[4],
            "brightness": feature_vector[5],
            "contrast": feature_vector[6]
        }
        results["features"] = features_info
        
        # Make prediction
        predicted_label, confidence = self.predict(image)
        results["prediction"] = predicted_label
        results["confidence"] = confidence
        
        # Check if correct (if true label provided)
        if true_label is not None:
            results["correct"] = (predicted_label == true_label)
            results["true_label"] = true_label
        
        return results