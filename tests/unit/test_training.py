#!/usr/bin/env python3
"""
Test document classifier training
"""

import os
import shutil
import cv2
import numpy as np
from models.document_classifier import DocumentClassifier

def test_training():
    """Test the document classifier training process"""
    print("Testing document classifier training...")
    
    # Create temporary training data
    temp_dir = "temp_train_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set up class directories
    classes = ["text_document", "receipt", "id_card"]
    for cls in classes:
        os.makedirs(os.path.join(temp_dir, cls), exist_ok=True)
    
    # Create some synthetic training images
    
    # Text document
    for i in range(3):
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Text Document", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(img, (50, 70), (550, 700), (0, 0, 0), 1)
        
        # Add some text-like lines
        for j in range(10):
            y = 100 + j * 40
            width = np.random.randint(200, 500)
            cv2.line(img, (50, y), (50 + width, y), (0, 0, 0), 2)
            
        file_path = os.path.join(temp_dir, "text_document", f"text_doc_{i}.jpg")
        cv2.imwrite(file_path, img)
    
    # Receipt
    for i in range(3):
        img = np.ones((1000, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Receipt", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(img, (50, 70), (350, 900), (0, 0, 0), 1)
        
        # Add some items
        for j in range(5):
            y = 100 + j * 40
            cv2.putText(img, f"Item {j+1}", (70, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, f"${np.random.randint(5, 50)}.{np.random.randint(0, 99):02d}", 
                       (250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        file_path = os.path.join(temp_dir, "receipt", f"receipt_{i}.jpg")
        cv2.imwrite(file_path, img)
    
    # ID card
    for i in range(3):
        img = np.ones((250, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "ID Card", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(img, (20, 20), (380, 230), (0, 0, 0), 1)
        
        # Add a photo area
        cv2.rectangle(img, (30, 70), (130, 170), (200, 200, 200), -1)
        
        file_path = os.path.join(temp_dir, "id_card", f"id_card_{i}.jpg")
        cv2.imwrite(file_path, img)
    
    # Load images for training
    images = []
    labels = []
    
    for cls in classes:
        cls_dir = os.path.join(temp_dir, cls)
        for file in os.listdir(cls_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(cls_dir, file)
                img = cv2.imread(file_path)
                if img is not None:
                    images.append(img)
                    labels.append(cls)
    
    print(f"Loaded {len(images)} training images")
    
    # Train the classifier
    classifier = DocumentClassifier()
    
    try:
        accuracy = classifier.train(images, labels)
        print(f"Training accuracy: {accuracy:.2f}")
        
        # Save the model
        model_path = "models/test_classifier.model"
        classifier.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        # Test the model on the training data
        correct = 0
        for img, true_label in zip(images, labels):
            pred_label, confidence = classifier.predict(img)
            if pred_label == true_label:
                correct += 1
                
        print(f"Accuracy on training data: {correct/len(images):.2f}")
        
        # Test feature extraction
        print("\nFeature extraction test:")
        for cls in classes:
            cls_dir = os.path.join(temp_dir, cls)
            files = os.listdir(cls_dir)
            if files:
                test_file = os.path.join(cls_dir, files[0])
                test_img = cv2.imread(test_file)
                
                features = classifier.extract_features(test_img).flatten()
                print(f"  {cls} features:")
                print(f"    - Aspect ratio: {features[0]:.2f}")
                print(f"    - Edge density: {features[1]:.4f}")
                print(f"    - Contour density: {features[2]:.4f}")
        
        return True
    
    except Exception as e:
        print(f"Training error: {e}")
        return False
    
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_training()