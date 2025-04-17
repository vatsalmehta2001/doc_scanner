"""
Utility functions for the document scanner application
"""

import os
import sys
import platform
import subprocess
import cv2
import numpy as np
import time
from typing import Optional, Tuple


def ensure_directory(directory):
    """
    Create the specified directory if it doesn't exist
    
    Args:
        directory: Path to the directory to create
    """
    os.makedirs(directory, exist_ok=True)


def get_system_info():
    """
    Get information about the current system
    
    Returns:
        dict: System information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "system": platform.system()
    }
    
    # Check if running on Apple Silicon
    if info["architecture"] == "arm64" and info["system"] == "Darwin":
        info["is_apple_silicon"] = True
    else:
        info["is_apple_silicon"] = False
    
    return info


def check_opencv_version():
    """
    Check OpenCV version and compatibility
    
    Returns:
        tuple: (version_string, is_compatible)
    """
    try:
        import cv2
        version = cv2.__version__
        
        # Check if version is at least 4.5 (good for Apple Silicon)
        major, minor, _ = version.split(".", 2)
        is_compatible = (int(major) > 4 or (int(major) == 4 and int(minor) >= 5))
        
        return version, is_compatible
    except ImportError:
        return "Not installed", False
    except Exception:
        return "Unknown", False


def display_apple_silicon_tips():
    """Display tips for Apple Silicon users"""
    print("Running on Apple Silicon Mac. For best performance:")
    print("1. Use native ARM64 Python and packages")
    print("2. Ensure OpenCV is compiled for ARM64")


def display_macos_camera_permission_help():
    """Display help for macOS camera permissions"""
    print("Camera access is required for document scanning.")
    print("Please grant camera permission in System Preferences > Security & Privacy > Camera")


class Timer:
    def __init__(self):
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed


class ImageEnhancer:
    def __init__(self):
        self.denoise_strength = 10
        self.sharpness = 1.5
        
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement"""
        # Denoise
        denoised = self._apply_denoising(image)
        
        # Color correction
        color_corrected = self._apply_color_correction(denoised)
        
        # Contrast enhancement
        contrast_enhanced = self._enhance_contrast(color_corrected)
        
        # Sharpen
        sharpened = self._sharpen_image(contrast_enhanced)
        
        return sharpened
    
    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply sophisticated denoising"""
        # Use FastNL Means Denoising
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=self.denoise_strength,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    def _apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic color correction"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Auto white balance
        a = self._normalize_channel(a)
        b = self._normalize_channel(b)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize color channel"""
        mean = np.mean(channel)
        std = np.std(channel)
        channel = (channel - mean) * (1 + std/128) + mean
        return np.clip(channel, 0, 255).astype(np.uint8)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        # Convert to YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        
        # Apply adaptive histogram equalization to Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        
        # Merge channels
        yuv = cv2.merge([y, u, v])
        
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply intelligent sharpening"""
        # Create sharpening kernel
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]]) / (self.sharpness * 5)
        
        # Apply kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend with original to prevent over-sharpening
        return cv2.addWeighted(image, 0.55, sharpened, 0.45, 0)