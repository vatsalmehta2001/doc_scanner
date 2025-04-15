"""
Utility functions for the document scanner application
"""

import os
import sys
import platform
import subprocess

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
        "architecture": platform.machine()
    }
    
    # Check if running on Apple Silicon
    if info["architecture"] == "arm64" and "macOS" in info["platform"]:
        info["is_apple_silicon"] = True
    else:
        info["is_apple_silicon"] = False
    
    return info

def check_camera_permission():
    """
    Check if camera permission is granted on macOS
    
    Returns:
        bool: True if camera permission appears to be granted, False otherwise
    """
    if platform.system() != "Darwin":  # Not macOS
        return True
    
    try:
        # Use tccutil to check camera permissions (requires admin)
        result = subprocess.run(
            ["tccutil", "status", "Camera"],
            capture_output=True,
            text=True
        )
        
        if "DENIED" in result.stdout:
            return False
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        # If we can't check, assume it might be an issue
        return None

def display_apple_silicon_tips():
    """Display tips for running on Apple Silicon Macs"""
    info = get_system_info()
    
    if info["is_apple_silicon"]:
        print("\nApple Silicon (M-series) compatibility tips:")
        print("-------------------------------------------")
        print("1. Ensure you're using Python for Apple Silicon (arm64)")
        print(f"   Current Python architecture: {info['architecture']}")
        print("2. Install packages with pip using the '--no-binary' flag if needed")
        print("   Example: pip install --no-binary :all: opencv-python")
        print("3. For OpenCV, the arm64 build is recommended")
        print("4. Make sure lite-camera is compatible with Apple Silicon")
        print("   If issues occur, contact the package maintainer")
    
    return info["is_apple_silicon"]

def display_macos_camera_permission_help():
    """Display help for macOS camera permissions"""
    if platform.system() == "Darwin":  # macOS
        print("\nCamera Permission Guide for macOS:")
        print("--------------------------------")
        print("1. Open System Settings (or System Preferences)")
        print("2. Navigate to Privacy & Security > Camera")
        print("3. Ensure that Terminal, VS Code, and/or your Python environment have permission")
        print("4. If needed, add the application manually and restart it")
        print("5. You might need to run the app once, deny permission, then grant it")