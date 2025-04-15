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
    """Display tips for running on Apple Silicon Macs"""
    info = get_system_info()
    
    if info["is_apple_silicon"]:
        cv_version, cv_compatible = check_opencv_version()
        
        print("\nApple Silicon (M-series) Compatibility Info:")
        print("-------------------------------------------")
        print(f"• System: {info['platform']}")
        print(f"• Python version: {info['python_version']}")
        print(f"• Architecture: {info['architecture']}")
        print(f"• OpenCV version: {cv_version}")
        print(f"• OpenCV compatibility: {'Good' if cv_compatible else 'May need update'}")
        
        if not cv_compatible:
            print("\nTip: For better performance on Apple Silicon, install the latest OpenCV:")
            print("  pip uninstall opencv-python -y")
            print("  pip install --no-binary :all: opencv-python")
    
    return info["is_apple_silicon"]


def display_macos_camera_permission_help():
    """Display help for macOS camera permissions"""
    if platform.system() == "Darwin":  # macOS
        print("\nCamera Permission Guide for macOS:")
        print("--------------------------------")
        print("1. Open System Settings > Privacy & Security > Camera")
        print("2. Ensure Terminal, VS Code, and/or your Python environment have permission")
        print("3. After granting permission, restart your terminal or VS Code")
        print("4. If issues persist, try running the app from Terminal directly:")
        print("   python -m doc_scanner.scanner")