"""
Document Scanner - A lightweight application for scanning documents with your camera
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .scanner import DocumentScanner, main
from .document import detect_document
from .utils import (
    get_system_info,
    check_camera_permission,
    display_apple_silicon_tips,
    display_macos_camera_permission_help
)

__all__ = [
    'DocumentScanner',
    'main',
    'detect_document',
    'get_system_info',
    'check_camera_permission',
    'display_apple_silicon_tips',
    'display_macos_camera_permission_help'
]