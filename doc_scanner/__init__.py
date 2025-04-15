"""
Document Scanner - A lightweight application for scanning documents with your camera
"""

__version__ = "0.1.0"

from .scanner import main
from .document import detect_document, enhance_document
from .utils import (
    get_system_info,
    display_apple_silicon_tips,
    display_macos_camera_permission_help
)

__all__ = [
    'main',
    'detect_document',
    'enhance_document',
    'get_system_info',
    'display_apple_silicon_tips',
    'display_macos_camera_permission_help'
]