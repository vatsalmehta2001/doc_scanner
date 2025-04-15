# Document Scanner

A lightweight Python application that uses your MacBook's built-in camera to scan documents in real-time. Optimized for Apple Silicon Macs (M-series chips) and macOS Sonoma.

## Features

- Minimal and fast with few dependencies
- Real-time document detection using edge detection
- Green rectangle overlay around detected documents
- Save scanned documents as JPEG images
- Compatible with Apple Silicon (M-series chips)
- Optimized for use with VS Code's integrated terminal

## Requirements

- Python 3.8 or higher
- macOS (optimized for Sonoma and Apple Silicon, but works on Intel Macs too)
- Built-in or external camera

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/doc-scanner.git
   cd doc-scanner
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

   For Apple Silicon Macs, if you encounter issues, try:
   ```bash
   pip install --no-binary :all: -e .
   ```

## Usage

### Command Line

Run the document scanner from the command line:

```bash
doc-scanner
```

Or with custom options:

```bash
doc-scanner --output scan_results --camera 1 --width 1920 --height 1080
```

### Options

- `--output`, `-o`: Directory to save scanned documents (default: `scans`)
- `--camera`, `-c`: Camera index (default: 0 for built-in camera)
- `--width`, `-W`: Camera width resolution (default: 1280)
- `--height`, `-H`: Camera height resolution (default: 720)

### Controls

- `s`: Save the current document as a JPEG image
- `q`: Quit the application

## Troubleshooting

### Camera Permissions on macOS

1. Go to System Settings (or System Preferences) > Privacy & Security > Camera
2. Make sure Terminal, VS Code, and/or your Python environment have permission to use the camera
3. If they don't appear in the list, you may need to run the app once, then grant permission when prompted
4. If still not working, try adding the application manually or restart your computer

### Apple Silicon (M-series) Compatibility

1. Make sure you're using Python for Apple Silicon (arm64):
   ```bash
   python -c "import platform; print(platform.machine())"
   ```
   This should output `arm64`.

2. If you experience issues with OpenCV or lite-camera:
   - Try reinstalling with the `--no-binary` flag:
     ```bash
     pip uninstall opencv-python lite-camera
     pip install --no-binary :all: opencv-python lite-camera
     ```
   - Make sure you're using native arm64 builds of libraries

3. If using Rosetta 2, try switching to a native arm64 Python distribution

### Common Issues

- **Black screen**: Check camera permissions
- **Slow performance**: Try reducing resolution with `--width` and `--height` options
- **Camera not found**: Verify the camera index with `--camera` option (try 0, 1, etc.)

## Development

### Running Tests

```bash
python -m unittest discover tests
```

### Project Structure

- `doc_scanner/`: Main package
  - `scanner.py`: Main application code
  - `document.py`: Document detection logic
  - `utils.py`: Utility functions
- `tests/`: Test files
- `requirements.txt`: Dependencies
- `setup.py`: Package configuration

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.