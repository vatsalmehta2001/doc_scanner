# Document Scanner

A powerful document scanning application that uses computer vision and OCR to capture, process, and extract text from documents.

## Features

- Real-time document edge detection
- Automatic perspective correction
- High-quality image capture and processing
- OCR text extraction with multiple PSM modes
- Structured data extraction (dates, emails, phone numbers, etc.)
- Support for various document types
- Console feedback and debugging information

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/doc_scanner.git
cd doc_scanner
```

2. Install dependencies:
```bash
pip install -e .
```

3. Make sure you have Tesseract OCR installed:
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`
- Windows: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

1. Run the document scanner:
```bash
python scripts/advanced_scanner.py
```

2. Controls:
- Press 'c' to capture a document
- Press 'q' to quit
- Press 'h' for help

3. The captured documents will be saved in the `scanned_documents` directory with:
- PNG image file
- TXT file with extracted text
- JSON file with structured data

## Project Structure

```
doc_scanner/
├── doc_scanner/           # Main package
│   ├── __init__.py
│   ├── document.py       # Document processing
│   ├── scanner.py        # Scanner implementation
│   └── text_processor.py # Text extraction
├── scripts/              # Command-line tools
│   └── advanced_scanner.py
├── tests/               # Test suite
│   ├── __init__.py
│   └── test_*.py
├── scanned_documents/   # Output directory
├── requirements.txt     # Dependencies
├── setup.py            # Package configuration
├── LICENSE             # License information
└── README.md          # This file
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
