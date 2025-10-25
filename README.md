# Webcam Webapp

A simple Python webapp that displays webcam input in a web browser, designed for macOS.

## Features

- Real-time webcam feed display
- Clean, responsive web interface
- Automatic camera detection and error handling
- Optimized for macOS webcam access

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Grant camera permissions when prompted by your browser

4. Press `Ctrl+C` to stop the application

## Requirements

- Python 3.7+
- macOS with built-in camera
- Modern web browser with camera access support

## Troubleshooting

- If you see "Camera not available", make sure to grant camera permissions to your browser
- If the video doesn't load, try refreshing the page
- Ensure no other applications are using the camera
