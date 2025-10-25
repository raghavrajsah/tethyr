# Tethyr Labs - AI-Powered AR Glasses

An AI-powered AR glasses development platform with real-time computer vision, object detection, and AI agent capabilities.

## Architecture

### AR Glasses (Snap Lens Studio)
**Location**: `snap/snap/Assets/CoordinateFetcher.ts`

- **File**: TypeScript component for Lens Studio
- **Function**: Captures camera frames from AR glasses and sends to server
- **To add new UI**: Edit `CoordinateFetcher.ts` in the `displayLabel()` method (lines 124-151) to modify visual markers and positioning
- **Server endpoint**: Sends frames to `http://localhost:5000/process` (configured in `serverUrl` property)

### AI Agent
**Location**: Multiple files in root directory

- **`ai_client.py`**: Gemini Live API and OpenRouter integration
  - `stream_to_gemini_live()`: Real-time streaming with Gemini
  - `send_to_openrouter()`: Multi-model support via OpenRouter
- **`ollama_client.py`**: Local Ollama integration
  - `get_ollama_response()`: Vision-capable models (llava, llama3.2-vision)
- **`grounding.py`**: YOLO-based object detection
  - `Grounding.detect()`: Run detection on frames
  - `Grounding.update_prompt()`: Modify detection classes dynamically

**To add new AI tools**: Create functions in `ai_client.py` or `ollama_client.py` following existing patterns

### Backend Servers
- **`snap/server.py`**: Flask server for AR glasses (port 5000)
  - Receives frames from glasses
  - Returns coordinates and labels for AR overlay
- **`app.py`**: Webcam webapp (port 5001)
  - Camera feed display
  - Frame analysis with AI models

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

## Usage

### Start AR Glasses Server
```bash
python snap/server.py
```

### Start Webcam Webapp
```bash
python app.py
# Visit http://localhost:5001
```

## Requirements

- Python 3.13+
- Snap Lens Studio (for AR glasses development)
- Ollama (optional, for local AI models)
- OpenRouter API key (optional, for cloud AI models)
- Google API key (optional, for Gemini Live)

## Environment Variables

- `GOOGLE_API_KEY`: For Gemini Live API
- `OPENROUTER_API_KEY`: For OpenRouter multi-model access
