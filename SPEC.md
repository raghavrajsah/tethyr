# API Specification

## Client → Server Messages

### 1. Handshake
```python
{
  "type": "handshake",
  "timestamp": 1698765432000,
  "device": "spectacles",
  "capabilities": {
    "video": true,
    "audio": true
  }
}
```

### 2. Video Frame
```python
{
  "type": "video_frame",
  "data": "base64_encoded_jpeg...",
  "timestamp": 1698765432000,
  "frame_number": 123,
  "resolution": {
    "width": 1280,
    "height": 720
  }
}
```

### 3. Audio Chunk
```python
{
  "type": "audio_chunk",
  "data": "base64_encoded_float32_samples...",
  "timestamp": 1698765432000,
  "frame_number": 123,
  "sample_rate": 16000,
  "samples": 1024,
  "channels": 1
}
```

---

## Server → Client Messages

### 1. Handshake Ack
```python
{
  "type": "handshake_ack",
  "server": "AR Processing Server",
  "timestamp": "2024-10-25T10:30:00.000Z"
}
```

### 2. Overlay (Text Display - Replaces Previous)
```python
{
  "type": "overlay",
  "text": "Look left!",
  "timestamp": "2024-10-25T10:30:00.000Z",
  "color": {  # Optional
    "r": 1.0,
    "g": 1.0,
    "b": 1.0,
    "a": 1.0
  },
  "position": {  # Optional
    "x": 0.5,
    "y": 0.5
  }
}
```

### 3. Bounding Box (Replaces Previous)
```python
{
  "type": "bbox",
  "bbox": {
    "x": 150,        # Pixel coordinates
    "y": 200,
    "width": 300,
    "height": 400,
    "label": "Person",
    "confidence": 0.95  # Optional
  },
  "timestamp": "2024-10-25T10:30:00.000Z",
  "color": {  # Optional
    "r": 1.0,
    "g": 0.0,
    "b": 0.0,
    "a": 0.8
  }
}
```

### 4. Clear
```python
{
  "type": "clear",
  "timestamp": "2024-10-25T10:30:00.000Z"
}
```

---

## Key Behaviors

- **Overlay**: Each new overlay message **replaces** the previous text. Server can stream updates by sending overlay messages continuously.
- **BBox**: Each new bbox message **replaces** the previous bounding box visualization.
- **Coordinates**: Bounding box uses pixel coordinates matching the resolution sent in video frames.
- **Audio Format**: Float32 samples, base64 encoded bytes (little-endian).