┌─────────────────┐
│ Snap Spectacles │ (Camera captures frames)
└────────┬────────┘
         │ WiFi
         ▼
┌─────────────────┐
│ Flask Server    │ (Receives frames)
│  • OpenCV       │
│  • YOLO         │ ← Grounding/object detection
│  • Ollama/Gemini│ ← Vision AI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agent Logic     │ (The "brain")
│  • State tracker│ ← What step are we on?
│  • Vision model │ ← What do I see?
│  • Planner      │ ← What should I show next?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Overlay Data    │ (JSON response)
│  {              │
│   step: 2,      │
│   text: "...",  │
│   arrow: {x,y}, │
│   highlight: {} │
│  }              │
└────────┬────────┘
         │ WiFi
         ▼
┌─────────────────┐
│ Lens Studio     │ (Renders overlays)
│ (Your JS code)  │
└─────────────────┘