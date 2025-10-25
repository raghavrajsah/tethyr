## Installation
```bash
uv init
uv sync
uv pip install git+https://github.com/ultralytics/CLIP.git
```


## Usage
```bash
uv run app.py
```

This will create a webpage that takes camera input, detects objects using YOLOE, and displays the objects' bounding boxes. 

The default prompt is "person". You can change the prompt using the textbox. The model supports a list of objects separated by comma, such as "person, light bulb, painting".


## Notes
app.py is just a placeholder for the actual AR input output. The core logic is just grounding.py. 

Grounding is the technical term for connecting a natural language description to the specific objects in an image.