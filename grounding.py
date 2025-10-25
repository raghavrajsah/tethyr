import threading
from ultralytics import YOLO


class Grounding:
    def __init__(self, model_path, initial_prompt="person"):
        """Initialize the YOLOE model with a default prompt"""
        self.model = None
        self.model_lock = threading.Lock()
        
        print("Loading YOLOE model...")
        self.model = YOLO(model_path)
        self.update_prompt(initial_prompt)
        print(f"YOLOE model loaded with prompt: '{self.prompt_text}'")
    
    def detect(self, frame):
        """Run detection on a frame and return results"""
        with self.model_lock:
            results = self.model.predict(frame, verbose=False, conf=0.1, iou=0.5)
        return results
    
    def update_prompt(self, prompt):
        """Update the text prompt for detection"""
        self.prompt_text = prompt
        with self.model_lock:
            classes = [cls.strip() for cls in prompt.split(',')]
            print(f"Updating YOLOE to detect: {classes}")
            text_embeddings = self.model.get_text_pe(classes)
            self.model.set_classes(classes, text_embeddings)
            print(f"Successfully updated detection prompt to: {classes}")

