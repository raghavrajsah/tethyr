"""
Object detection using YOLO grounding model
Runs continuous detection and allows Gemini to change what to look for
"""

import base64
import io
import threading
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image
from ultralytics import YOLO


class GroundingDetector:
    """YOLO-based object detector with continuous detection capability

    Architecture:
    - Runs detection continuously on incoming frames
    - Gemini can ONLY change the detection prompt (what to look for)
    - Detection happens independently of Gemini's tool calls
    - Thread-safe for concurrent access from multiple clients
    """

    def __init__(
        self,
        model_path: str = "yoloe-11m-seg.pt",
        initial_prompt: str = "person, object",
    ):
        """Initialize the YOLO model with a default prompt

        Args:
            model_path: Path to the YOLO model file
            initial_prompt: Initial prompt for object detection
        """

        self.model = None
        self.model_lock = threading.Lock()
        self.prompt_text = None

        logger.info("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.update_prompt(initial_prompt)
        logger.info(f"YOLO model loaded with prompt: '{self.prompt_text}'")

    def update_prompt(self, prompt: str):
        """Update the text prompt for detection (thread-safe)

        This is the ONLY thing Gemini should do via its tool.
        The actual detection runs continuously on every frame.

        Args:
            prompt: Comma-separated list of object classes to detect
        """
        # Cache check: only update if prompt has changed
        if self.prompt_text == prompt:
            logger.debug(f"Prompt unchanged, skipping re-embedding: '{prompt}'")
            return

        self.prompt_text = prompt
        with self.model_lock:
            classes = [cls.strip() for cls in prompt.split(",")]
            logger.info(f"Changed YOLO prompt to detect: {classes}")
            text_embeddings = self.model.get_text_pe(classes)
            self.model.set_classes(classes, text_embeddings)
            logger.info(f"Successfully updated detection classes to: {classes}")

    def get_current_prompt(self) -> str:
        """Get the current detection prompt

        Returns:
            Current prompt string
        """
        return self.prompt_text

    def detect_in_frame(
        self,
        frame: np.ndarray,
        conf: float = 0.1,
        iou: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Run detection on a frame and return structured results

        This is meant to be called continuously on every frame.

        Args:
            frame: Image frame (numpy array)
            conf: Confidence threshold
            iou: IOU threshold

        Returns:
            List of detection dictionaries with bbox, label, and confidence
        """
        try:
            with self.model_lock:
                results = self.model.predict(frame, verbose=False, conf=conf, iou=iou)

            # Get current classes for result parsing
            classes = [cls.strip() for cls in self.prompt_text.split(",")]

            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()

                    for box, conf_val, cls_id in zip(
                        boxes,
                        confidences,
                        class_ids,
                        strict=True,
                    ):
                        x1, y1, x2, y2 = box
                        detection = {
                            "bbox": {
                                "x": int(x1),
                                "y": int(y1),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1),
                            },
                            "label": classes[int(cls_id)],
                            "confidence": float(conf_val),
                        }
                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error in detect_in_frame: {e}", exc_info=True)
            return []

    def detect_in_base64(
        self,
        image_base64: str,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Convenience method to detect in base64 encoded image

        Args:
            image_base64: Base64 encoded image (JPEG/PNG)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS

        Returns:
            List of detection dictionaries with bbox, label, and confidence
        """
        try:
            # Decode base64 image
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]

            img_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(img_bytes))

            # Convert to numpy array for YOLO
            frame = np.array(image)

            return self.detect_in_frame(frame, conf=conf_threshold, iou=iou_threshold)

        except Exception as e:
            logger.error(f"Error in detect_in_base64: {e}", exc_info=True)
            return []


# Tool definition for Gemini Live API - ONLY changes the prompt
GROUNDING_TOOL_DECLARATION = {
    "name": "change_detection_target",
    "description": """Change what objects YOLO should detect in the video stream.

    Important: This does NOT trigger detection---detection runs continuously on every frame.
    This only changes WHAT objects to look for.

    Use this when the you want to ask the user to focus on a different object. For example, when
    wiring a light bulb, you may want to ask the user to focus on the bulb first, then the outlet, then the light fixture.""",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Comma-separated list of object types to detect (e.g., 'light fixture, outlet, faucet')",
            },
        },
        "required": ["prompt"],
    },
}
