from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
import os
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

class Camera:
    def __init__(self):
        self.camera = None
        self.frame = None
        self.is_running = False
        self.yoloe_model = None
        self.detection_enabled = True
        self.prompt_text = "person"  # Default prompt - can be changed
        
    def start(self):
        """Start the camera capture"""
        self.camera = cv2.VideoCapture(0)  # 0 is the default camera on macOS
        if not self.camera.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize YOLOE model
        print("Loading YOLOE model...")
        self.yoloe_model = YOLO("yoloe-11s-seg.pt")
        
        # Set custom classes for detection
        self._set_detection_prompt(self.prompt_text)
        print(f"YOLOE model loaded with prompt: '{self.prompt_text}'")
        
        self.is_running = True
        
        # Start a thread to continuously capture frames
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def _set_detection_prompt(self, prompt):
        """Set the text prompt for YOLOE detection"""
        self.prompt_text = prompt
        if self.yoloe_model:
            # Split by comma for multiple objects
            classes = [cls.strip() for cls in prompt.split(',')]
            print(f"Setting YOLOE classes to: {classes}")
            try:
                text_embeddings = self.yoloe_model.get_text_pe(classes)
                print(f"Text embeddings shape: {text_embeddings.shape if hasattr(text_embeddings, 'shape') else 'N/A'}")
                self.yoloe_model.set_classes(classes, text_embeddings)
                print("Successfully set classes")
            except Exception as e:
                print(f"Error setting classes: {e}")
                # Try alternative approach - just set classes without embeddings
                print("Trying to detect 'person' instead...")
                self.prompt_text = "person"
                classes = ["person"]
                self.yoloe_model.set_classes(classes, self.yoloe_model.get_text_pe(classes))
        
    def _capture_frames(self):
        """Continuously capture frames from the camera"""
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Flip the frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Run YOLOE detection if enabled
                if self.detection_enabled and self.yoloe_model:
                    frame = self._process_frame_with_yoloe(frame)
                
                self.frame = frame
            time.sleep(0.03)  # ~30 FPS
    
    def _process_frame_with_yoloe(self, frame):
        """Process frame with YOLOE and draw bounding boxes"""
        try:
            # Run YOLOE inference with lower confidence threshold
            # Note: Using very low confidence to see any detections
            results = self.yoloe_model.predict(frame, verbose=False, conf=0.05, iou=0.5)
            
            # Draw bounding boxes on the frame
            if len(results) > 0:
                result = results[0]
                
                # Debug: Print detection info
                if result.boxes is not None:
                    print(f"Detections found: {len(result.boxes)}")
                    if len(result.boxes) > 0:
                        print(f"Confidences: {result.boxes.conf.cpu().numpy()}")
                
                # Draw boxes and labels
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get box coordinates
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
                    class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        confidence = confidences[i]
                        
                        # Get class name
                        if class_ids is not None and len(result.names) > 0:
                            class_id = int(class_ids[i])
                            label = result.names.get(class_id, self.prompt_text)
                        else:
                            label = self.prompt_text
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green color
                        thickness = 2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw label with confidence
                        label_text = f"{label}: {confidence:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        font_thickness = 2
                        
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, font, font_scale, font_thickness
                        )
                        
                        # Draw background rectangle for text
                        cv2.rectangle(
                            frame,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            color,
                            -1  # Filled rectangle
                        )
                        
                        # Draw text
                        cv2.putText(
                            frame,
                            label_text,
                            (x1, y1 - 5),
                            font,
                            font_scale,
                            (0, 0, 0),  # Black text
                            font_thickness
                        )
        except Exception as e:
            print(f"Error in YOLOE processing: {e}")
        
        return frame
            
    def get_frame(self):
        """Get the latest frame"""
        return self.frame
        
    def stop(self):
        """Stop the camera capture"""
        self.is_running = False
        if self.camera:
            self.camera.release()

# Global camera instance
camera = Camera()

def generate_frames():
    """Generate video frames for streaming"""
    while True:
        frame = camera.get_frame()
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_photo', methods=['POST'])
def save_photo():
    """Save the current camera frame as a photo"""
    try:
        # Get the current frame from the camera
        frame = camera.get_frame()
        if frame is None:
            return jsonify({'success': False, 'error': 'No frame available'})
        
        # Create photos directory if it doesn't exist
        photos_dir = 'photos'
        if not os.path.exists(photos_dir):
            os.makedirs(photos_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'webcam_photo_{timestamp}.jpg'
        filepath = os.path.join(photos_dir, filename)
        
        # Save the frame as JPEG
        success = cv2.imwrite(filepath, frame)
        
        if success:
            return jsonify({'success': True, 'filename': filename})
        else:
            return jsonify({'success': False, 'error': 'Failed to save image'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    try:
        # Start the camera
        camera.start()
        print("Starting webcam webapp...")
        print("Open your browser and go to: http://localhost:5001")
        print("Press Ctrl+C to stop the application")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.stop()
