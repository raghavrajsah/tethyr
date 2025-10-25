from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
import os
from datetime import datetime
from ollama_client import get_ollama_response

app = Flask(__name__)

class Camera:
    def __init__(self):
        self.camera = None
        self.frame = None
        self.is_running = False
        
    def start(self):
        """Start the camera capture"""
        self.camera = cv2.VideoCapture(0)  # 0 is the default camera on macOS
        if not self.camera.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        
        # Start a thread to continuously capture frames
        self.capture_thread = threading.Thread(target=self._capture_frames)
        #this is where i would put the ollama request/response async 
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def _capture_frames(self):
        """Continuously capture frames from the camera"""
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Flip the frame horizontally for mirror effect
                self.frame = cv2.flip(frame, 1)
            time.sleep(0.03)  # ~30 FPS
            
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

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """Analyze the current camera frame with Ollama"""
    try:
        # Get the current frame from the camera
        frame = camera.get_frame()
        if frame is None:
            return jsonify({'success': False, 'error': 'No frame available'})

        # Get optional prompt and system prompt from request
        data = request.get_json() or {}
        prompt = data.get('prompt', 'What do you see in this image? Describe it briefly.')
        system_prompt = data.get('system_prompt', 'You are a helpful assistant that describes images concisely.')
        model = data.get('model', 'llava')

        # Call Ollama to analyze the frame
        response_text = get_ollama_response(
            prompt=prompt,
            image_frame=frame,
            system_prompt=system_prompt,
            model=model
        )

        return jsonify({
            'success': True,
            'response': response_text,
            'prompt': prompt
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
