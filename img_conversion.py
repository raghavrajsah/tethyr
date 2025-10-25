import base64
import cv2

def frame_to_base64(frame: any) -> str:
    """
    Convert OpenCV frame to base64 string for Ollama

    Args:
        frame: OpenCV frame (numpy array)

    Returns:
        Base64 encoded string
    """
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buffer).decode('utf-8')