import base64
from typing import Any, Optional

from ollama import ChatResponse, chat


def frame_to_base64(frame: Any) -> str:
    """
    Convert OpenCV frame to base64 string for Ollama

    Args:
        frame: OpenCV frame (numpy array)

    Returns:
        Base64 encoded string
    """
    import cv2

    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buffer).decode("utf-8")


def get_ollama_response(
    prompt: str,
    image_frame: Optional[Any] = None,
    system_prompt: Optional[str] = None,
    model: str = "llava",
) -> str:
    """
    Get response from Ollama with support for image frames, text, and system prompts

    Args:
        prompt: User prompt/question
        image_frame: Optional OpenCV frame (numpy array) or base64 string
        system_prompt: Optional system prompt to guide model behavior
        model: Ollama model to use (default: 'llava' for vision support)
               Other vision models: 'llama3.2-vision', 'llava-llama3', 'bakllava'

    Returns:
        String response from the model

    Example:
        >>> response = get_ollama_response(
        ...     prompt="What do you see in this image?",
        ...     image_frame=frame,
        ...     system_prompt="You are a helpful assistant that describes images in detail.",
        ...     model='llava'
        ... )
        >>> print(response)
    """
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    # Build user message
    user_message = {
        "role": "user",
        "content": prompt,
    }

    # Add image if provided
    if image_frame is not None:
        # Convert frame to base64 if it's a numpy array
        if hasattr(image_frame, "shape"):
            base64_image = frame_to_base64(image_frame)
        else:
            base64_image = image_frame  # Assume it's already base64

        user_message["images"] = [base64_image]

    messages.append(user_message)

    # Call Ollama
    response: ChatResponse = chat(model=model, messages=messages)

    return response["message"]["content"]
