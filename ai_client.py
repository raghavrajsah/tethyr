"""
AI Client - Simple functions to send image and text to AI models
"""

import asyncio
import json
import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

import requests

try:
    import websockets
except ImportError:
    websockets = None

from img_conversion import frame_to_base64


async def stream_to_gemini_live(
    image_frame: Any,
    text: str = "",
    api_key: str | None = None,
    model: str = "gemini-2.0-flash-exp",
) -> AsyncGenerator[dict[str, Any]]:
    """
    Stream image frame and text to Gemini Live API with real-time responses
    Uses WebSocket for bidirectional streaming communication

    Args:
        image_frame: OpenCV frame (numpy array) or base64 string
        text: Text prompt to send with the image
        api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        model: Model to use (default: gemini-2.0-flash-exp)

    Yields:
        Dictionary with streaming response chunks from Gemini

    Example:
        >>> async for response in stream_to_gemini_live(frame, "What do you see?"):
        ...     if 'text' in response:
        ...         print(response['text'], end='', flush=True)
    """
    if websockets is None:
        raise ImportError("websockets package required. Install with: uv add websockets")

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not set. Set GOOGLE_API_KEY environment variable or pass api_key parameter")

    # Convert frame to base64 if needed
    if hasattr(image_frame, "shape"):
        base64_image = frame_to_base64(image_frame)
    else:
        base64_image = image_frame

    # Gemini Live API WebSocket endpoint
    ws_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"

    try:
        async with websockets.connect(ws_url) as ws:
            # Send setup message
            setup_message = {
                "setup": {
                    "model": f"models/{model}",
                    "generation_config": {"response_modalities": ["TEXT"]},
                }
            }
            await ws.send(json.dumps(setup_message))

            # Wait for setup acknowledgment
            setup_response = await ws.recv()
            setup_data = json.loads(setup_response)

            if "setupComplete" not in setup_data:
                raise Exception(f"Setup failed: {setup_data}")

            # Send the actual content (image + text)
            content_parts = []

            if text:
                content_parts.append({"text": text})

            content_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64_image}})

            content_message = {
                "clientContent": {
                    "turns": [{"role": "user", "parts": content_parts}],
                    "turn_complete": True,
                }
            }

            await ws.send(json.dumps(content_message))

            # Receive streaming responses
            while True:
                try:
                    response = await ws.recv()
                    data = json.loads(response)

                    # Check for server content (the actual response)
                    if "serverContent" in data:
                        server_content = data["serverContent"]

                        if "modelTurn" in server_content:
                            model_turn = server_content["modelTurn"]

                            if "parts" in model_turn:
                                for part in model_turn["parts"]:
                                    if "text" in part:
                                        yield {"type": "text", "text": part["text"], "raw": data}

                            # Check if turn is complete
                            if server_content.get("turnComplete", False):
                                break

                    # Check for errors
                    if "error" in data:
                        yield {"type": "error", "error": data["error"], "raw": data}
                        break

                except websockets.exceptions.ConnectionClosed:
                    break

    except Exception as e:
        yield {"type": "error", "error": str(e)}


def stream_to_gemini_live_sync(
    image_frame: Any,
    text: str = "",
    api_key: str | None = None,
    model: str = "gemini-2.0-flash-exp",
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Synchronous wrapper for stream_to_gemini_live

    Args:
        image_frame: OpenCV frame (numpy array) or base64 string
        text: Text prompt to send with the image
        api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        model: Model to use
        callback: Optional callback function called for each response chunk

    Returns:
        List of all response chunks

    Example:
        >>> responses = stream_to_gemini_live_sync(
        ...     frame,
        ...     "What do you see?",
        ...     callback=lambda r: print(r.get('text', ''), end='')
        ... )
    """

    async def run():
        results = []
        async for response in stream_to_gemini_live(image_frame, text, api_key, model):
            results.append(response)
            if callback:
                callback(response)
        return results

    return asyncio.run(run())


def send_to_openrouter(
    prompt: str,
    images: list[Any] | None = None,
    model: str = "google/gemini-2.0-flash-exp:free",
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Send a text prompt and optional images to OpenRouter

    Args:
        prompt: Text prompt to send
        images: Optional list of image frames (OpenCV numpy arrays or base64 strings)
        model: Model identifier (e.g., "google/gemini-2.0-flash-exp:free", "anthropic/claude-3.5-sonnet")
        api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided)

    Returns:
        Dictionary with OpenRouter API response

    Example:
        >>> result = send_to_openrouter(
        ...     prompt="What's in this image?",
        ...     images=[frame],
        ...     model="google/gemini-2.0-flash-exp:free"
        ... )
        >>> print(result['choices'][0]['message']['content'])
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key not set. Set OPENROUTER_API_KEY environment variable or pass api_key parameter")

    base_url = "https://openrouter.ai/api/v1/chat/completions"

    # Build the content array
    content = [{"type": "text", "text": prompt}]

    # Add images if provided
    if images:
        for image in images:
            # Convert frame to base64 if it's a numpy array
            if hasattr(image, "shape"):
                base64_image = frame_to_base64(image)
            else:
                base64_image = image  # Assume it's already base64

            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

    # Build the request payload
    payload = {"model": model, "messages": [{"role": "user", "content": content}]}

    # Make API request
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.post(base_url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()
