import base64
import json

from flask import Flask, jsonify, request
from flask_cors import CORS

# Import context manager and AI clients
from agent_context import context_manager
from ai_client import send_to_openrouter
from ollama_client import get_ollama_response

app = Flask(__name__)
CORS(app)


# Predefined repair plans for different objects
REPAIR_PLANS = {
    "light fixture": [
        "Turn off power at the circuit breaker",
        "Remove the old light fixture cover",
        "Disconnect the wiring from the old fixture",
        "Connect wiring to the new fixture (match wire colors)",
        "Secure the new fixture to the ceiling box",
        "Attach the fixture cover and restore power",
    ],
    "faucet": [
        "Turn off water supply valves under the sink",
        "Remove faucet handle by unscrewing the set screw",
        "Unscrew and remove the old cartridge or valve",
        "Install the new cartridge (ensure proper alignment)",
        "Reattach the faucet handle",
        "Turn on water supply and test for leaks",
    ],
    "door hinge": [
        "Open the door and support it with a wedge",
        "Remove the hinge pin using a hammer and nail punch",
        "Unscrew the old hinge from the door",
        "Align and screw the new hinge to the door",
        "Reattach the door and insert the hinge pin",
        "Test door swing and adjust if needed",
    ],
    "outlet": [
        "Turn off power at the circuit breaker",
        "Remove the outlet cover plate",
        "Unscrew the outlet from the electrical box",
        "Disconnect wires from the old outlet (note positions)",
        "Connect wires to the new outlet (match positions)",
        "Screw outlet into box and replace cover plate",
    ],
    "cabinet door": [
        "Remove cabinet door by unscrewing hinges",
        "Mark hinge positions on the new door",
        "Attach hinges to the new door",
        "Align door with cabinet frame",
        "Screw hinges to cabinet frame",
        "Adjust door alignment and test opening/closing",
    ],
}

# Safety warnings for different repair types
SAFETY_WARNINGS = {
    "light fixture": "⚠️ SAFETY: Ensure power is OFF at circuit breaker before starting!",
    "faucet": "⚠️ SAFETY: Turn off water supply before starting to avoid flooding!",
    "door hinge": "⚠️ SAFETY: Support the door to prevent it from falling!",
    "outlet": "⚠️ SAFETY: Ensure power is OFF at circuit breaker before starting!",
    "cabinet door": "⚠️ SAFETY: Have someone help support the door during installation!",
}


def identify_object_from_frame(image_base64: str) -> str | None:
    """
    Use AI vision to identify what object is in the frame.

    Args:
        image_base64: Base64 encoded image

    Returns:
        Object type string (e.g., "light fixture") or None if not recognized
    """
    try:
        # Use Ollama for object detection
        # You can also use OpenRouter or Gemini here
        prompt = """Identify the main object in this image that might need repair.
        Respond with ONLY ONE of these options:
        - light fixture
        - faucet
        - door hinge
        - outlet
        - cabinet door
        - unknown

        Just respond with the object name, nothing else."""

        # Call Ollama with vision model
        response = get_ollama_response(
            prompt=prompt,
            image_frame=image_base64,  # Pass base64 directly
            model="llava",
        )

        # Clean up response
        detected_object = response.strip().lower()

        # Check if it's a known repair object
        if detected_object in REPAIR_PLANS:
            return detected_object

        return None

    except Exception as e:
        print(f"Error identifying object: {e}")
        return None


def analyze_step_completion(
    image_base64: str,
    current_step: str,
    context_info: str,
) -> dict:
    """
    Use AI to analyze if the current repair step appears complete.

    Args:
        image_base64: Base64 encoded image
        current_step: The instruction for current step
        context_info: Additional context about the repair

    Returns:
        Dict with 'complete' (bool) and 'feedback' (str)
    """
    try:
        prompt = f"""You are analyzing a repair in progress.

Context: {context_info}
Current step: {current_step}

Based on the image, does it look like this step is complete?
Respond in JSON format:
{{"complete": true/false, "feedback": "brief guidance"}}"""

        response = get_ollama_response(
            prompt=prompt,
            image_frame=image_base64,
            model="llava",
        )

        # Try to parse JSON response
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # If not valid JSON, assume step is not complete
            return {"complete": False, "feedback": response}

    except Exception as e:
        print(f"Error analyzing step: {e}")
        return {"complete": False, "feedback": "Unable to analyze step"}


@app.route("/process", methods=["POST"])
def process_frame():
    """Legacy endpoint - kept for backward compatibility"""
    data = request.get_json()
    with open("payload.json", "w") as f:
        f.write(json.dumps(data))

    return (
        jsonify(
            {
                "x": 320,
                "y": 240,
                "label": "Use /api/analyze instead",
            }
        ),
        200,
    )


@app.route("/api/analyze", methods=["POST"])
def analyze_frame():
    """
    Main endpoint for AR repair assistant with context/memory.

    Expected request body:
    {
        "session_id": "unique-session-id",
        "image": "base64-encoded-image",
        "timestamp": 1234567890
    }

    Returns overlay data for AR glasses:
    {
        "x": 320,
        "y": 240,
        "label": "Step 1: Turn off power...",
        "step": 1,
        "total_steps": 6,
        "completed": [1, 2],
        "is_complete": false
    }
    """
    try:
        data = request.get_json()

        # Extract request data
        session_id = data.get("session_id", "default-session")
        image_base64 = data.get("image")
        timestamp = data.get("timestamp")

        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        # Get or create context for this session
        context = context_manager.get_context(session_id)

        # Check if this is the first frame (no repair started yet)
        if not context.is_repair_started():
            # Use AI to identify what object needs repair
            detected_object = identify_object_from_frame(image_base64)

            if detected_object and detected_object in REPAIR_PLANS:
                # Start a new repair session
                repair_steps = REPAIR_PLANS[detected_object]
                context_manager.start_repair(
                    session_id=session_id,
                    object_type=detected_object,
                    repair_steps=repair_steps,
                )

                # Show safety warning
                safety_warning = SAFETY_WARNINGS.get(detected_object, "")
                if safety_warning:
                    context_manager.add_safety_warning(session_id, safety_warning)

                return jsonify(
                    {
                        "x": 320,  # Center of screen
                        "y": 100,  # Top area
                        "label": f"Detected: {detected_object.title()}",
                        "message": safety_warning,
                        "step": 0,
                        "total_steps": len(repair_steps),
                        "repair_started": True,
                    }
                )
            else:
                # Object not recognized
                return jsonify(
                    {
                        "x": 320,
                        "y": 240,
                        "label": "Point camera at repair object",
                        "message": "Unable to detect a repairable object",
                        "step": 0,
                        "total_steps": 0,
                    }
                )

        # Repair is in progress - show current step
        else:
            current_instruction = context.get_current_step_instruction()

            # Check if repair is complete
            if context.is_repair_complete():
                return jsonify(
                    {
                        "x": 320,
                        "y": 240,
                        "label": f"{context.object_type.title()} repair complete! ✓",
                        "message": "Great job! Repair finished.",
                        "step": context.current_step,
                        "total_steps": context.total_steps,
                        "completed_steps": context.completed_steps,
                        "is_complete": True,
                    }
                )

            # Show current step instruction
            return jsonify(
                {
                    "x": 320,
                    "y": 150,
                    "label": f"Step {context.current_step}/{context.total_steps}",
                    "message": current_instruction,
                    "step": context.current_step,
                    "total_steps": context.total_steps,
                    "completed_steps": context.completed_steps,
                    "is_complete": False,
                    "object_type": context.object_type,
                }
            )

    except Exception as e:
        print(f"Error in analyze_frame: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/confirm_step", methods=["POST"])
def confirm_step():
    """
    Manually confirm that current step is complete and advance to next step.

    Expected request body:
    {
        "session_id": "unique-session-id"
    }

    Returns:
    {
        "success": true,
        "next_step": 2,
        "next_instruction": "Remove the old light fixture cover",
        "is_complete": false
    }
    """
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default-session")

        # Get context
        context = context_manager.get_context(session_id)

        if not context.is_repair_started():
            return jsonify({"error": "No repair in progress"}), 400

        # Mark current step complete and advance
        context_manager.mark_step_complete(session_id)

        # Get updated context
        context = context_manager.get_context(session_id)

        # Check if repair is now complete
        if context.is_repair_complete():
            return jsonify(
                {
                    "success": True,
                    "is_complete": True,
                    "message": "Repair complete! All steps finished.",
                }
            )

        # Return next step info
        next_instruction = context.get_current_step_instruction()

        return jsonify(
            {
                "success": True,
                "next_step": context.current_step,
                "next_instruction": next_instruction,
                "total_steps": context.total_steps,
                "completed_steps": context.completed_steps,
                "is_complete": False,
            }
        )

    except Exception as e:
        print(f"Error in confirm_step: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """
    Reset a repair session to start over.

    Expected request body:
    {
        "session_id": "unique-session-id"
    }
    """
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default-session")

        context_manager.reset_session(session_id)

        return jsonify({"success": True, "message": "Session reset"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get status of a session"""
    session_id = request.args.get("session_id", "default-session")
    context = context_manager.get_context(session_id)

    return jsonify(
        {
            "session_id": session_id,
            "repair_started": context.is_repair_started(),
            "object_type": context.object_type,
            "current_step": context.current_step,
            "total_steps": context.total_steps,
            "completed_steps": context.completed_steps,
            "is_complete": context.is_repair_complete(),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
