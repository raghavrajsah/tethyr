"""
Tool declarations for Gemini to interact with the scratchpad
"""

# Tool declarations for scratchpad operations
SCRATCHPAD_TOOL_DECLARATIONS = [
    {
        "name": "read_scratchpad",
        "description": "Read your scratchpad notes. Use this to check what plan/steps you've written down before.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "write_scratchpad",
        "description": "Write or replace your scratchpad content. Use this to save your plan, current step, or any notes you need to remember. This REPLACES all previous content.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to write to your scratchpad. Format it however you like - bullet points, numbered steps, prose, etc.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "append_scratchpad",
        "description": "Add content to the end of your scratchpad without erasing what's already there. Use this to add new notes or steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to append to your scratchpad",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "clear_scratchpad",
        "description": "Clear all content from your scratchpad. Use when starting a completely new task.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
