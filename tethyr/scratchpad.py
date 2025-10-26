"""
Simple scratchpad for agents to keep notes and track plans
"""

from datetime import datetime


class Scratchpad:
    """A simple text-based scratchpad for the agent to keep notes"""

    def __init__(self):
        self.content: str = ""
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = datetime.now().isoformat()

    def read(self) -> str:
        """Read the current scratchpad content

        Returns:
            The current scratchpad content
        """
        return self.content

    def write(self, content: str) -> None:
        """Write/replace the scratchpad content

        Args:
            content: New content to write to scratchpad
        """
        self.content = content
        self.updated_at = datetime.now().isoformat()

    def append(self, content: str) -> None:
        """Append content to the scratchpad

        Args:
            content: Content to append
        """
        if self.content:
            self.content += "\n" + content
        else:
            self.content = content
        self.updated_at = datetime.now().isoformat()

    def clear(self) -> None:
        """Clear the scratchpad"""
        self.content = ""
        self.updated_at = datetime.now().isoformat()

    def is_empty(self) -> bool:
        """Check if scratchpad is empty

        Returns:
            True if scratchpad has no content
        """
        return not self.content.strip()


class ScratchpadManager:
    """Manages scratchpads for multiple clients"""

    def __init__(self):
        self.scratchpads: dict[str, Scratchpad] = {}  # client_id -> Scratchpad

    def get_scratchpad(self, client_id: str) -> Scratchpad:
        """Get or create a scratchpad for a client

        Args:
            client_id: The client identifier

        Returns:
            The Scratchpad object for this client
        """
        if client_id not in self.scratchpads:
            self.scratchpads[client_id] = Scratchpad()
        return self.scratchpads[client_id]

    def delete_scratchpad(self, client_id: str) -> bool:
        """Delete a client's scratchpad

        Args:
            client_id: The client identifier

        Returns:
            True if scratchpad was deleted, False if it didn't exist
        """
        if client_id in self.scratchpads:
            del self.scratchpads[client_id]
            return True
        return False
