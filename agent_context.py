"""
agent context system - memory and state management
- remember what object is being repaired
- track repair progress across frames
- store detected objects from previous frames
- maintain safety warnings state

"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RepairContext:
    """
    Stores the repair session state for a single user/device.

    This context persists across camera frames so the AI can provide
    coherent, step-by-step guidance without forgetting what's happening.
    """

    # Session identification
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Repair object information
    object_type: str | None = None  # e.g., "light fixture", "faucet", "door hinge"
    object_detected_at: datetime | None = None

    # Repair progress tracking
    current_step: int = 0  # 0 means not started, 1 means on step 1, etc.
    total_steps: int = 0
    completed_steps: list[int] = field(default_factory=list)  # List of completed step numbers
    repair_steps: list[str] = field(default_factory=list)  # List of instruction strings

    # Vision/detection history
    last_detected_objects: list[str] = field(default_factory=list)  # Objects seen in last frame
    detection_history: dict[str, int] = field(default_factory=dict)  # Object -> count of times seen

    # Safety and warnings
    safety_warning_shown: bool = False
    warnings_shown: list[str] = field(default_factory=list)  # Track which warnings were shown

    # Additional metadata
    custom_data: dict[str, Any] = field(default_factory=dict)  # For extensibility

    def update_timestamp(self) -> None:
        """Update the last_updated timestamp to now"""
        self.last_updated = datetime.now()

    def is_repair_started(self) -> bool:
        """Check if a repair session has been initiated"""
        return self.object_type is not None and len(self.repair_steps) > 0

    def is_repair_complete(self) -> bool:
        """Check if all repair steps are completed"""
        if not self.is_repair_started():
            return False
        return len(self.completed_steps) >= self.total_steps

    def get_current_step_instruction(self) -> str | None:
        """Get the instruction for the current step"""
        if self.current_step <= 0 or self.current_step > len(self.repair_steps):
            return None
        return self.repair_steps[self.current_step - 1]  # Convert to 0-indexed

    def get_next_step_instruction(self) -> str | None:
        """Get the instruction for the next step"""
        next_step = self.current_step + 1
        if next_step > len(self.repair_steps):
            return None
        return self.repair_steps[next_step - 1]  # Convert to 0-indexed


class ContextManager:
    """
    Manages multiple repair contexts across different user sessions.

    Uses in-memory storage (Python dict) - suitable for hackathon/prototype.
    In production, you'd want to persist this to Redis or a database.
    """

    def __init__(self):
        # Dictionary mapping session_id -> RepairContext
        self._contexts: dict[str, RepairContext] = {}

    def get_context(self, session_id: str) -> RepairContext:
        """
        Get or create a repair context for a given session ID.

        Args:
            session_id: Unique identifier for the user/device session

        Returns:
            RepairContext object for this session
        """
        if session_id not in self._contexts:
            self._contexts[session_id] = RepairContext(session_id=session_id)

        # Update timestamp on access
        self._contexts[session_id].update_timestamp()
        return self._contexts[session_id]

    def start_repair(
        self,
        session_id: str,
        object_type: str,
        repair_steps: list[str],
    ) -> RepairContext:
        """
        Initialize a new repair session with object type and steps.

        Args:
            session_id: Session identifier
            object_type: Type of object being repaired (e.g., "light fixture")
            repair_steps: List of step-by-step instructions

        Returns:
            Updated RepairContext
        """
        context = self.get_context(session_id)
        context.object_type = object_type
        context.object_detected_at = datetime.now()
        context.repair_steps = repair_steps
        context.total_steps = len(repair_steps)
        context.current_step = 1  # Start on step 1
        context.update_timestamp()
        return context

    def mark_step_complete(self, session_id: str) -> RepairContext:
        """
        Mark the current step as complete and advance to next step.

        Args:
            session_id: Session identifier

        Returns:
            Updated RepairContext
        """
        context = self.get_context(session_id)

        # Mark current step as completed (if not already)
        if context.current_step not in context.completed_steps:
            context.completed_steps.append(context.current_step)

        # Advance to next step (if not at the end)
        if context.current_step < context.total_steps:
            context.current_step += 1

        context.update_timestamp()
        return context

    def update_detected_objects(
        self,
        session_id: str,
        objects: list[str],
    ) -> RepairContext:
        """
        Update the list of detected objects in the current frame.

        Args:
            session_id: Session identifier
            objects: List of object labels detected in current frame

        Returns:
            Updated RepairContext
        """
        context = self.get_context(session_id)
        context.last_detected_objects = objects

        # Update detection history (count how many times each object was seen)
        for obj in objects:
            context.detection_history[obj] = context.detection_history.get(obj, 0) + 1

        context.update_timestamp()
        return context

    def add_safety_warning(self, session_id: str, warning: str) -> RepairContext:
        """
        Record that a safety warning was shown to the user.

        Args:
            session_id: Session identifier
            warning: Warning message that was shown

        Returns:
            Updated RepairContext
        """
        context = self.get_context(session_id)

        if warning not in context.warnings_shown:
            context.warnings_shown.append(warning)

        context.safety_warning_shown = True
        context.update_timestamp()
        return context

    def reset_session(self, session_id: str) -> RepairContext:
        """
        Reset a session to start a new repair.

        Args:
            session_id: Session identifier

        Returns:
            Fresh RepairContext
        """
        self._contexts[session_id] = RepairContext(session_id=session_id)
        return self._contexts[session_id]

    def get_all_sessions(self) -> dict[str, RepairContext]:
        """Get all active sessions (useful for debugging)"""
        return self._contexts.copy()

    def cleanup_old_sessions(self, max_age_minutes: int = 60) -> int:
        """
        Remove sessions that haven't been accessed in a while.

        Args:
            max_age_minutes: Sessions older than this are removed

        Returns:
            Number of sessions removed
        """
        now = datetime.now()
        sessions_to_remove = []

        for session_id, context in self._contexts.items():
            age_minutes = (now - context.last_updated).total_seconds() / 60
            if age_minutes > max_age_minutes:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self._contexts[session_id]

        return len(sessions_to_remove)


# Global context manager instance
# This persists across API requests (singleton pattern)
context_manager = ContextManager()
