import os
from typing import Any

from loguru import logger
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

SLACK_TOOL_DECLARATION = {
    "name": "send_slack_message",
    "description": """Send a message to a team member via Slack.

Use this when the user asks to:
- Send a message to someone
- Notify a colleague
- Ask someone for help
- Contact a team member

The recipient should be specified by their display name (e.g., "John Smith", "Jane Doe"),
not by user ID or email. The system will look up the correct user.""",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient_name": {
                "type": "string",
                "description": "The display name of the person to send the message to (e.g., 'John Smith', 'Jane Doe')",
            },
            "message": {
                "type": "string",
                "description": "The message content to send",
            },
        },
        "required": ["recipient_name", "message"],
    },
}


class SlackBot:
    """Handles Slack messaging operations with user lookup by display name."""

    def __init__(self, token: str | None = None):
        """Initialize Slack bot with optional token.

        Args:
            token: Slack bot token. If not provided, reads from SLACK_BOT_TOKEN env var.
        """
        self.token = token or os.getenv("SLACK_BOT_TOKEN")
        self.client: AsyncWebClient | None = None

        if self.token:
            self.client = AsyncWebClient(token=self.token)
            logger.info("Slack bot initialized successfully")
        else:
            logger.warning("SLACK_BOT_TOKEN not set - Slack messaging will be disabled")

    @property
    def is_enabled(self) -> bool:
        """Check if Slack bot is properly configured."""
        return self.client is not None

    async def send_message(self, recipient_name: str, message: str) -> dict[str, Any]:
        """Send a message to a user by their display name.

        Args:
            recipient_name: Display name of the recipient (e.g., "John Smith")
            message: Message content to send

        Returns:
            Dict with status and message about the operation
        """
        if not self.client:
            return {
                "status": "error",
                "message": "Slack integration not configured. Please set SLACK_BOT_TOKEN environment variable.",
            }

        try:
            recipient_name = recipient_name.strip()
            message = message.strip()

            if not recipient_name or not message:
                return {
                    "status": "error",
                    "message": "Both recipient_name and message are required",
                }

            logger.info(
                f"Attempting to send Slack message to '{recipient_name}': '{message[:50]}...'"
            )

            # Look up user by display name
            user_id = await self._find_user_by_display_name(recipient_name)

            if not user_id:
                return {
                    "status": "error",
                    "message": f"Could not find Slack user with display name '{recipient_name}'. "
                    "Please check the name and try again.",
                }

            # Send the message via DM
            response = await self.client.chat_postMessage(
                channel=user_id,  # Sending to user ID opens a DM
                text=message,
            )

            if response["ok"]:
                logger.info(
                    f"Successfully sent Slack message to {recipient_name} ({user_id})"
                )
                return {
                    "status": "success",
                    "message": f"Message sent to {recipient_name}",
                    "recipient": recipient_name,
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to send message: {response.get('error', 'Unknown error')}",
                }

        except SlackApiError as e:
            error_msg = e.response.get("error", str(e))
            logger.opt(exception=e).error("Slack API error")
            return {
                "status": "error",
                "message": f"Slack API error: {error_msg}",
            }
        except Exception as e:
            logger.opt(exception=e).error("Error sending Slack message")
            return {
                "status": "error",
                "message": f"Error sending message: {str(e)}",
            }

    async def _find_user_by_display_name(self, display_name: str) -> str | None:
        """Find a Slack user ID by their display name.

        Tries multiple matching strategies:
        1. Exact match on display_name
        2. Exact match on real_name (full name)
        3. Exact match on display_name_normalized
        4. Partial match as fallback

        Args:
            display_name: The display name to search for (case-insensitive)

        Returns:
            User ID if found, None otherwise
        """
        if not self.client:
            return None

        try:
            search_name = display_name.lower().strip()

            # Get all users in the workspace
            response = await self.client.users_list()

            if not response["ok"]:
                logger.error(f"Failed to fetch Slack users: {response.get('error')}")
                return None

            users = response.get("members", [])

            # Try multiple matching strategies
            for user in users:
                # Skip bots and deleted users
                if user.get("is_bot") or user.get("deleted"):
                    continue

                profile = user.get("profile", {})
                user_id = user.get("id")

                # Strategy 1: Match display name (most common)
                display_name_field = profile.get("display_name", "").lower().strip()
                if display_name_field == search_name:
                    logger.debug(
                        f"Found user by display_name: {display_name_field} -> {user_id}"
                    )
                    return user_id

                # Strategy 2: Match real name (full name)
                real_name = profile.get("real_name", "").lower().strip()
                if real_name == search_name:
                    logger.debug(f"Found user by real_name: {real_name} -> {user_id}")
                    return user_id

                # Strategy 3: Match display name without display_name_normalized
                display_name_normalized = (
                    profile.get("display_name_normalized", "").lower().strip()
                )
                if display_name_normalized == search_name:
                    logger.debug(
                        f"Found user by display_name_normalized: {display_name_normalized} -> {user_id}"
                    )
                    return user_id

            # No exact match found, try partial matches
            for user in users:
                if user.get("is_bot") or user.get("deleted"):
                    continue

                profile = user.get("profile", {})
                user_id = user.get("id")

                # Check if search term is contained in any name field
                display_name_field = profile.get("display_name", "").lower()
                real_name = profile.get("real_name", "").lower()

                if search_name in display_name_field or search_name in real_name:
                    logger.debug(
                        f"Found user by partial match: {display_name_field or real_name} -> {user_id}"
                    )
                    return user_id

            logger.warning(f"No Slack user found matching '{display_name}'")
            return None

        except SlackApiError as e:
            logger.opt(exception=e).error(
                f"Error looking up Slack user '{display_name}'"
            )
            return None
        except Exception as e:
            logger.opt(exception=e).error(
                f"Unexpected error looking up Slack user '{display_name}'"
            )
            return None
