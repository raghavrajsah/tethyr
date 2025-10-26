"""
Email supervisor tool for sending help requests
Allows Gemini to escalate to human supervisor when user needs more assistance
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from loguru import logger


class EmailSupervisor:
    """Email supervisor for escalating user requests to human help

    Architecture:
    - Sends email notifications when user needs human assistance
    - Uses SMTP to send emails
    - Thread-safe for concurrent access from multiple clients
    """

    def __init__(
        self,
        supervisor_email: str = "bzhao2@caltech.edu",
        smtp_server: str | None = None,
        smtp_port: int = 587,
    ):
        """Initialize the email supervisor

        Args:
            supervisor_email: Email address of the human supervisor
            smtp_server: SMTP server address (defaults to Gmail)
            smtp_port: SMTP server port (default: 587 for TLS)
        """
        self.supervisor_email = supervisor_email
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port

        # Get email credentials from environment
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")

        if not self.sender_email or not self.sender_password:
            logger.warning(
                "Email credentials not set. Set SENDER_EMAIL and SENDER_PASSWORD "
                "environment variables to enable email notifications."
            )

        logger.info(f"Initialized EmailSupervisor with supervisor: {self.supervisor_email}")

    def send_help_request(
        self,
        client_id: str,
        user_request: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Send an email to the supervisor requesting help

        Args:
            client_id: ID of the client requesting help
            user_request: The user's help request or issue description
            context: Optional additional context about the situation

        Returns:
            Dict with status and message
        """
        try:
            if not self.sender_email or not self.sender_password:
                logger.error("Cannot send email: credentials not configured")
                return {
                    "status": "error",
                    "message": "Email credentials not configured on server",
                }

            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = self.supervisor_email
            msg["Subject"] = f"AR Glasses Help Request - Client {client_id}"

            # Email body
            body = f"""
AR Glasses Help Request

Client ID: {client_id}
User Request: {user_request}
"""
            if context:
                body += f"\nAdditional Context:\n{context}\n"

            body += "\n---\nThis is an automated message from the AR Glasses assistance system."

            msg.attach(MIMEText(body, "plain"))

            # Send email
            logger.info(f"Sending help request email for client {client_id}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Successfully sent help request email for client {client_id}")
            return {
                "status": "success",
                "message": f"Help request sent to {self.supervisor_email}. A human supervisor will assist you shortly.",
            }

        except Exception as e:
            logger.opt(exception=e).error(f"Error sending help request email for client {client_id}")
            return {
                "status": "error",
                "message": f"Failed to send help request: {str(e)}",
            }


# Tool definition for Gemini Live API - Sends email to supervisor
EMAIL_SUPERVISOR_TOOL_DECLARATION = {
    "name": "request_human_help",
    "description": """Request help from a human supervisor when the user needs assistance beyond your capabilities.

    Use this when:
    - The user explicitly asks to speak with a human or supervisor
    - The task is too complex or dangerous for AR guidance alone
    - The user is frustrated or the current approach isn't working
    - You need expert human judgment for safety or quality reasons

    Do NOT use this for:
    - Simple clarification questions you can answer
    - Normal step-by-step guidance within your capabilities
    - Minor user confusion that can be resolved with better explanation""",
    "parameters": {
        "type": "object",
        "properties": {
            "user_request": {
                "type": "string",
                "description": "Summary of what the user needs help with",
            },
            "context": {
                "type": "string",
                "description": "Optional additional context about the situation, current task, or why human help is needed",
            },
        },
        "required": ["user_request"],
    },
}
