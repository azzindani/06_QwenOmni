"""
REST API routes.
"""

from typing import Optional
import json

from utils.logger import get_logger
from utils.hardware import detect_gpu
from config import get_config
from core.conversation.session import SessionManager, SessionConfig
from .schemas import (
    HealthResponse,
    SessionResponse,
    SessionCreateRequest,
    ErrorResponse,
    to_dict
)

logger = get_logger(__name__)


class APIRoutes:
    """REST API route handlers."""

    def __init__(self, session_manager: SessionManager, model_loaded_fn=None):
        """
        Initialize API routes.

        Args:
            session_manager: Session manager instance
            model_loaded_fn: Function to check if model is loaded
        """
        self.session_manager = session_manager
        self._model_loaded_fn = model_loaded_fn or (lambda: False)

    def health(self) -> dict:
        """
        Health check endpoint.

        Returns:
            Health status
        """
        response = HealthResponse(
            status="healthy",
            model_loaded=self._model_loaded_fn(),
            gpu_available=detect_gpu(),
            active_sessions=self.session_manager.get_session_count()
        )
        return to_dict(response)

    def get_config(self) -> dict:
        """
        Get current configuration.

        Returns:
            Configuration dict
        """
        config = get_config()
        return {
            "model_path": config.model_path,
            "quantization": config.quantization,
            "sample_rate_input": config.sample_rate_input,
            "sample_rate_output": config.sample_rate_output,
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
        }

    def create_session(self, request: Optional[SessionCreateRequest] = None) -> dict:
        """
        Create a new conversation session.

        Args:
            request: Session creation request

        Returns:
            Session info
        """
        config = None
        if request:
            config = SessionConfig(
                system_prompt=request.system_prompt or SessionConfig.system_prompt,
                max_history_turns=request.max_history_turns
            )

        session = self.session_manager.create_session(config)

        response = SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            message_count=session.get_message_count(),
            turn_count=session.get_turn_count()
        )
        return to_dict(response)

    def get_session(self, session_id: str) -> dict:
        """
        Get session information.

        Args:
            session_id: Session identifier

        Returns:
            Session info or error
        """
        session = self.session_manager.get_session(session_id)

        if not session:
            return to_dict(ErrorResponse(
                error="Session not found",
                detail=f"No session with ID: {session_id}"
            ))

        response = SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            message_count=session.get_message_count(),
            turn_count=session.get_turn_count()
        )
        return to_dict(response)

    def delete_session(self, session_id: str) -> dict:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        if self.session_manager.delete_session(session_id):
            return {"status": "deleted", "session_id": session_id}
        else:
            return to_dict(ErrorResponse(
                error="Session not found",
                detail=f"No session with ID: {session_id}"
            ))

    def list_sessions(self) -> dict:
        """
        List all active sessions.

        Returns:
            List of session IDs
        """
        return {
            "sessions": self.session_manager.list_sessions(),
            "count": self.session_manager.get_session_count()
        }

    def clear_session(self, session_id: str) -> dict:
        """
        Clear session history.

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        session = self.session_manager.get_session(session_id)

        if not session:
            return to_dict(ErrorResponse(
                error="Session not found",
                detail=f"No session with ID: {session_id}"
            ))

        session.clear()
        return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    print("=" * 60)
    print("API ROUTES TEST")
    print("=" * 60)

    manager = SessionManager()
    routes = APIRoutes(manager)

    # Test health
    health = routes.health()
    print(f"  Health: {health['status']}")

    # Test create session
    session = routes.create_session()
    print(f"  Created: {session['session_id'][:8]}...")

    # Test list sessions
    sessions = routes.list_sessions()
    print(f"  Sessions: {sessions['count']}")

    # Test delete
    result = routes.delete_session(session['session_id'])
    print(f"  Deleted: {result['status']}")

    print("  ✓ API routes working correctly")
    print("=" * 60)
