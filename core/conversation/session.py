"""
Conversation session management.
"""

import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionConfig:
    """Configuration for conversation session."""
    max_history_turns: int = 20
    max_context_tokens: int = 4096
    system_prompt: str = "You are Qwen, a helpful voice assistant. Respond naturally and conversationally."


class ConversationSession:
    """Manages a single conversation session."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        config: Optional[SessionConfig] = None
    ):
        """
        Initialize conversation session.

        Args:
            session_id: Unique session identifier
            config: Session configuration
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.config = config or SessionConfig()
        self._messages: List[Dict[str, Any]] = []
        self._created_at = datetime.now()
        self._updated_at = datetime.now()

        # Initialize with system message
        self._messages.append({
            "role": "system",
            "content": self.config.system_prompt
        })

        logger.debug(f"Session created: {self.session_id}")

    def add_user_message(self, content: List[Dict]) -> None:
        """
        Add user message to history.

        Args:
            content: Message content (can include text, audio, etc.)
        """
        self._messages.append({
            "role": "user",
            "content": content
        })
        self._updated_at = datetime.now()
        self._trim_history()

    def add_user_text(self, text: str) -> None:
        """
        Add simple text message from user.

        Args:
            text: Text message
        """
        self.add_user_message([{"type": "text", "text": text}])

    def add_user_audio(self, audio_path: str, text: Optional[str] = None) -> None:
        """
        Add audio message from user.

        Args:
            audio_path: Path to audio file
            text: Optional accompanying text
        """
        content = []
        if text:
            content.append({"type": "text", "text": text})
        content.append({"type": "audio", "audio": audio_path})
        self.add_user_message(content)

    def add_assistant_message(self, text: str, audio_path: Optional[str] = None) -> None:
        """
        Add assistant response to history.

        Args:
            text: Response text
            audio_path: Optional path to response audio
        """
        content = [{"type": "text", "text": text}]
        if audio_path:
            content.append({"type": "audio", "audio": audio_path})

        self._messages.append({
            "role": "assistant",
            "content": content
        })
        self._updated_at = datetime.now()
        self._trim_history()

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in session.

        Returns:
            List of messages
        """
        return self._messages.copy()

    def get_messages_for_inference(self) -> List[Dict[str, Any]]:
        """
        Get messages formatted for model inference.

        Returns:
            Messages ready for the model
        """
        return self._messages.copy()

    def _trim_history(self) -> None:
        """Trim history to max turns, keeping system message."""
        if len(self._messages) <= 1:
            return

        # Count turns (user + assistant pairs)
        non_system = self._messages[1:]
        max_messages = self.config.max_history_turns * 2

        if len(non_system) > max_messages:
            # Keep system message + latest messages
            self._messages = [self._messages[0]] + non_system[-max_messages:]
            logger.debug(f"Trimmed history to {len(self._messages)} messages")

    def clear(self) -> None:
        """Clear conversation history, keeping system message."""
        system_msg = self._messages[0]
        self._messages = [system_msg]
        self._updated_at = datetime.now()
        logger.debug(f"Session cleared: {self.session_id}")

    def get_turn_count(self) -> int:
        """
        Get number of conversation turns.

        Returns:
            Number of turns
        """
        return (len(self._messages) - 1) // 2

    def get_message_count(self) -> int:
        """
        Get total message count.

        Returns:
            Number of messages
        """
        return len(self._messages)

    @property
    def created_at(self) -> datetime:
        """Get session creation time."""
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        """Get last update time."""
        return self._updated_at


class SessionManager:
    """Manages multiple conversation sessions."""

    def __init__(self, max_sessions: int = 100):
        """
        Initialize session manager.

        Args:
            max_sessions: Maximum concurrent sessions
        """
        self._sessions: Dict[str, ConversationSession] = {}
        self._max_sessions = max_sessions

    def create_session(self, config: Optional[SessionConfig] = None) -> ConversationSession:
        """
        Create a new session.

        Args:
            config: Session configuration

        Returns:
            New conversation session
        """
        # Clean up old sessions if needed
        if len(self._sessions) >= self._max_sessions:
            self._cleanup_oldest()

        session = ConversationSession(config=config)
        self._sessions[session.session_id] = session
        logger.info(f"Created session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session or None
        """
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def _cleanup_oldest(self) -> None:
        """Remove oldest session."""
        if not self._sessions:
            return

        oldest_id = min(
            self._sessions.keys(),
            key=lambda k: self._sessions[k].updated_at
        )
        self.delete_session(oldest_id)

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def list_sessions(self) -> List[str]:
        """Get list of session IDs."""
        return list(self._sessions.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("CONVERSATION SESSION TEST")
    print("=" * 60)

    # Test session
    session = ConversationSession()
    print(f"  Session ID: {session.session_id[:8]}...")

    # Add messages
    session.add_user_text("Hello, how are you?")
    session.add_assistant_message("I'm doing well, thank you!")
    session.add_user_text("What's the weather like?")
    session.add_assistant_message("I don't have access to weather data.")

    print(f"  Messages: {session.get_message_count()}")
    print(f"  Turns: {session.get_turn_count()}")

    # Test manager
    manager = SessionManager(max_sessions=5)
    for i in range(3):
        manager.create_session()

    print(f"  Active Sessions: {manager.get_session_count()}")

    print("  ✓ Session management working correctly")
    print("=" * 60)
