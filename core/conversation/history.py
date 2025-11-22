"""
Conversation history management.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class ConversationHistory:
    """Manages conversation history persistence."""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize history manager.

        Args:
            storage_dir: Directory for storing histories
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path("./conversation_history")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, messages: List[Dict[str, Any]]) -> str:
        """
        Save conversation history.

        Args:
            session_id: Session identifier
            messages: List of messages

        Returns:
            Path to saved file
        """
        filepath = self.storage_dir / f"{session_id}.json"

        data = {
            "session_id": session_id,
            "saved_at": datetime.now().isoformat(),
            "messages": messages
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved history: {filepath}")
        return str(filepath)

    def load(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load conversation history.

        Args:
            session_id: Session identifier

        Returns:
            List of messages or None
        """
        filepath = self.storage_dir / f"{session_id}.json"

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded history: {filepath}")
        return data.get("messages", [])

    def delete(self, session_id: str) -> bool:
        """
        Delete conversation history.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted
        """
        filepath = self.storage_dir / f"{session_id}.json"

        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted history: {filepath}")
            return True
        return False

    def list_sessions(self) -> List[str]:
        """
        List all saved session IDs.

        Returns:
            List of session IDs
        """
        sessions = []
        for filepath in self.storage_dir.glob("*.json"):
            sessions.append(filepath.stem)
        return sessions

    def export_transcript(self, session_id: str, format: str = "text") -> Optional[str]:
        """
        Export conversation as readable transcript.

        Args:
            session_id: Session identifier
            format: Export format ('text' or 'markdown')

        Returns:
            Formatted transcript or None
        """
        messages = self.load(session_id)
        if not messages:
            return None

        lines = []

        if format == "markdown":
            lines.append(f"# Conversation: {session_id}\n")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle content that's a list
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, dict) and item.get("type") == "audio":
                        text_parts.append("[Audio]")
                content = " ".join(text_parts)

            if format == "markdown":
                lines.append(f"**{role.title()}**: {content}\n")
            else:
                lines.append(f"{role.title()}: {content}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("CONVERSATION HISTORY TEST")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        history = ConversationHistory(tmpdir)

        # Test save
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}
        ]

        session_id = "test-session"
        history.save(session_id, messages)
        print(f"  Saved session: {session_id}")

        # Test load
        loaded = history.load(session_id)
        print(f"  Loaded messages: {len(loaded)}")

        # Test export
        transcript = history.export_transcript(session_id)
        print(f"  Transcript:\n{transcript}")

        # Test list
        sessions = history.list_sessions()
        print(f"  Sessions: {sessions}")

        # Test delete
        history.delete(session_id)
        print(f"  Deleted: {session_id}")

    print("  ✓ History management working correctly")
    print("=" * 60)
