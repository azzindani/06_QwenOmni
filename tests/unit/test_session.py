"""
Unit tests for conversation session management.
"""

import pytest


@pytest.mark.unit
class TestConversationSession:
    """Tests for ConversationSession class."""

    def test_session_creation(self):
        """Test session creation."""
        from core.conversation.session import ConversationSession

        session = ConversationSession()

        assert session.session_id is not None
        assert len(session.session_id) > 0
        assert session.get_message_count() == 1  # System message

    def test_session_with_custom_id(self):
        """Test session with custom ID."""
        from core.conversation.session import ConversationSession

        session = ConversationSession(session_id="test-123")

        assert session.session_id == "test-123"

    def test_add_user_text(self):
        """Test adding user text message."""
        from core.conversation.session import ConversationSession

        session = ConversationSession()
        session.add_user_text("Hello")

        assert session.get_message_count() == 2
        messages = session.get_messages()
        assert messages[-1]["role"] == "user"

    def test_add_assistant_message(self):
        """Test adding assistant message."""
        from core.conversation.session import ConversationSession

        session = ConversationSession()
        session.add_user_text("Hello")
        session.add_assistant_message("Hi there!")

        assert session.get_message_count() == 3
        assert session.get_turn_count() == 1

    def test_turn_count(self):
        """Test turn counting."""
        from core.conversation.session import ConversationSession

        session = ConversationSession()

        # Add 3 turns
        for i in range(3):
            session.add_user_text(f"Message {i}")
            session.add_assistant_message(f"Response {i}")

        assert session.get_turn_count() == 3

    def test_clear_session(self):
        """Test clearing session."""
        from core.conversation.session import ConversationSession

        session = ConversationSession()
        session.add_user_text("Hello")
        session.add_assistant_message("Hi!")
        session.clear()

        assert session.get_message_count() == 1  # Only system message
        assert session.get_turn_count() == 0

    def test_history_trimming(self):
        """Test history trimming to max turns."""
        from core.conversation.session import ConversationSession, SessionConfig

        config = SessionConfig(max_history_turns=2)
        session = ConversationSession(config=config)

        # Add 5 turns
        for i in range(5):
            session.add_user_text(f"Message {i}")
            session.add_assistant_message(f"Response {i}")

        # Should be trimmed to 2 turns (4 messages) + system
        assert session.get_message_count() <= 5

    def test_get_messages_copy(self):
        """Test that get_messages returns a copy."""
        from core.conversation.session import ConversationSession

        session = ConversationSession()
        messages1 = session.get_messages()
        messages2 = session.get_messages()

        assert messages1 is not messages2


@pytest.mark.unit
class TestSessionManager:
    """Tests for SessionManager class."""

    def test_create_session(self):
        """Test session creation through manager."""
        from core.conversation.session import SessionManager

        manager = SessionManager()
        session = manager.create_session()

        assert session is not None
        assert manager.get_session_count() == 1

    def test_get_session(self):
        """Test retrieving session."""
        from core.conversation.session import SessionManager

        manager = SessionManager()
        session = manager.create_session()
        retrieved = manager.get_session(session.session_id)

        assert retrieved is session

    def test_get_nonexistent_session(self):
        """Test retrieving non-existent session."""
        from core.conversation.session import SessionManager

        manager = SessionManager()
        result = manager.get_session("nonexistent")

        assert result is None

    def test_delete_session(self):
        """Test session deletion."""
        from core.conversation.session import SessionManager

        manager = SessionManager()
        session = manager.create_session()
        result = manager.delete_session(session.session_id)

        assert result is True
        assert manager.get_session_count() == 0

    def test_max_sessions_limit(self):
        """Test max sessions limit."""
        from core.conversation.session import SessionManager

        manager = SessionManager(max_sessions=3)

        # Create 5 sessions
        for _ in range(5):
            manager.create_session()

        # Should be limited to 3
        assert manager.get_session_count() <= 3

    def test_list_sessions(self):
        """Test listing session IDs."""
        from core.conversation.session import SessionManager

        manager = SessionManager()
        session1 = manager.create_session()
        session2 = manager.create_session()

        sessions = manager.list_sessions()

        assert session1.session_id in sessions
        assert session2.session_id in sessions
