"""
Test conversation management components.

Run with: python -m core.conversation.test_conversation
"""

import sys


def test_session():
    """Test conversation session."""
    print("\n[1/3] Testing ConversationSession...")

    try:
        from core.conversation.session import ConversationSession, SessionConfig

        # Create session
        config = SessionConfig(max_history_turns=5)
        session = ConversationSession(config=config)

        assert session.session_id is not None
        assert session.get_message_count() == 1  # System message

        # Add messages
        session.add_user_text("Hello")
        session.add_assistant_message("Hi there!")
        session.add_user_text("How are you?")
        session.add_assistant_message("I'm doing well!")

        assert session.get_turn_count() == 2
        assert session.get_message_count() == 5

        # Test clear
        session.clear()
        assert session.get_turn_count() == 0

        print(f"  ✓ Session ID: {session.session_id[:8]}...")
        print(f"  ✓ Message handling working")
        print(f"  ✓ Turn counting correct")
        print(f"  ✓ Clear working")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_session_manager():
    """Test session manager."""
    print("\n[2/3] Testing SessionManager...")

    try:
        from core.conversation.session import SessionManager

        manager = SessionManager(max_sessions=5)

        # Create sessions
        sessions = []
        for i in range(3):
            s = manager.create_session()
            sessions.append(s)

        assert manager.get_session_count() == 3

        # Get session
        retrieved = manager.get_session(sessions[0].session_id)
        assert retrieved is sessions[0]

        # Delete session
        manager.delete_session(sessions[0].session_id)
        assert manager.get_session_count() == 2

        # Test max sessions
        for i in range(10):
            manager.create_session()
        assert manager.get_session_count() <= 5

        print(f"  ✓ Create/retrieve sessions")
        print(f"  ✓ Delete sessions")
        print(f"  ✓ Max sessions limit enforced")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_context_manager():
    """Test context manager."""
    print("\n[3/3] Testing ContextManager...")

    try:
        from core.conversation.context import ContextManager, ContextConfig

        config = ContextConfig(max_tokens=1000)
        manager = ContextManager(config)

        # Test token estimation
        text = "Hello world this is a test message"
        tokens = manager.estimate_tokens(text)
        assert tokens > 0

        # Test message fitting
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]

        fitted = manager.fit_messages(messages)
        assert len(fitted) > 0

        print(f"  ✓ Token estimation: '{text[:20]}...' -> ~{tokens} tokens")
        print(f"  ✓ Message fitting: {len(messages)} -> {len(fitted)}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all conversation tests."""
    print("=" * 60)
    print("CONVERSATION MANAGEMENT TEST")
    print("=" * 60)

    results = []

    results.append(("Session", test_session()))
    results.append(("SessionManager", test_session_manager()))
    results.append(("ContextManager", test_context_manager()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
