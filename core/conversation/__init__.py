"""Conversation management components."""

def __getattr__(name):
    if name == "ConversationSession":
        from .session import ConversationSession
        return ConversationSession
    if name == "SessionManager":
        from .session import SessionManager
        return SessionManager
    if name == "SessionConfig":
        from .session import SessionConfig
        return SessionConfig
    if name == "ConversationHistory":
        from .history import ConversationHistory
        return ConversationHistory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ConversationSession", "SessionManager", "SessionConfig", "ConversationHistory"]
