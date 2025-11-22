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
    if name == "ContextManager":
        from .context import ContextManager
        return ContextManager
    if name == "ContextConfig":
        from .context import ContextConfig
        return ContextConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ConversationSession", "SessionManager", "SessionConfig",
    "ConversationHistory", "ContextManager", "ContextConfig"
]
