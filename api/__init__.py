"""API layer components."""

def __getattr__(name):
    if name == "APIRoutes":
        from .routes import APIRoutes
        return APIRoutes
    if name == "WebSocketHandler":
        from .websocket import WebSocketHandler
        return WebSocketHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["APIRoutes", "WebSocketHandler"]
