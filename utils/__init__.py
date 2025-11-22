"""Utility functions and helpers."""

from .logger import get_logger

# Lazy imports for modules with heavy dependencies
def __getattr__(name):
    if name == "detect_gpu":
        from .hardware import detect_gpu
        return detect_gpu
    if name == "get_device_info":
        from .hardware import get_device_info
        return get_device_info
    if name == "load_audio":
        from .audio_utils import load_audio
        return load_audio
    if name == "save_audio":
        from .audio_utils import save_audio
        return save_audio
    if name == "get_tracker":
        from .performance import get_tracker
        return get_tracker
    if name == "PerformanceTracker":
        from .performance import PerformanceTracker
        return PerformanceTracker
    if name == "LRUCache":
        from .performance import LRUCache
        return LRUCache
    if name == "get_analytics":
        from .analytics import get_analytics
        return get_analytics
    if name == "AnalyticsTracker":
        from .analytics import AnalyticsTracker
        return AnalyticsTracker
    if name == "get_language_config":
        from .languages import get_language_config
        return get_language_config
    if name == "list_languages":
        from .languages import list_languages
        return list_languages
    if name == "get_system_prompt":
        from .languages import get_system_prompt
        return get_system_prompt
    if name == "get_auth_manager":
        from .auth import get_auth_manager
        return get_auth_manager
    if name == "AuthManager":
        from .auth import AuthManager
        return AuthManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "get_logger", "detect_gpu", "get_device_info", "load_audio", "save_audio",
    "get_tracker", "PerformanceTracker", "LRUCache",
    "get_analytics", "AnalyticsTracker",
    "get_language_config", "list_languages", "get_system_prompt",
    "get_auth_manager", "AuthManager"
]
