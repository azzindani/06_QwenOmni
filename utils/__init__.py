"""Utility functions and helpers."""

from .logger import get_logger
from .hardware import detect_gpu, get_device_info
from .audio_utils import load_audio, save_audio

# Lazy imports for heavier modules
def __getattr__(name):
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "get_logger", "detect_gpu", "get_device_info", "load_audio", "save_audio",
    "get_tracker", "PerformanceTracker", "LRUCache",
    "get_analytics", "AnalyticsTracker",
    "get_language_config", "list_languages", "get_system_prompt"
]
