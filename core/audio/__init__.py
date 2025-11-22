"""Audio processing components."""

def __getattr__(name):
    if name == "AudioPreprocessor":
        from .preprocessor import AudioPreprocessor
        return AudioPreprocessor
    if name == "VADDetector":
        from .vad import VADDetector
        return VADDetector
    if name == "VADConfig":
        from .vad import VADConfig
        return VADConfig
    if name == "StreamHandler":
        from .stream_handler import StreamHandler
        return StreamHandler
    if name == "StreamConfig":
        from .stream_handler import StreamConfig
        return StreamConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AudioPreprocessor", "VADDetector", "VADConfig", "StreamHandler", "StreamConfig"]
