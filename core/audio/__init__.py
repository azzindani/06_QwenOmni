"""Audio processing components."""

def __getattr__(name):
    if name == "AudioPreprocessor":
        from .preprocessor import AudioPreprocessor
        return AudioPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AudioPreprocessor"]
