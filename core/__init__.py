"""Core business logic for Qwen Omni Voice Assistant."""

def __getattr__(name):
    if name == "ModelManager":
        from .inference.model_manager import ModelManager
        return ModelManager
    if name == "ResponseGenerator":
        from .inference.generator import ResponseGenerator
        return ResponseGenerator
    if name == "AudioPreprocessor":
        from .audio.preprocessor import AudioPreprocessor
        return AudioPreprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ModelManager", "ResponseGenerator", "AudioPreprocessor"]
