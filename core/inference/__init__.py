"""Inference components for Qwen Omni."""

def __getattr__(name):
    if name == "ModelManager":
        from .model_manager import ModelManager
        return ModelManager
    if name == "ResponseGenerator":
        from .generator import ResponseGenerator
        return ResponseGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ModelManager", "ResponseGenerator"]
