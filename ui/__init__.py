"""User interface components."""

def __getattr__(name):
    if name == "GradioApp":
        from .gradio_app import GradioApp
        return GradioApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["GradioApp"]
