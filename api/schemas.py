"""
API request/response schemas.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HealthResponse:
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    active_sessions: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionCreateRequest:
    """Request to create a new session."""
    system_prompt: Optional[str] = None
    max_history_turns: int = 20


@dataclass
class SessionResponse:
    """Session information response."""
    session_id: str
    created_at: str
    message_count: int
    turn_count: int


@dataclass
class AudioProcessRequest:
    """Request to process audio."""
    session_id: str
    audio_data: bytes
    sample_rate: int = 16000
    return_audio: bool = True


@dataclass
class AudioProcessResponse:
    """Response from audio processing."""
    session_id: str
    text: str
    audio_base64: Optional[str] = None
    sample_rate: int = 24000
    processing_time_ms: float = 0


@dataclass
class ErrorResponse:
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class WebSocketMessage:
    """WebSocket message format."""
    type: str  # 'audio_chunk', 'response', 'error', 'status'
    data: Any
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def to_dict(obj) -> Dict[str, Any]:
    """Convert dataclass to dictionary."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    else:
        return obj
