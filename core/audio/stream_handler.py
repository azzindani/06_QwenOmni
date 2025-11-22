"""
Audio stream handler for managing streaming audio buffers.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from collections import deque

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StreamConfig:
    """Configuration for stream handler."""
    sample_rate: int = 16000
    chunk_size: int = 512
    max_buffer_duration: float = 60.0  # seconds
    dtype: str = "float32"


class StreamHandler:
    """Manages streaming audio buffers."""

    def __init__(self, config: Optional[StreamConfig] = None):
        """
        Initialize stream handler.

        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        self._buffer: deque = deque()
        self._total_samples = 0
        self._max_samples = int(
            self.config.max_buffer_duration * self.config.sample_rate
        )

    def add_chunk(self, chunk: bytes) -> None:
        """
        Add audio chunk to buffer.

        Args:
            chunk: Raw audio bytes
        """
        # Convert bytes to numpy array
        dtype = np.float32 if self.config.dtype == "float32" else np.int16
        audio = np.frombuffer(chunk, dtype=dtype)

        # Convert int16 to float32 if needed
        if dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        self._buffer.append(audio)
        self._total_samples += len(audio)

        # Check for buffer overflow
        while self._total_samples > self._max_samples:
            removed = self._buffer.popleft()
            self._total_samples -= len(removed)
            logger.warning("Buffer overflow - removed oldest chunk")

    def add_array(self, audio: np.ndarray) -> None:
        """
        Add numpy array to buffer.

        Args:
            audio: Audio array
        """
        self._buffer.append(audio.astype(np.float32))
        self._total_samples += len(audio)

        # Check for buffer overflow
        while self._total_samples > self._max_samples:
            removed = self._buffer.popleft()
            self._total_samples -= len(removed)

    def get_complete_audio(self) -> Optional[np.ndarray]:
        """
        Get complete audio from buffer.

        Returns:
            Concatenated audio array or None if empty
        """
        if not self._buffer:
            return None

        return np.concatenate(list(self._buffer))

    def get_latest_chunk(self, duration_ms: int = 100) -> Optional[np.ndarray]:
        """
        Get latest audio chunk of specified duration.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Latest audio chunk or None
        """
        if not self._buffer:
            return None

        samples_needed = int((duration_ms / 1000) * self.config.sample_rate)
        audio = self.get_complete_audio()

        if audio is None or len(audio) < samples_needed:
            return audio

        return audio[-samples_needed:]

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_samples = 0
        logger.debug("Stream buffer cleared")

    def get_buffer_duration(self) -> float:
        """
        Get current buffer duration in seconds.

        Returns:
            Buffer duration
        """
        return self._total_samples / self.config.sample_rate

    def get_buffer_samples(self) -> int:
        """
        Get total samples in buffer.

        Returns:
            Number of samples
        """
        return self._total_samples

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0


if __name__ == "__main__":
    print("=" * 60)
    print("STREAM HANDLER TEST")
    print("=" * 60)

    handler = StreamHandler()
    print(f"  Sample Rate: {handler.config.sample_rate}")
    print(f"  Max Duration: {handler.config.max_buffer_duration}s")

    # Test adding chunks
    for i in range(10):
        chunk = np.random.randn(1600).astype(np.float32)  # 100ms chunks
        handler.add_array(chunk)

    duration = handler.get_buffer_duration()
    samples = handler.get_buffer_samples()
    print(f"  Buffer Duration: {duration:.2f}s")
    print(f"  Buffer Samples: {samples}")

    # Get complete audio
    audio = handler.get_complete_audio()
    print(f"  Complete Audio Shape: {audio.shape}")

    # Clear
    handler.clear()
    print(f"  After Clear: {handler.is_empty()}")

    print("  ✓ StreamHandler working correctly")
    print("=" * 60)
