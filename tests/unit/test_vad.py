"""
Unit tests for VAD detector.
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestVADConfig:
    """Tests for VADConfig."""

    def test_default_config(self):
        """Test default VAD configuration."""
        from core.audio.vad import VADConfig

        config = VADConfig()

        assert config.threshold == 0.5
        assert config.min_speech_duration_ms == 250
        assert config.min_silence_duration_ms == 800
        assert config.sample_rate == 16000

    def test_custom_config(self):
        """Test custom VAD configuration."""
        from core.audio.vad import VADConfig

        config = VADConfig(
            threshold=0.7,
            min_silence_duration_ms=1000
        )

        assert config.threshold == 0.7
        assert config.min_silence_duration_ms == 1000


@pytest.mark.unit
class TestVADDetector:
    """Tests for VADDetector class."""

    def test_detector_init(self):
        """Test VAD detector initialization."""
        from core.audio.vad import VADDetector, VADConfig

        config = VADConfig()
        detector = VADDetector(config)

        assert detector.config == config
        assert not detector._loaded

    def test_reset(self):
        """Test detector reset."""
        from core.audio.vad import VADDetector

        detector = VADDetector()
        detector._speech_started = True
        detector._silence_samples = 1000
        detector._speech_buffer = [np.zeros(100)]

        detector.reset()

        assert not detector._speech_started
        assert detector._silence_samples == 0
        assert len(detector._speech_buffer) == 0

    def test_is_speech_ended_no_speech(self):
        """Test is_speech_ended when no speech started."""
        from core.audio.vad import VADDetector

        detector = VADDetector()

        assert not detector.is_speech_ended()

    def test_get_speech_audio_empty(self):
        """Test get_speech_audio when buffer is empty."""
        from core.audio.vad import VADDetector

        detector = VADDetector()

        assert detector.get_speech_audio() is None

    def test_get_speech_audio_with_data(self):
        """Test get_speech_audio with buffered data."""
        from core.audio.vad import VADDetector

        detector = VADDetector()
        chunk1 = np.ones(1000, dtype=np.float32)
        chunk2 = np.ones(1000, dtype=np.float32) * 0.5

        detector._speech_buffer = [chunk1, chunk2]

        audio = detector.get_speech_audio()

        assert audio is not None
        # Note: may be trimmed due to silence removal


@pytest.mark.unit
class TestStreamHandler:
    """Tests for StreamHandler class."""

    def test_handler_init(self):
        """Test stream handler initialization."""
        from core.audio.stream_handler import StreamHandler

        handler = StreamHandler()

        assert handler.config.sample_rate == 16000
        assert handler.is_empty()

    def test_add_array(self):
        """Test adding numpy array."""
        from core.audio.stream_handler import StreamHandler

        handler = StreamHandler()
        audio = np.random.randn(1600).astype(np.float32)

        handler.add_array(audio)

        assert not handler.is_empty()
        assert handler.get_buffer_samples() == 1600

    def test_get_complete_audio(self):
        """Test getting complete audio."""
        from core.audio.stream_handler import StreamHandler

        handler = StreamHandler()

        # Add multiple chunks
        for _ in range(5):
            handler.add_array(np.random.randn(1600).astype(np.float32))

        audio = handler.get_complete_audio()

        assert audio is not None
        assert len(audio) == 8000

    def test_clear_buffer(self):
        """Test clearing buffer."""
        from core.audio.stream_handler import StreamHandler

        handler = StreamHandler()
        handler.add_array(np.random.randn(1600).astype(np.float32))
        handler.clear()

        assert handler.is_empty()
        assert handler.get_buffer_samples() == 0

    def test_buffer_duration(self):
        """Test buffer duration calculation."""
        from core.audio.stream_handler import StreamHandler

        handler = StreamHandler()
        # Add 1 second of audio (16000 samples at 16kHz)
        handler.add_array(np.random.randn(16000).astype(np.float32))

        duration = handler.get_buffer_duration()

        assert abs(duration - 1.0) < 0.001

    def test_get_latest_chunk(self):
        """Test getting latest chunk."""
        from core.audio.stream_handler import StreamHandler

        handler = StreamHandler()
        handler.add_array(np.random.randn(16000).astype(np.float32))

        chunk = handler.get_latest_chunk(duration_ms=100)

        assert chunk is not None
        assert len(chunk) == 1600  # 100ms at 16kHz
