"""
Shared test fixtures and configuration.
"""

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_audio():
    """Generate a sample audio array for testing."""
    duration = 1.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a simple sine wave
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def sample_audio_file(sample_audio):
    """Create a temporary audio file for testing."""
    import soundfile as sf

    audio, sr = sample_audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr)
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def test_config():
    """Create a test configuration."""
    from config import Config
    return Config(
        model_path="Qwen/Qwen2.5-Omni-3B",
        quantization="4bit",
        temperature=0.6,
        max_new_tokens=128,
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (no GPU required)")
    config.addinivalue_line("markers", "integration: Integration tests (requires GPU)")
    config.addinivalue_line("markers", "slow: Slow tests")
