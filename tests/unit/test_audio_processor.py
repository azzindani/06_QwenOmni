"""
Unit tests for audio processing.
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class."""

    def test_init(self, test_config):
        """Test preprocessor initialization."""
        from core.audio.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor(test_config)

        assert preprocessor.target_sr == 16000

    def test_normalize_audio(self, test_config):
        """Test audio normalization."""
        from core.audio.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor(test_config)

        # Create audio with values outside [-1, 1]
        audio = np.array([0.0, 2.0, -2.0, 1.0], dtype=np.float32)
        normalized = preprocessor._normalize(audio)

        assert normalized.max() <= 1.0
        assert normalized.min() >= -1.0

    def test_to_mono(self, test_config):
        """Test stereo to mono conversion."""
        from core.audio.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor(test_config)

        # Create stereo audio
        stereo = np.array([[1.0, 0.5], [0.0, 1.0], [-1.0, -0.5]], dtype=np.float32)
        mono = preprocessor._to_mono(stereo)

        assert len(mono.shape) == 1
        assert len(mono) == 3

    def test_mono_passthrough(self, test_config):
        """Test mono audio passes through unchanged."""
        from core.audio.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor(test_config)

        mono = np.array([1.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float32)
        result = preprocessor._to_mono(mono)

        np.testing.assert_array_equal(mono, result)

    def test_process_pipeline(self, test_config, sample_audio):
        """Test full processing pipeline."""
        from core.audio.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor(test_config)
        audio, sr = sample_audio

        processed = preprocessor.process(audio, sr)

        assert processed is not None
        assert len(processed.shape) == 1
        assert processed.max() <= 1.0
        assert processed.min() >= -1.0

    def test_duration_limit(self, test_config):
        """Test audio duration limiting."""
        from core.audio.preprocessor import AudioPreprocessor

        # Set short max duration
        test_config.max_audio_duration = 0.5
        preprocessor = AudioPreprocessor(test_config)

        # Create 2 second audio
        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32)

        processed = preprocessor.process(audio, sr)

        # Should be truncated to 0.5 seconds
        max_samples = int(0.5 * sr)
        assert len(processed) <= max_samples


@pytest.mark.unit
class TestAudioUtils:
    """Tests for audio utility functions."""

    def test_get_audio_duration(self):
        """Test duration calculation."""
        from utils.audio_utils import get_audio_duration

        sr = 16000
        audio = np.zeros(sr * 3)  # 3 seconds

        duration = get_audio_duration(audio, sr)

        assert duration == 3.0

    def test_normalize_audio(self):
        """Test audio normalization function."""
        from utils.audio_utils import normalize_audio

        audio = np.array([0.0, 4.0, -2.0], dtype=np.float32)
        normalized = normalize_audio(audio)

        assert normalized.max() == 1.0
        assert normalized.min() == -0.5

    def test_convert_to_mono(self):
        """Test stereo to mono conversion."""
        from utils.audio_utils import convert_to_mono

        stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        mono = convert_to_mono(stereo)

        assert len(mono.shape) == 1
        np.testing.assert_array_almost_equal(mono, [0.5, 0.5])

    def test_save_and_load_audio(self, sample_audio):
        """Test saving and loading audio files."""
        import tempfile
        import os
        from utils.audio_utils import save_audio, load_audio

        audio, sr = sample_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Save
            save_audio(audio, temp_path, sr)
            assert os.path.exists(temp_path)

            # Load
            loaded, loaded_sr = load_audio(temp_path, target_sr=sr)
            assert loaded_sr == sr
            assert len(loaded) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
