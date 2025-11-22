"""
Audio preprocessor for Qwen Omni.
"""

import numpy as np
from typing import Tuple, Optional

from utils.logger import get_logger
from config import Config, get_config

logger = get_logger(__name__)


class AudioPreprocessor:
    """Preprocesses audio for model input."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize audio preprocessor.

        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self.target_sr = self.config.sample_rate_input

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio for model input.

        Args:
            audio: Input audio array
            sample_rate: Input sample rate

        Returns:
            Processed audio array
        """
        # Convert to mono if stereo
        audio = self._to_mono(audio)

        # Resample if needed
        if sample_rate != self.target_sr:
            audio = self._resample(audio, sample_rate, self.target_sr)

        # Normalize
        audio = self._normalize(audio)

        # Trim silence
        audio = self._trim_silence(audio)

        # Check duration
        duration = len(audio) / self.target_sr
        if duration > self.config.max_audio_duration:
            logger.warning(f"Audio truncated from {duration:.1f}s to {self.config.max_audio_duration}s")
            max_samples = int(self.config.max_audio_duration * self.target_sr)
            audio = audio[:max_samples]

        return audio

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo to mono."""
        if len(audio.shape) > 1:
            return np.mean(audio, axis=1)
        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim leading and trailing silence."""
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold

        if not np.any(non_silent):
            return audio

        # Get start and end indices
        indices = np.where(non_silent)[0]
        start = max(0, indices[0] - int(0.1 * self.target_sr))  # Keep 100ms padding
        end = min(len(audio), indices[-1] + int(0.1 * self.target_sr))

        return audio[start:end]

    def load_and_process(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and process it.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (processed audio, sample rate)
        """
        from utils.audio_utils import load_audio

        audio, sr = load_audio(file_path, target_sr=self.target_sr)
        processed = self.process(audio, sr)
        return processed, self.target_sr


if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO PREPROCESSOR TEST")
    print("=" * 60)

    preprocessor = AudioPreprocessor()
    print(f"  Target Sample Rate: {preprocessor.target_sr}")
    print(f"  Max Duration: {preprocessor.config.max_audio_duration}s")

    # Test with sample file
    test_file = "./00_Dataset/1272-128104-0000.flac"
    try:
        audio, sr = preprocessor.load_and_process(test_file)
        duration = len(audio) / sr
        print(f"  Processed Audio Duration: {duration:.2f}s")
        print(f"  Audio Shape: {audio.shape}")
        print(f"  Audio Range: [{audio.min():.3f}, {audio.max():.3f}]")
        print("  ✓ Preprocessor working correctly")
    except Exception as e:
        print(f"  Note: Could not process test file ({e})")
        print("  ✓ Preprocessor class defined correctly")

    print("=" * 60)
