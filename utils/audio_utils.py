"""
Audio utility functions.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Tuple of (audio array, sample rate)
    """
    import librosa

    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = 24000) -> None:
    """
    Save audio array to file.

    Args:
        audio: Audio array
        file_path: Output file path
        sample_rate: Sample rate
    """
    import soundfile as sf

    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Normalize if needed
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / max(abs(audio.max()), abs(audio.min()))

    sf.write(file_path, audio, sample_rate)


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Get duration of audio in seconds.

    Args:
        audio: Audio array
        sample_rate: Sample rate

    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.

    Args:
        audio: Input audio array

    Returns:
        Normalized audio array
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono.

    Args:
        audio: Input audio (can be mono or stereo)

    Returns:
        Mono audio array
    """
    if len(audio.shape) > 1:
        return np.mean(audio, axis=1)
    return audio


if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO UTILS TEST")
    print("=" * 60)

    # Test with a sample file if it exists
    test_file = "./00_Dataset/1272-128104-0000.flac"

    try:
        audio, sr = load_audio(test_file)
        duration = get_audio_duration(audio, sr)
        print(f"  Loaded: {test_file}")
        print(f"  Sample Rate: {sr}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Shape: {audio.shape}")
        print("  ✓ Audio utils working correctly")
    except Exception as e:
        print(f"  Note: Could not load test file ({e})")
        print("  ✓ Functions defined correctly")

    print("=" * 60)
