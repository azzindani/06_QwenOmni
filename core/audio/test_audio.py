"""
Test audio processing components.

Run with: python -m core.audio.test_audio
"""

import sys


def test_preprocessor():
    """Test audio preprocessor."""
    print("\n[1/3] Testing AudioPreprocessor...")

    try:
        import numpy as np
        from core.audio.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor()

        # Test with synthetic audio
        sr = 16000
        audio = np.random.randn(sr * 2).astype(np.float32)  # 2 seconds

        processed = preprocessor.process(audio, sr)

        assert processed is not None
        assert len(processed.shape) == 1
        assert processed.max() <= 1.0
        assert processed.min() >= -1.0

        duration = len(processed) / sr
        print(f"  ✓ Processed {len(audio)} samples -> {len(processed)} samples")
        print(f"  ✓ Duration: {duration:.2f}s")
        print(f"  ✓ Range: [{processed.min():.3f}, {processed.max():.3f}]")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_stream_handler():
    """Test stream handler."""
    print("\n[2/3] Testing StreamHandler...")

    try:
        import numpy as np
        from core.audio.stream_handler import StreamHandler

        handler = StreamHandler()

        # Add chunks
        for i in range(10):
            chunk = np.random.randn(1600).astype(np.float32)  # 100ms
            handler.add_array(chunk)

        duration = handler.get_buffer_duration()
        samples = handler.get_buffer_samples()

        assert not handler.is_empty()
        assert abs(duration - 1.0) < 0.01

        # Get complete audio
        audio = handler.get_complete_audio()
        assert audio is not None
        assert len(audio) == 16000

        # Clear
        handler.clear()
        assert handler.is_empty()

        print(f"  ✓ Buffered {samples} samples ({duration:.2f}s)")
        print(f"  ✓ Retrieved complete audio")
        print(f"  ✓ Clear working")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_vad():
    """Test VAD detector."""
    print("\n[3/3] Testing VADDetector...")

    try:
        from core.audio.vad import VADDetector, VADConfig

        config = VADConfig(threshold=0.5)
        detector = VADDetector(config)

        # Test initialization
        assert not detector._loaded
        assert detector.config.threshold == 0.5

        # Test reset
        detector._speech_started = True
        detector.reset()
        assert not detector._speech_started

        print(f"  ✓ VADConfig: threshold={config.threshold}")
        print(f"  ✓ VADDetector initialized")
        print(f"  ✓ Reset working")
        print(f"  Note: Full VAD test requires torch/silero model")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all audio tests."""
    print("=" * 60)
    print("AUDIO PROCESSING TEST")
    print("=" * 60)

    results = []

    results.append(("Preprocessor", test_preprocessor()))
    results.append(("StreamHandler", test_stream_handler()))
    results.append(("VADDetector", test_vad()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
