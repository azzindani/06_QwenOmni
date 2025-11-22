"""
Integration tests for inference components.
Requires GPU and model to be available.
"""

import pytest
import numpy as np


@pytest.mark.integration
@pytest.mark.slow
class TestModelManager:
    """Integration tests for ModelManager."""

    def test_model_loading(self):
        """Test model loading with real model."""
        from core.inference.model_manager import ModelManager
        from config import Config

        config = Config(model_path="Qwen/Qwen2.5-Omni-3B")
        manager = ModelManager(config)

        assert not manager.is_loaded()

        manager.load_model()

        assert manager.is_loaded()
        assert manager.get_model() is not None
        assert manager.get_processor() is not None

        manager.unload_model()

        assert not manager.is_loaded()


@pytest.mark.integration
@pytest.mark.slow
class TestResponseGenerator:
    """Integration tests for ResponseGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator with loaded model."""
        from core.inference.model_manager import ModelManager
        from core.inference.generator import ResponseGenerator
        from config import Config

        config = Config(model_path="Qwen/Qwen2.5-Omni-3B")
        manager = ModelManager(config)
        manager.load_model()

        generator = ResponseGenerator(manager, config)
        yield generator

        manager.unload_model()

    def test_generate_from_audio(self, generator, sample_audio_file):
        """Test generating response from audio file."""
        result = generator.generate_from_audio(
            audio_path=sample_audio_file,
            return_audio=True
        )

        assert result is not None
        assert result.text is not None
        assert len(result.text) > 0

    def test_generate_text_only(self, generator, sample_audio_file):
        """Test generating text-only response."""
        result = generator.generate_from_audio(
            audio_path=sample_audio_file,
            return_audio=False
        )

        assert result is not None
        assert result.text is not None
        assert result.audio is None

    def test_generate_with_prompt(self, generator, sample_audio_file):
        """Test generating with text prompt."""
        result = generator.generate_from_audio(
            audio_path=sample_audio_file,
            prompt="Transcribe this audio",
            return_audio=False
        )

        assert result is not None
        assert result.text is not None


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_conversation_flow(self):
        """Test full conversation flow."""
        from core.inference.model_manager import ModelManager
        from core.inference.generator import ResponseGenerator
        from core.conversation.session import SessionManager
        from config import Config
        import tempfile
        from utils.audio_utils import save_audio

        # Setup
        config = Config(model_path="Qwen/Qwen2.5-Omni-3B")
        manager = ModelManager(config)
        manager.load_model()

        session_manager = SessionManager()
        session = session_manager.create_session()
        generator = ResponseGenerator(manager, config)

        # Create test audio
        sr = 16000
        t = np.linspace(0, 1, sr)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio(audio, f.name, sr)
            audio_path = f.name

        try:
            # Process audio
            session.add_user_audio(audio_path)
            result = generator.generate_from_audio(audio_path, return_audio=True)
            session.add_assistant_message(result.text)

            # Verify
            assert session.get_turn_count() == 1
            assert result.text is not None

        finally:
            manager.unload_model()
            import os
            if os.path.exists(audio_path):
                os.unlink(audio_path)
