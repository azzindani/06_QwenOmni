"""
Unit tests for configuration.
"""

import pytest
import os


@pytest.mark.unit
class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        from config import Config

        config = Config()

        assert config.model_path == "Qwen/Qwen2.5-Omni-3B"
        assert config.quantization == "4bit"
        assert config.sample_rate_input == 16000
        assert config.sample_rate_output == 24000
        assert config.temperature == 0.6
        assert config.max_new_tokens == 256
        assert config.port == 7860

    def test_custom_config(self):
        """Test custom configuration values."""
        from config import Config

        config = Config(
            model_path="custom/model",
            temperature=0.8,
            port=8080,
        )

        assert config.model_path == "custom/model"
        assert config.temperature == 0.8
        assert config.port == 8080

    def test_config_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        from config import Config

        monkeypatch.setenv("MODEL_PATH", "env/model")
        monkeypatch.setenv("TEMPERATURE", "0.9")
        monkeypatch.setenv("PORT", "9000")

        config = Config.from_env()

        assert config.model_path == "env/model"
        assert config.temperature == 0.9
        assert config.port == 9000

    def test_get_config_singleton(self):
        """Test global config singleton."""
        from config import get_config, set_config, Config

        # Reset global config
        set_config(Config(port=1234))

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2
        assert config1.port == 1234

    def test_set_config(self):
        """Test setting global config."""
        from config import get_config, set_config, Config

        new_config = Config(port=5555)
        set_config(new_config)

        config = get_config()
        assert config.port == 5555
