"""
Central configuration for Qwen Omni Voice Assistant.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Main configuration class with all settings."""

    # Model Configuration
    model_path: str = "Qwen/Qwen2.5-Omni-3B"
    quantization: str = "4bit"
    device_map: str = "auto"
    use_flash_attention: bool = True

    # Audio Configuration
    sample_rate_input: int = 16000
    sample_rate_output: int = 24000
    max_audio_duration: float = 30.0  # seconds

    # Generation Configuration
    temperature: float = 0.6
    max_new_tokens: int = 256
    top_p: float = 0.9
    do_sample: bool = True

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False

    # System Prompt
    system_prompt: str = "You are Qwen, a helpful voice assistant. Respond naturally and conversationally."

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            model_path=os.getenv("MODEL_PATH", cls.model_path),
            quantization=os.getenv("QUANTIZATION", cls.quantization),
            device_map=os.getenv("DEVICE_MAP", cls.device_map),
            use_flash_attention=os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true",
            sample_rate_input=int(os.getenv("SAMPLE_RATE_INPUT", cls.sample_rate_input)),
            sample_rate_output=int(os.getenv("SAMPLE_RATE_OUTPUT", cls.sample_rate_output)),
            max_audio_duration=float(os.getenv("MAX_AUDIO_DURATION", cls.max_audio_duration)),
            temperature=float(os.getenv("TEMPERATURE", cls.temperature)),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", cls.max_new_tokens)),
            top_p=float(os.getenv("TOP_P", cls.top_p)),
            host=os.getenv("HOST", cls.host),
            port=int(os.getenv("PORT", cls.port)),
            share=os.getenv("SHARE", "false").lower() == "true",
            system_prompt=os.getenv("SYSTEM_PROMPT", cls.system_prompt),
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG TEST")
    print("=" * 60)

    config = get_config()
    print(f"  Model: {config.model_path}")
    print(f"  Quantization: {config.quantization}")
    print(f"  Sample Rate In: {config.sample_rate_input}")
    print(f"  Sample Rate Out: {config.sample_rate_output}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Port: {config.port}")

    print("  ✓ Config loaded successfully")
    print("=" * 60)
