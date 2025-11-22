"""
Model manager for Qwen Omni.
"""

import torch
from typing import Optional, Any

from utils.logger import get_logger
from utils.hardware import detect_gpu
from config import Config, get_config

logger = get_logger(__name__)


class ModelManager:
    """Manages loading and lifecycle of Qwen Omni model."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize model manager.

        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self._model = None
        self._processor = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the Qwen Omni model and processor."""
        if self._loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model: {self.config.model_path}")

        # Import here to avoid loading at module import time
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig

        # Configure quantization
        quantization_config = None
        if self.config.quantization == "4bit" and detect_gpu():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")

        # Load model
        model_kwargs = {
            "device_map": self.config.device_map,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if self.config.use_flash_attention and detect_gpu():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")

        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.config.model_path,
            **model_kwargs
        )

        # Load processor
        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.config.model_path)

        self._loaded = True
        logger.info("Model loaded successfully")

        # Log memory usage
        if detect_gpu():
            memory_used = torch.cuda.max_memory_reserved() / 1024**3
            logger.info(f"GPU memory used: {memory_used:.2f} GB")

    def unload_model(self) -> None:
        """Unload the model and free memory."""
        if not self._loaded:
            return

        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._loaded = False

        if detect_gpu():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")

    def get_model(self) -> Any:
        """
        Get the loaded model.

        Returns:
            The Qwen Omni model

        Raises:
            RuntimeError: If model not loaded
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    def get_processor(self) -> Any:
        """
        Get the loaded processor.

        Returns:
            The Qwen Omni processor

        Raises:
            RuntimeError: If model not loaded
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._processor

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL MANAGER TEST")
    print("=" * 60)

    from utils.hardware import get_device_info

    info = get_device_info()
    print(f"  GPU Available: {info['cuda_available']}")

    if info['cuda_available']:
        manager = ModelManager()
        print(f"  Model Path: {manager.config.model_path}")
        print("  Note: Call manager.load_model() to load the model")
        print("  ✓ ModelManager initialized successfully")
    else:
        print("  Note: GPU not available, model loading requires GPU")
        print("  ✓ ModelManager class defined correctly")

    print("=" * 60)
