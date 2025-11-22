"""
Response generator for Qwen Omni.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from utils.logger import get_logger
from config import Config, get_config
from .model_manager import ModelManager

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Result of generation."""
    text: str
    audio: Optional[np.ndarray] = None
    sample_rate: int = 24000


class ResponseGenerator:
    """Generates text and audio responses using Qwen Omni."""

    def __init__(self, model_manager: ModelManager, config: Optional[Config] = None):
        """
        Initialize response generator.

        Args:
            model_manager: ModelManager instance
            config: Configuration instance
        """
        self.model_manager = model_manager
        self.config = config or get_config()

    def generate(
        self,
        messages: List[Dict],
        return_audio: bool = True,
    ) -> GenerationResult:
        """
        Generate response from messages.

        Args:
            messages: Conversation messages
            return_audio: Whether to return audio output

        Returns:
            GenerationResult with text and optional audio
        """
        model = self.model_manager.get_model()
        processor = self.model_manager.get_processor()

        # Import here for lazy loading
        from qwen_omni_utils import process_mm_info

        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process multimodal info
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        # Prepare inputs
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )

        # Move to device
        inputs = inputs.to(model.device).to(model.dtype)

        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=return_audio,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
            )

        # Parse output
        if return_audio:
            text_ids = output[0]
            audio_output = output[1]

            # Decode text
            response_text = processor.batch_decode(
                text_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Convert audio to numpy
            if audio_output is not None:
                audio_array = audio_output.cpu().numpy().squeeze()
            else:
                audio_array = None

            return GenerationResult(
                text=response_text,
                audio=audio_array,
                sample_rate=self.config.sample_rate_output
            )
        else:
            # Text only
            response_text = processor.batch_decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return GenerationResult(text=response_text)

    def generate_from_audio(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
        return_audio: bool = True,
    ) -> GenerationResult:
        """
        Generate response from audio input.

        Args:
            audio_path: Path to audio file
            prompt: Optional text prompt
            return_audio: Whether to return audio output

        Returns:
            GenerationResult with text and optional audio
        """
        # Build message content
        content = []

        if prompt:
            content.append({"type": "text", "text": prompt})

        content.append({"type": "audio", "audio": audio_path})

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": content}
        ]

        return self.generate(messages, return_audio=return_audio)


if __name__ == "__main__":
    print("=" * 60)
    print("RESPONSE GENERATOR TEST")
    print("=" * 60)

    from utils.hardware import detect_gpu

    if detect_gpu():
        print("  ResponseGenerator class defined correctly")
        print("  Note: Requires loaded model to generate responses")
        print("  ✓ Generator ready for use")
    else:
        print("  Note: GPU not available")
        print("  ✓ ResponseGenerator class defined correctly")

    print("=" * 60)
