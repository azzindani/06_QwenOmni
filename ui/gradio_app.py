"""
Gradio web interface for Qwen Omni Voice Assistant.
"""

import gradio as gr
import numpy as np
import tempfile
import os
from typing import Tuple, List, Optional

from utils.logger import get_logger
from utils.audio_utils import save_audio
from config import Config, get_config
from core.inference.model_manager import ModelManager
from core.inference.generator import ResponseGenerator

logger = get_logger(__name__)


class GradioApp:
    """Gradio web interface for voice conversations."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Gradio app.

        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self.model_manager = ModelManager(self.config)
        self.generator: Optional[ResponseGenerator] = None
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._model_loaded:
            logger.info("Loading model...")
            self.model_manager.load_model()
            self.generator = ResponseGenerator(self.model_manager, self.config)
            self._model_loaded = True
            logger.info("Model ready")

    def process_audio(
        self,
        audio_input: Tuple[int, np.ndarray],
        chat_history: List[Tuple[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        """
        Process audio input and generate response.

        Args:
            audio_input: Tuple of (sample_rate, audio_array)
            chat_history: Current chat history
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (output_audio_path, response_text, updated_history)
        """
        if audio_input is None:
            return None, "Please provide audio input.", chat_history

        self._ensure_model_loaded()

        # Update config with UI parameters
        self.config.temperature = temperature
        self.config.max_new_tokens = max_tokens

        # Save input audio to temp file
        sample_rate, audio_array = audio_input

        # Convert to float32 and normalize
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0

        # Handle stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_input_path = f.name
            save_audio(audio_array, temp_input_path, sample_rate)

        try:
            # Generate response
            result = self.generator.generate_from_audio(
                audio_path=temp_input_path,
                return_audio=True
            )

            # Save output audio
            output_audio_path = None
            if result.audio is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    output_audio_path = f.name
                    save_audio(result.audio, output_audio_path, result.sample_rate)

            # Update chat history
            chat_history.append(("[Audio Input]", result.text))

            return output_audio_path, result.text, chat_history

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None, f"Error: {str(e)}", chat_history
        finally:
            # Cleanup temp input file
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)

    def clear_history(self) -> Tuple[List, str]:
        """Clear chat history."""
        return [], ""

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Qwen Omni Voice Assistant") as interface:
            gr.Markdown("# Qwen Omni Voice Assistant")
            gr.Markdown("Speak into your microphone and get voice responses!")

            with gr.Row():
                with gr.Column(scale=1):
                    # Input
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="numpy",
                        label="Audio Input"
                    )

                    # Parameters
                    with gr.Accordion("Generation Parameters", open=False):
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=self.config.temperature,
                            step=0.1,
                            label="Temperature"
                        )
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=1024,
                            value=self.config.max_new_tokens,
                            step=64,
                            label="Max Tokens"
                        )

                    # Buttons
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=1):
                    # Output
                    audio_output = gr.Audio(
                        label="Audio Response",
                        autoplay=True
                    )
                    text_output = gr.Textbox(
                        label="Response Text",
                        lines=3
                    )

            # Chat history
            chatbot = gr.Chatbot(label="Conversation History")

            # Event handlers
            submit_btn.click(
                fn=self.process_audio,
                inputs=[audio_input, chatbot, temperature, max_tokens],
                outputs=[audio_output, text_output, chatbot]
            )

            clear_btn.click(
                fn=self.clear_history,
                outputs=[chatbot, text_output]
            )

            # Info
            gr.Markdown(f"""
            ### Model Info
            - **Model**: {self.config.model_path}
            - **Quantization**: {self.config.quantization}

            ### Usage
            1. Click the microphone button or upload an audio file
            2. Adjust parameters if needed
            3. Click Submit to get a response
            """)

        return interface

    def launch(self, **kwargs) -> None:
        """
        Launch the Gradio interface.

        Args:
            **kwargs: Additional arguments for gr.launch()
        """
        interface = self.create_interface()

        # Default launch parameters
        launch_kwargs = {
            "server_name": self.config.host,
            "server_port": self.config.port,
            "share": self.config.share,
        }
        launch_kwargs.update(kwargs)

        logger.info(f"Launching Gradio on {self.config.host}:{self.config.port}")
        interface.launch(**launch_kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("GRADIO APP TEST")
    print("=" * 60)

    app = GradioApp()
    print(f"  Model: {app.config.model_path}")
    print(f"  Host: {app.config.host}")
    print(f"  Port: {app.config.port}")
    print("  ✓ GradioApp initialized successfully")
    print("  Run app.launch() to start the server")
    print("=" * 60)
