"""
Gradio web interface for Qwen Omni Voice Assistant.
"""

import gradio as gr
import numpy as np
import tempfile
import os
import time
from typing import Tuple, List, Optional

from utils.logger import get_logger
from utils.audio_utils import save_audio
from utils.hardware import detect_gpu, get_device_info
from config import Config, get_config
from core.inference.model_manager import ModelManager
from core.inference.generator import ResponseGenerator
from core.conversation.session import ConversationSession, SessionManager
from api.routes import APIRoutes

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

        # Session management
        self.session_manager = SessionManager()
        self.current_session: Optional[ConversationSession] = None
        self.api_routes = APIRoutes(
            self.session_manager,
            model_loaded_fn=lambda: self._model_loaded
        )

    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._model_loaded:
            logger.info("Loading model...")
            self.model_manager.load_model()
            self.generator = ResponseGenerator(self.model_manager, self.config)
            self._model_loaded = True
            logger.info("Model ready")

    def _ensure_session(self) -> ConversationSession:
        """Ensure a session exists."""
        if self.current_session is None:
            self.current_session = self.session_manager.create_session()
        return self.current_session

    def new_session(self) -> str:
        """Create a new conversation session."""
        self.current_session = self.session_manager.create_session()
        return f"New session created: {self.current_session.session_id[:8]}..."

    def get_status(self) -> str:
        """Get current system status."""
        health = self.api_routes.health()
        device_info = get_device_info()

        status_lines = [
            f"**Model Loaded:** {'Yes' if health['model_loaded'] else 'No'}",
            f"**GPU Available:** {'Yes' if health['gpu_available'] else 'No'}",
            f"**Active Sessions:** {health['active_sessions']}",
        ]

        if device_info['devices']:
            for dev in device_info['devices']:
                status_lines.append(f"**GPU {dev['index']}:** {dev['name']} ({dev['total_memory_gb']} GB)")

        if self.current_session:
            status_lines.append(f"**Current Session:** {self.current_session.session_id[:8]}...")
            status_lines.append(f"**Conversation Turns:** {self.current_session.get_turn_count()}")

        return "\n".join(status_lines)

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
        session = self._ensure_session()

        # Update config with UI parameters
        self.config.temperature = temperature
        self.config.max_new_tokens = max_tokens

        start_time = time.time()

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
            # Add user message to session
            session.add_user_audio(temp_input_path)

            # Generate response
            result = self.generator.generate_from_audio(
                audio_path=temp_input_path,
                return_audio=True
            )

            # Add assistant response to session
            session.add_assistant_message(result.text)

            processing_time = (time.time() - start_time) * 1000

            # Save output audio
            output_audio_path = None
            if result.audio is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    output_audio_path = f.name
                    save_audio(result.audio, output_audio_path, result.sample_rate)

            # Update chat history with timing info
            response_text = f"{result.text}\n\n*({processing_time:.0f}ms)*"
            chat_history.append(("[Audio Input]", result.text))

            return output_audio_path, response_text, chat_history

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None, f"Error: {str(e)}", chat_history
        finally:
            # Cleanup temp input file
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)

    def clear_history(self) -> Tuple[List, str, str]:
        """Clear chat history and create new session."""
        if self.current_session:
            self.current_session.clear()
        status = self.new_session()
        return [], "", status

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Qwen Omni Voice Assistant") as interface:
            gr.Markdown("## 🎙️ Qwen2.5 Omni Voice Chat (Audio In/Out Only)")

            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="🎤 Speak"
                )
                audio_output = gr.Audio(
                    label="🔊 Response",
                    type="filepath",
                    interactive=True,
                    autoplay=True
                )

            with gr.Accordion("⚙️ Parameters", open=False):
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=self.config.temperature,
                    step=0.1,
                    label="Temperature"
                )
                max_tokens = gr.Slider(
                    minimum=128,
                    maximum=4096,
                    value=self.config.max_new_tokens,
                    step=1,
                    label="Max new tokens"
                )

            submit_btn = gr.Button("Send")

            # Hidden components for session management
            chatbot = gr.Chatbot(visible=False)
            text_output = gr.Textbox(visible=False)
            status_display = gr.Markdown(visible=False)

            # Event handlers
            submit_btn.click(
                fn=self.process_audio,
                inputs=[audio_input, chatbot, temperature, max_tokens],
                outputs=[audio_output, text_output, chatbot]
            )

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
