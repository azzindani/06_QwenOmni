"""
WebSocket handler for streaming audio.
"""

import asyncio
import json
import base64
import time
import tempfile
import os
from typing import Dict, Optional, Any

from utils.logger import get_logger
from utils.audio_utils import save_audio
from config import get_config
from core.conversation.session import SessionManager
from core.audio.stream_handler import StreamHandler
from core.audio.vad import VADDetector, VADConfig

logger = get_logger(__name__)


class WebSocketHandler:
    """Handles WebSocket connections for audio streaming."""

    def __init__(
        self,
        session_manager: SessionManager,
        generator=None,
    ):
        """
        Initialize WebSocket handler.

        Args:
            session_manager: Session manager instance
            generator: Response generator instance
        """
        self.session_manager = session_manager
        self.generator = generator
        self._connections: Dict[str, Any] = {}
        self._stream_handlers: Dict[str, StreamHandler] = {}
        self._vad_detectors: Dict[str, VADDetector] = {}

    async def connect(self, websocket, session_id: Optional[str] = None) -> str:
        """
        Handle new WebSocket connection.

        Args:
            websocket: WebSocket connection
            session_id: Optional existing session ID

        Returns:
            Session ID
        """
        # Get or create session
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session:
                session = self.session_manager.create_session()
                session_id = session.session_id
        else:
            session = self.session_manager.create_session()
            session_id = session.session_id

        # Store connection
        self._connections[session_id] = websocket
        self._stream_handlers[session_id] = StreamHandler()
        self._vad_detectors[session_id] = VADDetector(VADConfig())

        logger.info(f"WebSocket connected: {session_id}")

        # Send connection confirmation
        await self._send_message(websocket, {
            "type": "connected",
            "session_id": session_id
        })

        return session_id

    async def disconnect(self, session_id: str) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            session_id: Session identifier
        """
        if session_id in self._connections:
            del self._connections[session_id]

        if session_id in self._stream_handlers:
            del self._stream_handlers[session_id]

        if session_id in self._vad_detectors:
            del self._vad_detectors[session_id]

        logger.info(f"WebSocket disconnected: {session_id}")

    async def receive_audio(self, session_id: str, audio_data: bytes) -> None:
        """
        Process received audio chunk.

        Args:
            session_id: Session identifier
            audio_data: Raw audio bytes
        """
        if session_id not in self._stream_handlers:
            logger.error(f"No stream handler for session: {session_id}")
            return

        stream_handler = self._stream_handlers[session_id]
        vad_detector = self._vad_detectors[session_id]

        # Add to buffer
        stream_handler.add_chunk(audio_data)

        # Get latest chunk for VAD
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        # Process through VAD
        vad_result = vad_detector.process_chunk(audio_array)

        # Send VAD status
        websocket = self._connections.get(session_id)
        if websocket:
            await self._send_message(websocket, {
                "type": "vad_status",
                "is_speech": vad_result.is_speech,
                "confidence": vad_result.confidence
            })

        # Check if speech ended
        if vad_detector.is_speech_ended():
            await self._process_complete_utterance(session_id)

    async def _process_complete_utterance(self, session_id: str) -> None:
        """
        Process a complete speech utterance.

        Args:
            session_id: Session identifier
        """
        vad_detector = self._vad_detectors[session_id]
        websocket = self._connections.get(session_id)

        if not websocket or not self.generator:
            return

        # Get speech audio
        audio = vad_detector.get_speech_audio()
        if audio is None or len(audio) < 1600:  # Less than 100ms
            vad_detector.reset()
            return

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            save_audio(audio, temp_path, 16000)

        try:
            # Send processing status
            await self._send_message(websocket, {
                "type": "status",
                "status": "processing"
            })

            start_time = time.time()

            # Get session and add audio message
            session = self.session_manager.get_session(session_id)
            if session:
                session.add_user_audio(temp_path)

                # Generate response
                result = self.generator.generate_from_audio(
                    audio_path=temp_path,
                    return_audio=True
                )

                # Add assistant response to session
                session.add_assistant_message(result.text)

                processing_time = (time.time() - start_time) * 1000

                # Prepare response
                response_data = {
                    "type": "response",
                    "text": result.text,
                    "processing_time_ms": processing_time
                }

                # Include audio if available
                if result.audio is not None:
                    # Save and encode audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        audio_path = f.name
                        save_audio(result.audio, audio_path, result.sample_rate)

                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                        response_data["audio_base64"] = base64.b64encode(audio_bytes).decode()
                        response_data["sample_rate"] = result.sample_rate

                    os.unlink(audio_path)

                await self._send_message(websocket, response_data)

        except Exception as e:
            logger.error(f"Error processing utterance: {e}")
            await self._send_message(websocket, {
                "type": "error",
                "error": str(e)
            })
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            vad_detector.reset()
            self._stream_handlers[session_id].clear()

    async def send_response(self, session_id: str, response: dict) -> None:
        """
        Send response to client.

        Args:
            session_id: Session identifier
            response: Response data
        """
        websocket = self._connections.get(session_id)
        if websocket:
            await self._send_message(websocket, response)

    async def _send_message(self, websocket, data: dict) -> None:
        """
        Send JSON message through WebSocket.

        Args:
            websocket: WebSocket connection
            data: Data to send
        """
        try:
            message = json.dumps(data)
            await websocket.send(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def get_active_connections(self) -> int:
        """Get number of active connections."""
        return len(self._connections)


if __name__ == "__main__":
    print("=" * 60)
    print("WEBSOCKET HANDLER TEST")
    print("=" * 60)

    manager = SessionManager()
    handler = WebSocketHandler(manager)

    print(f"  Active Connections: {handler.get_active_connections()}")
    print("  ✓ WebSocketHandler class defined correctly")
    print("  Note: Requires async context to test connections")
    print("=" * 60)
