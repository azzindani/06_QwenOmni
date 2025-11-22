# Qwen Omni Voice Assistant

A production-ready voice conversation system powered by Qwen2.5-Omni model with real-time audio streaming capabilities.

---

## Project Overview

### Vision
Build a voice assistant that enables natural, turn-based conversations with audio input and output, similar to a phone call experience.

### Expectations & Limitations

**What This System Will Do:**
- Accept live audio streaming from users via WebSocket
- Detect end of speech using Voice Activity Detection (VAD)
- Process complete utterances through Qwen Omni model
- Return both text and synthesized audio responses
- Maintain multi-round conversation context
- Provide a Gradio-based web interface

**What This System Will NOT Do:**
- Real-time word-by-word streaming (model limitation)
- Interrupt or process partial speech
- Sub-second response times (expect 1-3 seconds latency)

**Expected Latency:**
- Speech detection: ~100-300ms after user stops speaking
- Model inference: ~1-2 seconds (3B model with 4-bit quantization)
- Audio synthesis: ~200-500ms
- **Total round-trip: 1.5-3 seconds**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GRADIO UI LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Audio Input │  │ Audio Output│  │ Conversation Display    │  │
│  │ (Streaming) │  │ (Playback)  │  │ (Text + Controls)       │  │
│  └──────┬──────┘  └──────▲──────┘  └─────────────────────────┘  │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
          ▼                │
┌─────────────────────────────────────────────────────────────────┐
│                      API / SERVICE LAYER                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ WebSocket       │  │ Session         │  │ Health Check    │  │
│  │ Handler         │  │ Manager         │  │ Endpoint        │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────────┘  │
└───────────┼────────────────────┼────────────────────────────────┘
            │                    │
            ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ VAD         │  │ Audio       │  │ Conversation│              │
│  │ Detector    │  │ Processor   │  │ Manager     │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              INFERENCE ENGINE                               ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  ││
│  │  │ Model       │  │ Processor   │  │ Response Generator  │  ││
│  │  │ Manager     │  │ (Tokenizer) │  │ (Text + Audio)      │  ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Config      │  │ Logger      │  │ Hardware    │              │
│  │ Manager     │  │ Utils       │  │ Detection   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Speech → Microphone → WebSocket → Audio Buffer → VAD Detection
    ↓
Speech End Detected → Complete Audio Chunk
    ↓
Audio Preprocessing (16kHz, normalization)
    ↓
Qwen Omni Inference (with conversation history)
    ↓
Text Response + Audio Response (24kHz)
    ↓
WebSocket → Gradio UI → Speaker Playback
```

---

## Directory Structure

```
qwen_omni/
├── README.md                    # This file
├── WORKFLOW.md                  # Development workflow guide
├── requirements.txt             # Python dependencies
├── config.py                    # Central configuration
├── main.py                      # Application entry point
├── Dockerfile                   # Container deployment
├── docker-compose.yml           # Multi-service orchestration
├── .env.example                 # Environment variables template
│
├── core/                        # Core business logic
│   ├── __init__.py
│   ├── inference/               # Model inference
│   │   ├── __init__.py
│   │   ├── model_manager.py     # Model loading & management
│   │   ├── processor.py         # Input/output processing
│   │   └── generator.py         # Response generation
│   │
│   ├── audio/                   # Audio processing
│   │   ├── __init__.py
│   │   ├── vad.py               # Voice Activity Detection
│   │   ├── preprocessor.py      # Audio normalization
│   │   └── stream_handler.py    # Streaming buffer management
│   │
│   └── conversation/            # Conversation management
│       ├── __init__.py
│       ├── session.py           # Session state management
│       ├── history.py           # Conversation history
│       └── context.py           # Context window management
│
├── api/                         # API layer
│   ├── __init__.py
│   ├── websocket.py             # WebSocket handlers
│   ├── routes.py                # REST endpoints
│   └── schemas.py               # Request/response models
│
├── ui/                          # User interface
│   ├── __init__.py
│   ├── gradio_app.py            # Main Gradio application
│   ├── components/              # UI components
│   │   ├── __init__.py
│   │   ├── audio_input.py       # Microphone input component
│   │   ├── audio_output.py      # Audio playback component
│   │   └── chat_display.py      # Conversation display
│   └── static/                  # Static assets
│       └── styles.css
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── logger.py                # Logging configuration
│   ├── hardware.py              # Hardware detection
│   └── audio_utils.py           # Audio helper functions
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── unit/                    # Unit tests (no GPU)
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_vad.py
│   │   ├── test_audio_processor.py
│   │   ├── test_session.py
│   │   ├── test_history.py
│   │   └── test_schemas.py
│   │
│   └── integration/             # Integration tests (requires GPU)
│       ├── __init__.py
│       ├── test_model_manager.py
│       ├── test_inference.py
│       ├── test_websocket.py
│       └── test_end_to_end.py
│
└── 00_Dataset/                  # Test audio files (existing)
    ├── *.wav
    └── *.flac
```

---

## Features Roadmap

### Phase 0: MVP (Minimum Viable Product) ✅
**Goal: Basic voice conversation working end-to-end**

- [x] **Configuration Management**
  - Central config.py with all settings
  - Environment variable support
  - Model selection (3B/7B)

- [x] **Model Infrastructure**
  - Model loading with 4-bit quantization
  - Processor initialization
  - Basic inference function

- [x] **Audio Processing**
  - Audio file loading (16kHz)
  - Basic preprocessing
  - Audio output saving (24kHz)

- [x] **Simple Gradio UI**
  - Audio file upload
  - Text display
  - Audio playback

- [x] **Basic Tests**
  - Config loading tests
  - Audio preprocessing tests

---

### Phase 1: Production Ready ✅
**Goal: Stable, testable, deployable system**

- [x] **Voice Activity Detection**
  - Silero VAD integration
  - Speech boundary detection
  - Configurable thresholds

- [x] **Streaming Support**
  - Audio buffer management
  - Chunk processing
  - WebSocket integration

- [x] **Conversation Management**
  - Session state tracking
  - Multi-round history
  - Context window management

- [x] **Enhanced Gradio UI**
  - Real-time microphone input
  - Streaming audio output
  - Conversation history display
  - Parameter controls (temperature, max_tokens)

- [x] **Error Handling**
  - Graceful degradation
  - User-friendly error messages
  - Automatic recovery

- [x] **Logging & Monitoring**
  - Structured logging
  - Performance metrics
  - Request tracking

- [x] **Complete Test Suite**
  - All unit tests
  - Integration tests
  - End-to-end tests

- [x] **Docker Deployment**
  - Dockerfile
  - docker-compose.yml
  - Health check endpoint

---

### Phase 2: Enhanced Features ✅
**Goal: Better user experience and performance**

- [x] **Advanced VAD**
  - Adaptive thresholds
  - Noise reduction
  - Echo cancellation

- [x] **Performance Optimization**
  - Model warm-up
  - Response caching
  - Batch processing

- [x] **Multi-Language Support**
  - Language detection
  - Configurable system prompts
  - Translation mode

- [ ] **UI Enhancements** (Partial)
  - Visual waveform display
  - Speaking indicator
  - Interrupt button
  - Dark/light theme

- [x] **Session Persistence**
  - Save/load conversations
  - Export transcripts
  - Session replay

---

### Phase 3: Advanced Features ✅
**Goal: Production-grade enterprise features**

- [x] **Multi-User Support**
  - Concurrent sessions
  - User authentication
  - Rate limiting

- [x] **Analytics Dashboard**
  - Usage statistics
  - Response latency tracking
  - Error rate monitoring

- [x] **API Gateway**
  - REST API for integration
  - API key management
  - Webhook support

- [x] **Kubernetes Deployment**
  - Helm charts
  - Auto-scaling
  - Load balancing

---

### Outstanding Items

The following items need additional work:

1. **UI Enhancements** (Phase 2)
   - [ ] Visual waveform display component
   - [ ] Real-time speaking indicator animation
   - [ ] Interrupt/stop button during generation
   - [ ] Theme toggle (dark/light mode)

2. **Authentication** (Phase 3)
   - [ ] JWT token authentication
   - [ ] API key management UI
   - [ ] Rate limiting middleware

3. **Additional Testing**
   - [ ] Load testing scripts
   - [ ] Performance benchmarks
   - [ ] Security audit

4. **Documentation**
   - [ ] API documentation (OpenAPI/Swagger)
   - [ ] Deployment guide
   - [ ] Troubleshooting guide

---

## Component Specifications

### Core Components

#### 1. ModelManager (`core/inference/model_manager.py`)
**Purpose:** Load and manage the Qwen Omni model

```python
class ModelManager:
    def __init__(self, config: Config)
    def load_model(self) -> None
    def unload_model(self) -> None
    def get_model(self) -> Qwen2_5OmniForConditionalGeneration
    def get_processor(self) -> Qwen2_5OmniProcessor
    def is_loaded(self) -> bool
```

**Tests:**
- `test_model_load_3b` - Load 3B model successfully
- `test_model_load_7b` - Load 7B model successfully
- `test_model_quantization` - Verify 4-bit quantization
- `test_model_device_placement` - Check GPU allocation
- `test_model_reload` - Unload and reload model

---

#### 2. VADDetector (`core/audio/vad.py`)
**Purpose:** Detect speech boundaries in audio stream

```python
class VADDetector:
    def __init__(self, config: VADConfig)
    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult
    def is_speech_ended(self) -> bool
    def reset(self) -> None
    def get_speech_audio(self) -> np.ndarray
```

**Tests:**
- `test_vad_detect_speech` - Detect speech in audio
- `test_vad_detect_silence` - Detect silence periods
- `test_vad_speech_boundary` - Find speech start/end
- `test_vad_noise_robustness` - Handle background noise
- `test_vad_reset` - Reset state between utterances

---

#### 3. AudioPreprocessor (`core/audio/preprocessor.py`)
**Purpose:** Normalize and prepare audio for model

```python
class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000)
    def process(self, audio: np.ndarray, sr: int) -> np.ndarray
    def normalize(self, audio: np.ndarray) -> np.ndarray
    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray
```

**Tests:**
- `test_resample_to_16khz` - Resample from various rates
- `test_normalize_audio` - Normalize amplitude
- `test_handle_stereo` - Convert stereo to mono
- `test_handle_silence` - Process silent audio
- `test_handle_clipping` - Handle clipped audio

---

#### 4. StreamHandler (`core/audio/stream_handler.py`)
**Purpose:** Manage streaming audio buffer

```python
class StreamHandler:
    def __init__(self, config: StreamConfig)
    def add_chunk(self, chunk: bytes) -> None
    def get_complete_audio(self) -> Optional[np.ndarray]
    def clear(self) -> None
    def get_buffer_duration(self) -> float
```

**Tests:**
- `test_buffer_accumulation` - Accumulate audio chunks
- `test_buffer_overflow` - Handle max buffer size
- `test_chunk_conversion` - Convert bytes to numpy
- `test_buffer_clear` - Clear buffer state

---

#### 5. ConversationSession (`core/conversation/session.py`)
**Purpose:** Manage individual conversation session

```python
class ConversationSession:
    def __init__(self, session_id: str, config: SessionConfig)
    def add_user_message(self, content: List[Dict]) -> None
    def add_assistant_message(self, text: str, audio: np.ndarray) -> None
    def get_messages(self) -> List[Dict]
    def clear(self) -> None
    def get_context_length(self) -> int
```

**Tests:**
- `test_session_create` - Create new session
- `test_add_messages` - Add user/assistant messages
- `test_message_format` - Verify message structure
- `test_context_tracking` - Track context length
- `test_session_clear` - Clear session history

---

#### 6. ResponseGenerator (`core/inference/generator.py`)
**Purpose:** Generate text and audio responses

```python
class ResponseGenerator:
    def __init__(self, model_manager: ModelManager, config: GenerationConfig)
    def generate(self, messages: List[Dict], audio: np.ndarray) -> GenerationResult
    def generate_text_only(self, messages: List[Dict]) -> str
    def generate_with_audio(self, messages: List[Dict]) -> Tuple[str, np.ndarray]
```

**Tests:**
- `test_generate_text_response` - Generate text only
- `test_generate_audio_response` - Generate with audio
- `test_generation_params` - Test temperature, max_tokens
- `test_multimodal_input` - Handle audio + text input
- `test_empty_input` - Handle edge cases

---

### UI Components

#### 7. GradioApp (`ui/gradio_app.py`)
**Purpose:** Main Gradio web interface

```python
class GradioApp:
    def __init__(self, config: Config)
    def create_interface(self) -> gr.Blocks
    def launch(self, **kwargs) -> None
```

**Features:**
- Audio input (microphone streaming)
- Audio output (auto-playback)
- Conversation history display
- Parameter controls
- Clear/reset buttons
- Status indicators

**Tests:**
- `test_gradio_interface_creation` - Interface builds without error
- `test_audio_input_handler` - Process audio input
- `test_conversation_display` - Display updates correctly
- `test_parameter_controls` - Controls affect generation

---

### API Components

#### 8. WebSocketHandler (`api/websocket.py`)
**Purpose:** Handle WebSocket connections for streaming

```python
class WebSocketHandler:
    def __init__(self, session_manager: SessionManager)
    async def connect(self, websocket: WebSocket) -> str
    async def receive_audio(self, session_id: str, data: bytes) -> None
    async def send_response(self, session_id: str, response: dict) -> None
    async def disconnect(self, session_id: str) -> None
```

**Tests:**
- `test_websocket_connect` - Establish connection
- `test_websocket_receive` - Receive audio data
- `test_websocket_send` - Send response
- `test_websocket_disconnect` - Clean disconnect
- `test_concurrent_connections` - Handle multiple clients

---

## Test Strategy

### Test Categories

```python
# Unit tests - no GPU required
@pytest.mark.unit
def test_config_loading():
    pass

# Integration tests - requires GPU and model
@pytest.mark.integration
def test_full_inference():
    pass

# Slow tests - take more than 10 seconds
@pytest.mark.slow
def test_model_loading():
    pass
```

### Running Tests

```bash
# Run all unit tests (CI/CD)
pytest -m unit -v

# Run integration tests (local with GPU)
pytest -m integration -v

# Run specific component tests
pytest tests/unit/test_vad.py -v

# Run with coverage
pytest -m unit --cov=qwen_omni --cov-report=html

# Run single module directly
python -m qwen_omni.core.audio.vad
```

### Test Coverage Requirements

| Component | Minimum Coverage |
|-----------|-----------------|
| config.py | 90% |
| core/audio/* | 85% |
| core/conversation/* | 85% |
| core/inference/* | 75% |
| api/* | 80% |
| ui/* | 70% |

---

## Configuration

### Environment Variables

```bash
# .env.example

# Model Configuration
MODEL_PATH=Qwen/Qwen2.5-Omni-3B
QUANTIZATION=4bit
DEVICE_MAP=auto

# Audio Configuration
SAMPLE_RATE_INPUT=16000
SAMPLE_RATE_OUTPUT=24000
VAD_THRESHOLD=0.5
SILENCE_DURATION_MS=800

# Generation Configuration
TEMPERATURE=0.6
MAX_NEW_TOKENS=256
TOP_P=0.9

# Server Configuration
HOST=0.0.0.0
PORT=7860
MAX_SESSIONS=10

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Config Dataclass

```python
@dataclass
class Config:
    # Model
    model_path: str = "Qwen/Qwen2.5-Omni-3B"
    quantization: str = "4bit"
    device_map: str = "auto"

    # Audio
    sample_rate_input: int = 16000
    sample_rate_output: int = 24000
    vad_threshold: float = 0.5
    silence_duration_ms: int = 800

    # Generation
    temperature: float = 0.6
    max_new_tokens: int = 256

    # Server
    host: str = "0.0.0.0"
    port: int = 7860
```

---

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "main.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  qwen-omni:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/app/models/Qwen2.5-Omni-3B
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### System Requirements

**Minimum (3B Model):**
- GPU: 6GB VRAM (4-bit quantization)
- RAM: 16GB
- Storage: 10GB

**Recommended (7B Model):**
- GPU: 12GB VRAM (4-bit quantization)
- RAM: 32GB
- Storage: 20GB

---

## Development Setup

### Installation

```bash
# Clone repository
git clone <repository-url>
cd qwen_omni

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install custom transformers (required)
pip install git+https://github.com/BakerBunker/transformers@21dbefaa54e5bf180464696aa70af0bfc7a61d53

# Copy environment template
cp .env.example .env
# Edit .env with your settings
```

### Running Locally

```bash
# Run Gradio UI
python main.py

# Run with custom port
python main.py --port 8080

# Run tests
pytest -m unit -v
```

---

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/config` | Get current config |
| POST | `/session` | Create new session |
| DELETE | `/session/{id}` | End session |

### WebSocket

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client→Server | Establish connection |
| `audio_chunk` | Client→Server | Send audio data |
| `response` | Server→Client | Text + audio response |
| `error` | Server→Client | Error message |
| `disconnect` | Both | End connection |

---

## Contributing

1. Create feature branch: `git checkout -b claude/feature-name-session-id`
2. Follow development workflow in WORKFLOW.md
3. Ensure all unit tests pass: `pytest -m unit -v`
4. Commit with clear messages
5. Push and create PR

---

## License

[Add license information]

---

## Acknowledgments

- Qwen team for the Qwen2.5-Omni model
- Hugging Face for transformers library
- Gradio team for the UI framework
