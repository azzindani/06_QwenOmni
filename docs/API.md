# API Documentation

## Overview

The Qwen Omni Voice Assistant provides both REST and WebSocket APIs for integration.

## Authentication

### API Keys

All API endpoints require authentication via API key.

```bash
# Include in header
Authorization: Bearer qo_your_api_key_here

# Or as query parameter
?api_key=qo_your_api_key_here
```

### Rate Limiting

- Default: 100 requests per minute
- Headers returned:
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Seconds until reset

---

## REST Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "active_sessions": 5,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

### Configuration

```
GET /config
```

**Response:**
```json
{
  "model_path": "Qwen/Qwen2.5-Omni-3B",
  "quantization": "4bit",
  "sample_rate_input": 16000,
  "sample_rate_output": 24000,
  "temperature": 0.6,
  "max_new_tokens": 256
}
```

---

### Sessions

#### Create Session

```
POST /session
```

**Request:**
```json
{
  "system_prompt": "You are a helpful assistant",
  "max_history_turns": 20
}
```

**Response:**
```json
{
  "session_id": "abc123",
  "created_at": "2024-01-01T00:00:00",
  "message_count": 1,
  "turn_count": 0
}
```

#### Get Session

```
GET /session/{session_id}
```

**Response:**
```json
{
  "session_id": "abc123",
  "created_at": "2024-01-01T00:00:00",
  "message_count": 5,
  "turn_count": 2
}
```

#### Delete Session

```
DELETE /session/{session_id}
```

**Response:**
```json
{
  "status": "deleted",
  "session_id": "abc123"
}
```

#### List Sessions

```
GET /sessions
```

**Response:**
```json
{
  "sessions": ["abc123", "def456"],
  "count": 2
}
```

#### Clear Session History

```
POST /session/{session_id}/clear
```

**Response:**
```json
{
  "status": "cleared",
  "session_id": "abc123"
}
```

---

### Audio Processing

#### Process Audio

```
POST /process
```

**Request:**
```
Content-Type: multipart/form-data

audio: <audio file>
session_id: abc123
return_audio: true
```

**Response:**
```json
{
  "session_id": "abc123",
  "text": "Hello! How can I help you?",
  "audio_base64": "UklGRi...",
  "sample_rate": 24000,
  "processing_time_ms": 1500
}
```

---

## WebSocket API

### Connection

```
ws://host:port/ws?session_id=abc123
```

### Messages

#### Client → Server

**Audio Chunk:**
```json
{
  "type": "audio_chunk",
  "data": "<base64_audio>",
  "sample_rate": 16000
}
```

**Control:**
```json
{
  "type": "control",
  "action": "stop" | "reset"
}
```

#### Server → Client

**Connected:**
```json
{
  "type": "connected",
  "session_id": "abc123"
}
```

**VAD Status:**
```json
{
  "type": "vad_status",
  "is_speech": true,
  "confidence": 0.85
}
```

**Processing:**
```json
{
  "type": "status",
  "status": "processing"
}
```

**Response:**
```json
{
  "type": "response",
  "text": "Hello!",
  "audio_base64": "UklGRi...",
  "sample_rate": 24000,
  "processing_time_ms": 1500
}
```

**Error:**
```json
{
  "type": "error",
  "error": "Error message"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error type",
  "detail": "Detailed message",
  "timestamp": "2024-01-01T00:00:00"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Rate Limited |
| 500 | Internal Error |

---

## SDK Examples

### Python

```python
import requests

API_KEY = "qo_your_key"
BASE_URL = "http://localhost:7860"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Create session
response = requests.post(f"{BASE_URL}/session", headers=headers)
session_id = response.json()["session_id"]

# Process audio
with open("audio.wav", "rb") as f:
    files = {"audio": f}
    data = {"session_id": session_id, "return_audio": "true"}
    response = requests.post(f"{BASE_URL}/process",
                            headers=headers, files=files, data=data)
    result = response.json()
    print(result["text"])
```

### JavaScript

```javascript
const API_KEY = 'qo_your_key';
const BASE_URL = 'http://localhost:7860';

// WebSocket connection
const ws = new WebSocket(`ws://localhost:7860/ws?api_key=${API_KEY}`);

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'response') {
        console.log('Response:', msg.text);
        // Play audio
        const audio = new Audio(`data:audio/wav;base64,${msg.audio_base64}`);
        audio.play();
    }
};

// Send audio chunk
ws.send(JSON.stringify({
    type: 'audio_chunk',
    data: audioBase64,
    sample_rate: 16000
}));
```

---

## Webhooks

Configure webhook URL to receive async notifications:

```
POST /webhooks
```

**Request:**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["session.created", "response.completed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Events

- `session.created` - New session started
- `session.ended` - Session terminated
- `response.completed` - Audio processed successfully
- `error.occurred` - Processing error

### Webhook Payload

```json
{
  "event": "response.completed",
  "timestamp": "2024-01-01T00:00:00",
  "data": {
    "session_id": "abc123",
    "text": "Response text",
    "processing_time_ms": 1500
  },
  "signature": "sha256=..."
}
```
