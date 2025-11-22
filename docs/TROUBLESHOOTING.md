# Troubleshooting Guide

## Common Issues

### GPU Not Detected

**Symptoms:**
- "CUDA not available" error
- Model loads on CPU (very slow)

**Solutions:**

1. Check NVIDIA driver:
```bash
nvidia-smi
```

2. Verify CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

3. Install correct PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

4. Docker: Ensure NVIDIA Container Toolkit:
```bash
docker run --gpus all nvidia/cuda:12.1-base nvidia-smi
```

---

### Out of Memory (OOM)

**Symptoms:**
- "CUDA out of memory" error
- Process killed

**Solutions:**

1. Use smaller model:
```bash
MODEL_PATH=Qwen/Qwen2.5-Omni-3B python main.py
```

2. Reduce batch size / max tokens:
```bash
MAX_NEW_TOKENS=128 python main.py
```

3. Clear cache between inferences:
```python
torch.cuda.empty_cache()
```

4. Use 4-bit quantization (default):
```bash
QUANTIZATION=4bit python main.py
```

---

### Model Loading Fails

**Symptoms:**
- "Error loading model" message
- Timeout during load

**Solutions:**

1. Check disk space:
```bash
df -h
```

2. Verify HuggingFace access:
```bash
huggingface-cli whoami
```

3. Use local model:
```bash
MODEL_PATH=/path/to/local/model python main.py
```

4. Check network (model download):
```bash
curl -I https://huggingface.co
```

---

### Audio Processing Errors

**Symptoms:**
- "Could not load audio" error
- Garbled output

**Solutions:**

1. Check audio format:
```bash
ffprobe input.wav
```

2. Install FFmpeg:
```bash
# Ubuntu
sudo apt install ffmpeg

# Mac
brew install ffmpeg
```

3. Verify sample rate (should be 16kHz):
```python
import librosa
audio, sr = librosa.load("input.wav", sr=16000)
```

4. Check audio duration (max 30s default):
```bash
MAX_AUDIO_DURATION=60 python main.py
```

---

### WebSocket Connection Issues

**Symptoms:**
- "Connection refused" error
- WebSocket disconnects

**Solutions:**

1. Check server is running:
```bash
curl http://localhost:7860/health
```

2. Verify port not blocked:
```bash
netstat -tlnp | grep 7860
```

3. Check firewall:
```bash
sudo ufw allow 7860
```

4. Increase timeout:
```javascript
const ws = new WebSocket(url);
ws.timeout = 30000;
```

---

### Slow Response Times

**Symptoms:**
- Response takes >5 seconds
- Gradio interface unresponsive

**Solutions:**

1. Warm up model first:
```python
from utils.performance import ModelWarmup
warmup = ModelWarmup(model_manager)
warmup.warmup()
```

2. Reduce max tokens:
```bash
MAX_NEW_TOKENS=128 python main.py
```

3. Use Flash Attention:
```bash
pip install flash-attn --no-build-isolation
USE_FLASH_ATTENTION=true python main.py
```

4. Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

---

### Import Errors

**Symptoms:**
- "ModuleNotFoundError" errors
- Missing dependencies

**Solutions:**

1. Install all requirements:
```bash
pip install -r requirements.txt
```

2. Install custom transformers:
```bash
pip install git+https://github.com/BakerBunker/transformers@21dbefaa54e5bf180464696aa70af0bfc7a61d53
```

3. Reinstall in order:
```bash
pip uninstall transformers -y
pip install -r requirements.txt
```

---

### Session Errors

**Symptoms:**
- "Session not found" error
- Lost conversation history

**Solutions:**

1. Check session exists:
```bash
curl http://localhost:7860/sessions
```

2. Create new session:
```bash
curl -X POST http://localhost:7860/session
```

3. Increase max sessions:
```bash
MAX_SESSIONS=100 python main.py
```

---

### Docker Issues

**Symptoms:**
- Container exits immediately
- "Permission denied" errors

**Solutions:**

1. Check logs:
```bash
docker logs qwen-omni
```

2. Run with GPU:
```bash
docker run --gpus all ...
```

3. Fix permissions:
```bash
docker run --user $(id -u):$(id -g) ...
```

4. Increase shared memory:
```bash
docker run --shm-size=2g ...
```

---

### Kubernetes Issues

**Symptoms:**
- Pod stuck in "Pending"
- CrashLoopBackOff

**Solutions:**

1. Check pod status:
```bash
kubectl describe pod <pod-name> -n qwen-omni
```

2. Check GPU scheduling:
```bash
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
```

3. Check resource requests:
```yaml
resources:
  requests:
    nvidia.com/gpu: 1
  limits:
    nvidia.com/gpu: 1
```

4. View logs:
```bash
kubectl logs -f <pod-name> -n qwen-omni
```

---

## Performance Tuning

### Optimize for Latency

```bash
# Use smaller model
MODEL_PATH=Qwen/Qwen2.5-Omni-3B

# Reduce tokens
MAX_NEW_TOKENS=128

# Lower temperature (less sampling)
TEMPERATURE=0.3

# Enable Flash Attention
USE_FLASH_ATTENTION=true
```

### Optimize for Throughput

```bash
# Use batch processing (if supported)
BATCH_SIZE=4

# Increase max sessions
MAX_SESSIONS=100

# Use async processing
```

---

## Getting Help

1. Check logs:
```bash
# Local
tail -f logs/app.log

# Docker
docker logs -f qwen-omni

# Kubernetes
kubectl logs -f deployment/qwen-omni -n qwen-omni
```

2. Enable debug logging:
```bash
LOG_LEVEL=DEBUG python main.py
```

3. Run diagnostics:
```bash
python -m utils.hardware
python -m config
```

4. File an issue with:
- Error message
- Steps to reproduce
- Environment info (OS, GPU, Python version)
- Logs
