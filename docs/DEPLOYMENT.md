# Deployment Guide

## Prerequisites

- NVIDIA GPU with 6GB+ VRAM (8GB+ recommended)
- NVIDIA Driver 525+
- Docker with NVIDIA Container Toolkit
- OR Python 3.9+ with CUDA support

---

## Local Development

### 1. Clone and Setup

```bash
git clone <repository-url>
cd qwen_omni

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install custom transformers
pip install git+https://github.com/BakerBunker/transformers@21dbefaa54e5bf180464696aa70af0bfc7a61d53

# Optional: Flash Attention
pip install flash-attn --no-build-isolation
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Run

```bash
python main.py

# With options
python main.py --port 8080 --model Qwen/Qwen2.5-Omni-7B
```

---

## Docker Deployment

### 1. Build Image

```bash
docker build -t qwen-omni:latest .
```

### 2. Run Container

```bash
docker run -d \
  --name qwen-omni \
  --gpus all \
  -p 7860:7860 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e MODEL_PATH=Qwen/Qwen2.5-Omni-3B \
  qwen-omni:latest
```

### 3. Docker Compose

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Kubernetes Deployment

### 1. Prerequisites

- Kubernetes cluster with GPU nodes
- NVIDIA device plugin installed
- `kubectl` configured

### 2. Deploy

```bash
# Create namespace
kubectl create namespace qwen-omni

# Apply configurations
kubectl apply -k k8s/

# Or apply individually
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 3. Verify

```bash
# Check pods
kubectl get pods -n qwen-omni

# Check service
kubectl get svc -n qwen-omni

# View logs
kubectl logs -f deployment/qwen-omni -n qwen-omni
```

### 4. Access

```bash
# Get external IP
kubectl get svc qwen-omni -n qwen-omni

# Port forward for testing
kubectl port-forward svc/qwen-omni 7860:80 -n qwen-omni
```

---

## Production Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `Qwen/Qwen2.5-Omni-3B` | Model to load |
| `QUANTIZATION` | `4bit` | Quantization mode |
| `TEMPERATURE` | `0.6` | Generation temperature |
| `MAX_NEW_TOKENS` | `256` | Max response tokens |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `7860` | Listen port |
| `LOG_LEVEL` | `INFO` | Logging level |

### Resource Requirements

#### 3B Model (4-bit)
- GPU: 6GB VRAM
- RAM: 16GB
- Storage: 10GB

#### 7B Model (4-bit)
- GPU: 12GB VRAM
- RAM: 32GB
- Storage: 20GB

---

## Scaling

### Horizontal Scaling

The HPA automatically scales based on CPU/memory:

```yaml
# k8s/hpa.yaml
minReplicas: 1
maxReplicas: 5
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

### GPU Considerations

- Each replica requires a GPU
- Ensure sufficient GPU nodes
- Use node affinity for GPU placement

---

## Monitoring

### Health Checks

```bash
# Liveness probe
curl http://localhost:7860/health

# Response
{"status": "healthy", "model_loaded": true, ...}
```

### Metrics

Export Prometheus metrics:

```bash
# Add to deployment
- name: ENABLE_METRICS
  value: "true"
```

Metrics available at `/metrics`:
- `qwen_requests_total`
- `qwen_request_duration_seconds`
- `qwen_active_sessions`

### Logging

Structured JSON logging:

```json
{
  "timestamp": "2024-01-01T00:00:00",
  "level": "INFO",
  "message": "Request processed",
  "session_id": "abc123",
  "duration_ms": 1500
}
```

---

## Security

### TLS/HTTPS

Use an ingress controller or reverse proxy:

```yaml
# Nginx ingress example
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qwen-omni-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
spec:
  tls:
  - hosts:
    - qwen.example.com
    secretName: qwen-tls
  rules:
  - host: qwen.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: qwen-omni
            port:
              number: 80
```

### Network Policies

Restrict pod communication:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: qwen-omni-policy
spec:
  podSelector:
    matchLabels:
      app: qwen-omni
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress
    ports:
    - port: 7860
```

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

---

## Backup and Recovery

### Conversation History

```bash
# Backup
kubectl cp qwen-omni-pod:/app/conversation_history ./backup

# Restore
kubectl cp ./backup qwen-omni-pod:/app/conversation_history
```

### Model Cache

Mount persistent volume for HuggingFace cache to avoid re-downloading.

---

## Updates

### Rolling Update

```bash
# Update image
kubectl set image deployment/qwen-omni \
  qwen-omni=qwen-omni:v2.0 \
  -n qwen-omni

# Monitor rollout
kubectl rollout status deployment/qwen-omni -n qwen-omni

# Rollback if needed
kubectl rollout undo deployment/qwen-omni -n qwen-omni
```
