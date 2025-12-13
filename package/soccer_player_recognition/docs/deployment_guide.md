# Deployment Guide

## Overview

This guide covers deploying the Soccer Player Recognition System to production environments, including cloud platforms, containers, and on-premises setups. It covers different deploymente strategies, scaling considerations, monitoring, and maintenance.

## Table of Contents

1. [Deployment Architecture](#deployment-architecture)
2. [Environment Setup](#environment-setup)
3. [Container Deployment](#container-deployment)
4. [Cloud Platform Deployment](#cloud-platform-deployment)
5. [On-Premises Deployment](#on-premises-deployment)
6. [API Service Deployment](#api-service-deployment)
7. [Scaling Strategies](#scaling-strategies)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Security Considerations](#security-considerations)
10. [Maintenance and Updates](#maintenance-and-updates)

## Deployment Architecture

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer / API Gateway              │
├─────────────────────────────────────────────────────────────┤
│  Web Server (Nginx/Apache)  │  Web Server (Nginx/Apache)    │
├─────────────────────────────────────────────────────────────┤
│         Application Servers (Multiple Instances)             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ PlayerRecognizer│  │ PlayerRecognizer│  │ PlayerRec...│ │
│  │ Instance 1      │  │ Instance 2      │  │ Instance N  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│            Model Storage & Caching Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Model Cache    │  │  Image Cache    │  │ Result Cache│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Storage Backend                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Model Storage  │  │  Input Storage  │  │ Output Store│ │
│  │  (S3/GCS/Azure) │  │  (Database)     │  │  (Database)  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Patterns

#### 1. Microservices Architecture
```yaml
services:
  detection-service:
    image: soccer-recognition:detection
    replicas: 3
    environment:
      - MODEL_TYPE=rf_detr
    
  identification-service:
    image: soccer-recognition:identification
    replicas: 3
    environment:
      - MODEL_TYPE=siglip
      
  segmentation-service:
    image: soccer-recognition:segmentation
    replicas: 2
    environment:
      - MODEL_TYPE=sam2
      
  classification-service:
    image: soccer-recognition:classification
    replicas: 2
    environment:
      - MODEL_TYPE=resnet
```

#### 2. Monolithic Deployment
```yaml
services:
  soccer-recognition-api:
    image: soccer-recognition:latest
    replicas: 5
    environment:
      - ENABLE_ALL_MODELS=true
      - MEMORY_EFFICIENT=false
```

#### 3. Hybrid Architecture
```yaml
services:
  web-api:
    image: soccer-recognition:api
    replicas: 3
    
  inference-workers:
    image: soccer-recognition:worker
    replicas: 10
    environment:
      - GPU_REQUIRED=true
```

## Environment Setup

### Production Environment Variables

```bash
# Core Configuration
export SOCCER_MODEL_PATH="/models/pretrained"
export SOCCER_DEVICE="cuda:0"
export SOCCER_MEMORY_EFFICIENT="false"
export SOCCER_LOG_LEVEL="INFO"
export SOCCER_LOG_DIR="/var/log/soccer-recognition"

# Database Configuration
export SOCCER_DB_HOST="postgres.internal"
export SOCCER_DB_PORT="5432"
export SOCCER_DB_NAME="soccer_recognition"
export SOCCER_DB_USER="soccer_user"
export SOCCER_DB_PASSWORD="${DB_PASSWORD}"

# Redis Configuration
export SOCCER_REDIS_HOST="redis.internal"
export SOCCER_REDIS_PORT="6379"
export SOCCER_REDIS_DB="0"

# Cloud Storage
export SOCCER_STORAGE_TYPE="s3"  # s3, gcs, azure
export SOCCER_STORAGE_BUCKET="soccer-models-prod"
export SOCCER_AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY}"
export SOCCER_AWS_SECRET_ACCESS_KEY="${AWS_SECRET_KEY}"

# API Configuration
export SOCCER_API_HOST="0.0.0.0"
export SOCCER_API_PORT="8080"
export SOCCER_API_WORKERS="4"
export SOCCER_API_TIMEOUT="300"

# Security
export SOCCER_API_KEY="${API_KEY}"
export SOCCER_RATE_LIMIT="1000"
export SOCCER_CORS_ORIGINS="https://yourdomain.com"
```

### Configuration Files

#### production_config.yaml
```yaml
# Production Configuration
project:
  name: "Soccer Recognition API"
  version: "1.0.0"
  environment: "production"

system:
  device: "cuda:0"
  memory_efficient: false
  mixed_precision: true
  gradient_checkpointing: false
  max_memory_fraction: 0.8
  num_workers: 8

models:
  rf_detr:
    model_path: "models/rf_detr_epoch_20.pth"
    confidence_threshold: 0.6
    nms_threshold: 0.5
    batch_size: 4
    
  siglip:
    model_path: "models/siglip-vit-so400m-14-e384.pt"
    temperature: 100.0
    batch_size: 32
    
  sam2:
    model_path: "models/sam2_hiera_large.pt"
    confidence_threshold: 0.75
    batch_size: 1
    
  resnet:
    model_path: "models/resnet50-0676ba61.pth"
    architecture: "resnet50"
    batch_size: 64

performance:
  cache_size_mb: 2048
  enable_caching: true
  cache_ttl: 3600
  remove_dropout_inference: true
  fuse_conv_bn: true

logging:
  level: "INFO"
  log_file_path: "/var/log/soccer-recognition/app.log"
  log_performance_metrics: true
  log_model_summary: false

api:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  timeout: 300
  rate_limit: 1000
  cors_origins: ["https://yourdomain.com"]
```

## Container Deployment

### Docker Setup

#### Dockerfile
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python path
ENV PYTHONPATH=/app:/app/soccer_player_recognition
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install application
RUN pip3 install -e .

# Create directories
RUN mkdir -p /app/models /app/logs /app/data /app/outputs

# Set up non-root user
RUN useradd -m -u 1000 soccer && \
    chown -R soccer:soccer /app
USER soccer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "-m", "soccer_player_recognition.api.server"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  soccer-recognition-api:
    build: .
    container_name: soccer-recognition-api
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - SOCCER_ENV=production
      - SOCCER_MODEL_PATH=/app/models
      - SOCCER_LOG_DIR=/app/logs
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./data:/app/data
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - soccer-network
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:13
    container_name: soccer-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=soccer_recognition
      - POSTGRES_USER=soccer_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - soccer-network

  redis:
    image: redis:6-alpine
    container_name: soccer-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - soccer-network

  nginx:
    image: nginx:alpine
    container_name: soccer-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - soccer-recognition-api
    networks:
      - soccer-network

volumes:
  postgres_data:
  redis_data:

networks:
  soccer-network:
    driver: bridge
```

#### .dockerignore
```dockerfile
.git
.gitignore
README.md
.env
*.log
.ipynb_checkpoints
__pycache__
*.pyc
*.pyo
*.pyd
.Python
.venv
venv/
.coverage
.pytest_cache/
docs/
tests/
*.md
```

### Kubernetes Deployment

#### k8s-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: soccer-recognition-api
  labels:
    app: soccer-recognition
    version: v1.0.0
spec:
  replicas: 5
  selector:
    matchLabels:
      app: soccer-recognition
  template:
    metadata:
      labels:
        app: soccer-recognition
        version: v1.0.0
    spec:
      containers:
      - name: soccer-recognition
        image: soccer-recognition:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
        env:
        - name: SOCCER_ENV
          value: "production"
        - name: SOCCER_MODEL_PATH
          value: "/models"
        - name: SOCCER_LOG_DIR
          value: "/logs"
        - name: SOCCER_REDIS_HOST
          value: "redis-service"
        - name: SOCCER_DB_HOST
          value: "postgres-service"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
        - name: log-storage
          mountPath: /logs
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: models-pvc
      - name: log-storage
        persistentVolumeClaim:
          claimName: logs-pvc
      nodeSelector:
        gpu: "true"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: soccer-recognition-service
spec:
  selector:
    app: soccer-recognition
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: soccer-recognition-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: soccer-recognition-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Cloud Platform Deployment

### AWS Deployment

#### AWS ECS Configuration
```json
{
  "family": "soccer-recognition",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/soccerTaskRole",
  "containerDefinitions": [
    {
      "name": "soccer-recognition",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/soccer-recognition:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "SOCCER_ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/soccer-recognition",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ]
    }
  ]
}
```

#### AWS EKS with NVIDIA GPU Operator
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: soccer-recognition
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: soccer-recognition-api
  namespace: soccer-recognition
spec:
  replicas: 3
  selector:
    matchLabels:
      app: soccer-recognition-api
  template:
    metadata:
      labels:
        app: soccer-recognition-api
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: soccer-recognition
        image: soccer-recognition:latest
        ports:
        - containerPort: 8080
        env:
        - name: SOCCER_ENV
          value: "production"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: soccer-recognition-service
  namespace: soccer-recognition
spec:
  selector:
    app: soccer-recognition-api
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Google Cloud Platform

#### Cloud Run Configuration
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: soccer-recognition-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/gpu: "1"
        run.googleapis.com/cpu: "4"
        run.googleapis.com/memory: "16Gi"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/PROJECT_ID/soccer-recognition:latest
        ports:
        - containerPort: 8080
        env:
        - name: SOCCER_ENV
          value: "production"
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Azure Deployment

#### Azure Container Instances
```yaml
apiVersion: 2021-03-01
location: eastus
name: soccer-recognition-rg
resourceGroup: soccer-recognition-rg
tags:
  project: soccer-recognition
---
apiVersion: 2021-03-01
type: Microsoft.ContainerInstance/containerGroups
name: soccer-recognition-cg
location: eastus
resourceGroup: soccer-recognition-rg
properties:
  containers:
  - name: soccer-recognition
    properties:
      image: yourregistry.azurecr.io/soccer-recognition:latest
      resources:
        requests:
          cpu: 4
          memoryInGb: 16
          gpu:
            count: 1
            sku: V100
      ports:
      - port: 8080
        protocol: TCP
      environmentVariables:
      - name: SOCCER_ENV
        value: production
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - port: 8080
      protocol: tcp
```

## On-Premises Deployment

### System Requirements

#### Minimum Hardware Requirements
- **CPU**: Intel Xeon E5-2680 v4 (8 cores minimum)
- **GPU**: NVIDIA RTX 3080 or better (10GB+ VRAM)
- **RAM**: 32GB DDR4
- **Storage**: 500GB NVMe SSD
- **Network**: 1Gbps Ethernet

#### Recommended Hardware
- **CPU**: Intel Xeon Gold 6248R (24 cores)
- **GPU**: NVIDIA A100 or RTX 4090 (24GB+ VRAM)
- **RAM**: 128GB DDR4 ECC
- **Storage**: 2TB NVMe SSD RAID 1
- **Network**: 10Gbps Ethernet

#### High-Performance Setup
- **CPU**: Intel Xeon Platinum 8380 (40 cores)
- **GPU**: NVIDIA A100 80GB (multiple GPUs)
- **RAM**: 256GB DDR4 ECC
- **Storage**: 4TB NVMe SSD RAID 10
- **Network**: 25Gbps Ethernet with RDMA

### Installation Scripts

#### install_production.sh
```bash
#!/bin/bash
set -e

echo "🚀 Installing Soccer Recognition System - Production"

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install NVIDIA drivers
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-11-8 -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2

# Install additional dependencies
sudo apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    nginx \
    postgresql \
    redis-server

# Create application directory
sudo mkdir -p /opt/soccer-recognition
sudo chown $USER:$USER /opt/soccer-recognition

# Create systemd service
sudo tee /etc/systemd/system/soccer-recognition.service > /dev/null <<EOF
[Unit]
Description=Soccer Recognition API
After=network.target

[Service]
Type=notify
User=soccer
Group=soccer
WorkingDirectory=/opt/soccer-recognition
ExecStart=/opt/soccer-recognition/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
Restart=always
RestartSec=10
Environment=PATH=/opt/soccer-recognition/venv/bin

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Installation completed!"
echo "Please run: newgrp docker"
echo "Then execute: ./setup_application.sh"
```

#### setup_application.sh
```bash
#!/bin/bash
set -e

echo "🔧 Setting up Soccer Recognition Application"

# Clone repository
cd /opt/soccer-recognition
git clone https://github.com/your-org/soccer-player-recognition.git .

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Download models
python scripts/download_models.py

# Setup database
sudo -u postgres createdb soccer_recognition
sudo -u postgres psql -c "CREATE USER soccer_user WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE soccer_recognition TO soccer_user;"

# Setup Nginx
sudo tee /etc/nginx/sites-available/soccer-recognition > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        access_log off;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/soccer-recognition /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Enable and start services
sudo systemctl enable soccer-recognition
sudo systemctl start soccer-recognition
sudo systemctl enable postgresql
sudo systemctl enable redis-server

echo "✅ Application setup completed!"
echo "🌐 Service should be running at http://your-domain.com"
```

## API Service Deployment

### FastAPI Server Implementation

```python
# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from typing import List, Optional

from soccer_player_recognition import PlayerRecognizer, save_results
from app.models import AnalysisRequest, AnalysisResponse
from app.cache import CacheManager
from app.database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Soccer Player Recognition API",
    description="Advanced AI system for soccer player analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SOCCER_CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
cache_manager = CacheManager()
db_manager = DatabaseManager()
recognizer = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global recognizer
    
    logger.info("Starting Soccer Recognition API")
    
    # Initialize recognizer
    recognizer = PlayerRecognizer(
        config_path="config/production_config.yaml",
        device=os.getenv("SOCCER_DEVICE", "cuda"),
        memory_efficient=False
    )
    
    logger.info("PlayerRecognizer initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Soccer Recognition API")
    
    if recognizer:
        recognizer.cleanup_memory()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": recognizer.get_model_status() if recognizer else False,
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        if recognizer and recognizer.get_model_status():
            return {"status": "ready"}
        raise HTTPException(status_code=503, detail="Models not loaded")
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/api/v1/analyze/image", response_model=AnalysisResponse)
async def analyze_image(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze single image"""
    try:
        # Check cache first
        cache_key = f"image_{hash(request.image_path)}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        # Load and process image
        image = load_image(request.image_path)
        
        # Perform analysis
        results = recognizer.analyze_scene(
            image,
            player_candidates=request.player_candidates,
            team_context=request.team_context,
            analysis_type=request.analysis_type,
            confidence_threshold=request.confidence_threshold
        )
        
        # Format response
        response = AnalysisResponse(
            image_path=request.image_path,
            analysis_type=request.analysis_type,
            results=results,
            processing_time=results['metadata']['total_time']
        )
        
        # Cache result
        cache_manager.set(cache_key, response, ttl=3600)
        
        # Store in database
        background_tasks.add_task(
            db_manager.store_analysis,
            request.image_path,
            response
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze/video")
async def analyze_video(
    video_file: UploadFile = File(...),
    player_candidates: Optional[List[str]] = None,
    background_tasks: BackgroundTasks = None
):
    """Analyze video file"""
    try:
        # Save uploaded video
        video_path = f"/tmp/{video_file.filename}"
        with open(video_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Process video
        results = recognizer.analyze_scene(
            video_path,
            player_candidates=player_candidates or [],
            analysis_type="full"
        )
        
        # Clean up uploaded file
        os.remove(video_path)
        
        return {
            "video_filename": video_file.filename,
            "results": results,
            "total_frames": results.get('metadata', {}).get('num_frames', 0)
        }
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_statistics():
    """Get system statistics"""
    if not recognizer:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "performance_stats": recognizer.get_performance_stats(),
        "model_status": recognizer.get_model_status(),
        "cache_stats": cache_manager.get_stats()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("SOCCER_API_HOST", "0.0.0.0"),
        port=int(os.getenv("SOCCER_API_PORT", 8080)),
        workers=int(os.getenv("SOCCER_API_WORKERS", 4))
    )
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/soccer-recognition
upstream soccer_backend {
    server 127.0.0.1:8080;
    server 127.0.0.1:8081 backup;
    server 127.0.0.1:8082 backup;
    
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;

server {
    listen 80;
    server_name soccer-recognition.yourdomain.com;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Client body size limit (for video uploads)
    client_max_body_size 500M;
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://soccer_backend/health;
        access_log off;
    }
    
    # API endpoints with rate limiting
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://soccer_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # File upload endpoint with stricter rate limiting
    location /api/v1/analyze/video {
        limit_req zone=upload burst=5 nodelay;
        
        proxy_pass http://soccer_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Extended timeouts for video processing
        proxy_connect_timeout 10s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        
        # Disable buffering for uploads
        proxy_buffering off;
        proxy_request_buffering off;
    }
}

# SSL configuration
server {
    listen 443 ssl http2;
    server_name soccer-recognition.yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # Same location blocks as above
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
```

## Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration
```nginx
# /etc/nginx/upstream.conf
upstream soccer_backend {
    least_conn;
    
    server 10.0.1.10:8080 weight=3;
    server 10.0.1.11:8080 weight=3;
    server 10.0.1.12:8080 weight=2;
    server 10.0.1.13:8080 weight=2;
    
    # Health checks
    check interval=3000 rise=2 fall=5 type=http;
    check_http_send "GET /health HTTP/1.0\r\n\r\n";
    check_http_expect_alive http_2xx http_3xx;
}
```

#### Auto-scaling Configuration
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: soccer-recognition-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: soccer-recognition-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

### Vertical Scaling

#### GPU Resource Management
```python
# Dynamic GPU allocation
class GPUResourceManager:
    def __init__(self):
        self.gpu_pool = GPUPool()
        self.request_queue = Queue()
        
    def allocate_gpu(self, job_type: str, memory_requirement: int) -> GPU:
        """Allocate GPU based on job requirements"""
        if job_type == "detection":
            return self.gpu_pool.allocate(
                min_memory=4*1024**3,  # 4GB
                max_memory=8*1024**3   # 8GB
            )
        elif job_type == "segmentation":
            return self.gpu_pool.allocate(
                min_memory=8*1024**3,  # 8GB
                max_memory=16*1024**3  # 16GB
            )
        
    def optimize_batch_size(self, gpu: GPU, model_type: str) -> int:
        """Optimize batch size for GPU"""
        if model_type == "rf_detr":
            return min(gpu.memory // (1024**3) * 2, 8)
        elif model_type == "sam2":
            return 1  # SAM2 is memory intensive
        else:
            return 4
```

### Database Scaling

#### PostgreSQL Configuration
```sql
-- Enable connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Optimize for read-heavy workload
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

SELECT pg_reload_conf();
```

#### Redis Configuration
```conf
# /etc/redis/redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
tcp-keepalive 300
timeout 0
tcp-backlog 511
```

## Monitoring and Logging

### Prometheus Configuration

#### prometheus.yml
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "soccer_rules.yml"

scrape_configs:
  - job_name: 'soccer-recognition-api'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
      
  - job_name: 'nvidia-dcgm-exporter'
    static_configs:
      - targets: ['localhost:9400']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### soccer_rules.yml
```yaml
groups:
- name: soccer-recognition
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }}s"
      
  - alert: GPUMemoryUsage
    expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "GPU memory usage critical"
      description: "GPU memory usage is {{ $value | humanizePercentage }}"
      
  - alert: ModelLoadFailure
    expr: soccer_model_loaded == 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Model failed to load"
      description: "One or more models failed to load"
```

### Grafana Dashboard

#### grafana-dashboard.json
```json
{
  "dashboard": {
    "title": "Soccer Recognition Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "singlestat",
        "targets": [
          {
            "expr": "nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100",
            "legendFormat": "GPU Memory %"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "table",
        "targets": [
          {
            "expr": "soccer_model_inference_time_seconds",
            "legendFormat": "{{model}}"
          }
        ]
      }
    ]
  }
}
```

### ELK Stack Configuration

#### logstash.conf
```ruby
input {
  file {
    path => "/var/log/soccer-recognition/app.log"
    start_position => "beginning"
    codec => "json"
  }
}

filter {
  if [level] == "ERROR" {
    mutate {
      add_tag => ["alert"]
    }
  }
  
  if [request_id] {
    mutate {
      add_field => {"@metadata" => {"request_id" => "%{request_id}"}}
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "soccer-logs-%{+YYYY.MM.dd}"
  }
  
  if "alert" in [tags] {
    email {
      to => "alerts@yourdomain.com"
      subject => "Soccer Recognition Alert"
      body => "%{message}"
    }
  }
}
```

## Security Considerations

### Authentication and Authorization

#### JWT Token Middleware
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            os.getenv("JWT_SECRET_KEY"),
            algorithms=["HS256"]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.get("/api/v1/protected")
async def protected_endpoint(user_id: str = Depends(verify_token)):
    """Protected endpoint requiring authentication"""
    return {"message": f"Hello user {user_id}"}
```

#### Rate Limiting Implementation
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/analyze/image")
@limiter.limit("10/minute")
async def analyze_image(
    request: Request,
    analysis_request: AnalysisRequest,
    user_id: str = Depends(verify_token)
):
    """Rate-limited endpoint"""
    # Implementation here
    pass
```

### Network Security

#### Firewall Configuration
```bash
#!/bin/bash
# firewall.sh

# Reset iptables
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (restrict to specific IP)
iptables -A INPUT -p tcp --dport 22 -s YOUR_IP -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow internal network
iptables -A INPUT -s 10.0.0.0/8 -j ACCEPT

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "DROPPED: "

# Drop all other traffic
iptables -A INPUT -j DROP
```

#### SSL/TLS Configuration
```nginx
# SSL configuration
server {
    listen 443 ssl http2;
    server_name soccer-recognition.yourdomain.com;
    
    # SSL certificates
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    
    # SSL protocols and ciphers
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    
    # SSL session configuration
    ssl_session_cache shared:SSL:50m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    
    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/nginx/ssl/chain.pem;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    # Location blocks
    location / {
        proxy_pass http://backend;
        # ... proxy configuration
    }
}
```

### Data Protection

#### Encryption at Rest
```python
# Database encryption
from cryptography.fernet import Fernet

class EncryptedDatabaseManager:
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)
    
    def encrypt_result(self, result: dict) -> bytes:
        """Encrypt analysis results"""
        json_data = json.dumps(result).encode()
        return self.cipher.encrypt(json_data)
    
    def decrypt_result(self, encrypted_data: bytes) -> dict:
        """Decrypt analysis results"""
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
```

#### Secure File Handling
```python
import hashlib
import os
from pathlib import Path

class SecureFileHandler:
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate uploaded file"""
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi'}
        if file_path.suffix.lower() not in allowed_extensions:
            return False
        
        # Check file size (max 500MB)
        if file_path.stat().st_size > 500 * 1024 * 1024:
            return False
        
        # Check file signature
        with open(file_path, 'rb') as f:
            header = f.read(16)
            if not self._is_valid_file_header(header):
                return False
        
        return True
    
    def _is_valid_file_header(self, header: bytes) -> bool:
        """Check file magic numbers"""
        # JPEG
        if header[:2] == b'\xff\xd8':
            return True
        # PNG
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return True
        # MP4
        if header[:4] in [b'ftyp', b'isom']:
            return True
        return False
    
    def secure_filename(self, original_filename: str) -> str:
        """Generate secure filename"""
        # Remove path traversal
        filename = Path(original_filename).name
        
        # Generate hash
        hash_obj = hashlib.sha256(filename.encode())
        hash_suffix = hash_obj.hexdigest()[:16]
        
        # Keep extension
        extension = Path(filename).suffix
        safe_filename = f"{hash_suffix}{extension}"
        
        return safe_filename
```

## Maintenance and Updates

### Update Strategy

#### Blue-Green Deployment
```bash
#!/bin/bash
# blue_green_deploy.sh

set -e

CURRENT_VERSION=$(kubectl get deployment soccer-recognition-blue -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d':' -f2)

echo "Current version: $CURRENT_VERSION"
read -p "Enter new version: " NEW_VERSION

echo "🚀 Starting blue-green deployment"

# Deploy to green environment
kubectl set image deployment/soccer-recognition-green \
    soccer-recognition=soccer-recognition:$NEW_VERSION

# Wait for green deployment
echo "⏳ Waiting for green deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/soccer-recognition-green

# Run health checks
echo "🔍 Running health checks..."
python scripts/health_check.py --version $NEW_VERSION

# Switch traffic to green
kubectl patch service soccer-recognition-service \
    -p '{"spec":{"selector":{"version":"green"}}}'

echo "✅ Deployment successful!"
echo "Traffic switched to version $NEW_VERSION"

# Clean up blue deployment after delay
echo "🧹 Blue deployment will be cleaned up in 10 minutes..."
(sleep 600 && kubectl delete deployment soccer-recognition-blue) &
```

#### Rollback Script
```bash
#!/bin/bash
# rollback.sh

CURRENT_VERSION=$(kubectl get service soccer-recognition-service -o jsonpath='{.spec.selector.version}')

echo "Current environment: $CURRENT_VERSION"
echo "Available versions:"
echo "  - blue (previous)"
echo "  - green (current)"

read -p "Rollback to version: " TARGET_VERSION

echo "🔄 Rolling back to $TARGET_VERSION..."

# Switch traffic
kubectl patch service soccer-recognition-service \
    -p "{\"spec\":{\"selector\":{\"version\":\"$TARGET_VERSION\"}}}"

# Verify rollback
echo "⏳ Waiting for rollback to complete..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/soccer-recognition-$TARGET_VERSION

echo "✅ Rollback completed!"

# Update version tracking
kubectl label service soccer-recognition-service version=$TARGET_VERSION --overwrite
```

### Backup and Recovery

#### Database Backup
```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p $BACKUP_DIR

echo "💾 Starting database backup..."

# PostgreSQL backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | \
    gzip > $BACKUP_DIR/database.sql.gz

# Model files backup
tar -czf $BACKUP_DIR/models.tar.gz /opt/soccer-recognition/models/

# Configuration backup
tar -czf $BACKUP_DIR/config.tar.gz /opt/soccer-recognition/config/

# Upload to S3
aws s3 sync $BACKUP_DIR s3://soccer-backups/$(date +%Y-%m-%d)/

echo "✅ Backup completed: $BACKUP_DIR"

# Cleanup old backups (keep last 30 days)
find /backups -type d -mtime +30 -exec rm -rf {} +
```

#### Model Versioning
```python
# model_manager.py
import semver
from pathlib import Path

class ModelManager:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
    
    def get_latest_model(self, model_type: str) -> Path:
        """Get latest model version"""
        model_path = self.models_dir / model_type
        versions = [f.stem for f in model_path.glob("v*")]
        return max(versions, key=semver.Version.parse)
    
    def deploy_model(self, model_type: str, model_path: Path, version: str):
        """Deploy new model version"""
        target_path = self.models_dir / model_type / f"v{version}"
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        import shutil
        shutil.copy2(model_path, target_path / "model.pth")
        
        # Create metadata
        metadata = {
            "version": version,
            "model_type": model_type,
            "deployed_at": datetime.now().isoformat(),
            "model_path": str(target_path / "model.pth")
        }
        
        with open(target_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Update symlink to latest
        latest_path = self.models_dir / model_type / "latest"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(target_path)
        
        print(f"✅ Model {model_type} v{version} deployed successfully")
```

### Monitoring and Alerts

#### Health Check Script
```python
# scripts/health_check.py
import requests
import json
import time
import argparse
from datetime import datetime

def check_api_health(base_url: str):
    """Check API health"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)

def check_model_loading(base_url: str):
    """Check model loading status"""
    try:
        response = requests.get(f"{base_url}/api/v1/stats", timeout=30)
        if response.status_code == 200:
            stats = response.json()
            return stats.get('model_status', {})
        else:
            return {}
    except Exception as e:
        return {}

def performance_test(base_url: str, num_requests: int = 10):
    """Run performance test"""
    print(f"Running performance test with {num_requests} requests...")
    
    times = []
    for i in range(num_requests):
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/api/v1/analyze/image",
                json={
                    "image_path": "test_image.jpg",
                    "analysis_type": "detection"
                },
                timeout=60
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if i % 5 == 0:
                print(f"  Completed {i+1}/{num_requests} requests")
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    avg_time = sum(times) / len(times) if times else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0
    
    print(f"\nPerformance Results:")
    print(f"  Average response time: {avg_time:.3f}s")
    print(f"  Min response time: {min_time:.3f}s")
    print(f"  Max response time: {max_time:.3f}s")
    print(f"  Success rate: {len(times)}/{num_requests} ({len(times)/num_requests*100:.1f}%)")
    
    return {
        'average_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'success_rate': len(times) / num_requests
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--version", help="Expected model version")
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    
    print(f"🏥 Health check at {datetime.now()}")
    print(f"Target URL: {args.base_url}")
    
    # Check API health
    is_healthy, health_data = check_api_health(args.base_url)
    print(f"API Health: {'✅' if is_healthy else '❌'}")
    
    if not is_healthy:
        print(f"Error: {health_data}")
        return 1
    
    # Check models
    model_status = check_model_loading(args.base_url)
    print("\nModel Status:")
    for model, status in model_status.get('models', {}).items():
        loaded = status.get('loaded', False)
        print(f"  {model}: {'✅' if loaded else '❌'}")
    
    # Version check
    if args.version:
        current_version = model_status.get('version', 'unknown')
        if current_version == args.version:
            print(f"✅ Version check passed: {current_version}")
        else:
            print(f"❌ Version mismatch: expected {args.version}, got {current_version}")
            return 1
    
    # Performance test
    if args.performance:
        perf_results = performance_test(args.base_url)
        if perf_results['success_rate'] < 0.9:
            print(f"❌ Performance test failed: success rate {perf_results['success_rate']:.1%}")
            return 1
        elif perf_results['average_time'] > 5.0:
            print(f"⚠️  Slow performance: average time {perf_results['average_time']:.3f}s")
        else:
            print(f"✅ Performance test passed")
    
    print("\n🎉 All health checks passed!")
    return 0

if __name__ == "__main__":
    exit(main())
```

This comprehensive deployment guide covers all aspects of deploying the Soccer Player Recognition System to production. It includes multiple deploymente strategies, scaling approaches, security considerations, and maintenance procedures. Choose the approprite sections based on your specific requirements and deploymente environment.