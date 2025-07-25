# Model Deployment in MLOps

This section covers strategies and best practices for deploying machine learning models to production.

## Model Serving

### 1. REST API Development
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionInput(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(input_data: PredictionInput):
    prediction = model.predict([input_data.features])
    return {"prediction": prediction.tolist()}
```

### 2. gRPC Service
```python
import grpc
from concurrent import futures
import prediction_pb2
import prediction_pb2_grpc

class PredictionService(prediction_pb2_grpc.PredictorServicer):
    def __init__(self):
        self.model = joblib.load('model.pkl')
        
    def Predict(self, request, context):
        features = request.features
        prediction = self.model.predict([features])
        return prediction_pb2.PredictionResponse(
            prediction=prediction[0]
        )
```

## Containerization

### 1. Docker Configuration
```dockerfile
# Dockerfile for ML model serving
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model/ model/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose Setup
```yaml
version: '3'
services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model
    environment:
      - MODEL_PATH=/app/model/model.pkl
      - LOG_LEVEL=INFO
    
  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Kubernetes Deployment

### 1. Deployment Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-service
  template:
    metadata:
      labels:
        app: model-service
    spec:
      containers:
      - name: model-container
        image: model-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1"
```

### 2. Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Model Versioning

### 1. Version Control
```python
class ModelRegistry:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        
    def save_model(self, model, version: str, metadata: Dict):
        """Save model with version and metadata"""
        model_path = f"{self.storage_path}/{version}"
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        joblib.dump(model, f"{model_path}/model.pkl")
        
        # Save metadata
        with open(f"{model_path}/metadata.json", 'w') as f:
            json.dump(metadata, f)
            
    def load_model(self, version: str):
        """Load model by version"""
        model_path = f"{self.storage_path}/{version}"
        return joblib.load(f"{model_path}/model.pkl")
```

### 2. A/B Testing
```python
class ABTestingRouter:
    def __init__(self, model_a, model_b, split_ratio: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        
    def predict(self, features):
        """Route prediction to either model A or B"""
        if random.random() < self.split_ratio:
            return {
                'model': 'A',
                'prediction': self.model_a.predict(features)
            }
        else:
            return {
                'model': 'B',
                'prediction': self.model_b.predict(features)
            }
```

## Performance Optimization

### 1. Model Optimization
```python
import onnx
import tensorflow as tf

def optimize_model(model_path: str, output_path: str):
    """Optimize model for inference"""
    # Convert to ONNX
    model = tf.keras.models.load_model(model_path)
    onnx_model = tf2onnx.convert.from_keras(model)
    
    # Optimize
    optimized_model = onnx.optimizer.optimize(onnx_model)
    onnx.save(optimized_model, output_path)
```

### 2. Batch Processing
```python
class BatchPredictor:
    def __init__(self, model, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.queue = []
        
    async def add_to_queue(self, features):
        """Add request to queue"""
        self.queue.append(features)
        if len(self.queue) >= self.batch_size:
            return await self.process_batch()
        return None
        
    async def process_batch(self):
        """Process a batch of requests"""
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        return self.model.predict(batch)
```

## Monitoring and Logging

### 1. Metrics Collection
```python
from prometheus_client import Counter, Histogram

# Define metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions'
)

latency_histogram = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction'
)

# Use in API
@app.post("/predict")
@latency_histogram.time()
async def predict(input_data: PredictionInput):
    prediction = model.predict([input_data.features])
    prediction_counter.inc()
    return {"prediction": prediction.tolist()}
```

### 2. Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure logging for the application"""
    logger = logging.getLogger('model_service')
    logger.setLevel(logging.INFO)
    
    # File handler
    handler = RotatingFileHandler(
        'model_service.log',
        maxBytes=10000000,
        backupCount=5
    )
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger
```

## Best Practices

### 1. Deployment
- Use containers
- Implement health checks
- Set resource limits
- Enable monitoring
- Plan for scaling

### 2. Security
- Encrypt data
- Authenticate requests
- Rate limit APIs
- Monitor access
- Regular updates

### 3. Performance
- Optimize models
- Use caching
- Implement batching
- Monitor latency
- Scale horizontally

## Resources

### Documentation
- FastAPI
- Docker
- Kubernetes
- Prometheus
- Grafana

### Tools
- Model serving frameworks
- Containerization tools
- Orchestration platforms
- Monitoring systems

### Tutorials
- Deployment guides
- Performance tuning
- Security best practices
- Monitoring setup

## Assessment Questions

1. How do you choose between different serving options?
2. What are key considerations for containerization?
3. How do you implement effective monitoring?
4. What are best practices for scaling ML services?

## Projects

1. Build a model serving API
2. Create a containerized deployment
3. Set up monitoring and logging
4. Implement A/B testing
