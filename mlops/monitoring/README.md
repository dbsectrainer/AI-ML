# ML Model Monitoring

This section covers comprehensive monitoring strategies for machine learning models in production.

## Performance Monitoring

### 1. Metric Collection
```python
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class ModelMetrics:
    model_id: str
    timestamp: datetime
    accuracy: float
    latency_ms: float
    throughput: int
    memory_usage: float
    
class MetricsCollector:
    def __init__(self):
        self.metrics_store = []
        
    def record_prediction(self, prediction, actual, latency_ms):
        """Record prediction metrics"""
        metrics = ModelMetrics(
            model_id=self.model_id,
            timestamp=datetime.now(),
            accuracy=float(prediction == actual),
            latency_ms=latency_ms,
            throughput=self.calculate_throughput(),
            memory_usage=self.get_memory_usage()
        )
        self.metrics_store.append(metrics)
```

### 2. Performance Dashboard
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_performance_dashboard(metrics: List[ModelMetrics]):
    """Create interactive performance dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Latency', 'Throughput', 'Memory')
    )
    
    # Accuracy trend
    fig.add_trace(
        go.Scatter(
            x=[m.timestamp for m in metrics],
            y=[m.accuracy for m in metrics],
            name="Accuracy"
        ),
        row=1, col=1
    )
    
    # Add other metrics...
    return fig
```

## Data Drift Detection

### 1. Feature Drift Monitor
```python
class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_stats = self._calculate_stats(reference_data)
        self.drift_threshold = 0.1
        
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Detect drift in feature distributions"""
        current_stats = self._calculate_stats(current_data)
        drift_scores = {}
        
        for feature in self.reference_stats:
            drift_scores[feature] = self._calculate_drift(
                self.reference_stats[feature],
                current_stats[feature]
            )
            
        return drift_scores
        
    def _calculate_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate distribution statistics"""
        stats = {}
        for column in data.columns:
            stats[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'quantiles': data[column].quantile([0.25, 0.5, 0.75])
            }
        return stats
```

### 2. Concept Drift Detection
```python
class ConceptDriftDetector:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.baseline_performance = None
        self.current_window = []
        
    def update(self, prediction: float, actual: float) -> bool:
        """Update detector with new prediction"""
        performance = self._calculate_performance(prediction, actual)
        self.current_window.append(performance)
        
        if len(self.current_window) >= self.window_size:
            if self.baseline_performance is None:
                self.baseline_performance = np.mean(self.current_window)
            else:
                current_performance = np.mean(self.current_window)
                if self._is_significant_drift(current_performance):
                    return True
            self.current_window = []
        return False
```

## System Monitoring

### 1. Resource Monitoring
```python
import psutil
import GPUtil

class SystemMonitor:
    def collect_metrics(self) -> Dict:
        """Collect system resource metrics"""
        metrics = {
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'load_avg': psutil.getloadavg()
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'percent': psutil.disk_usage('/').percent
            }
        }
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            metrics['gpu'] = [{
                'id': gpu.id,
                'load': gpu.load,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal
            } for gpu in gpus]
        except:
            metrics['gpu'] = None
            
        return metrics
```

### 2. Alert System
```python
class AlertSystem:
    def __init__(self):
        self.alert_rules = []
        self.notification_channels = []
        
    def add_rule(self, metric: str, threshold: float, 
                 comparison: str, duration: int):
        """Add monitoring rule"""
        self.alert_rules.append({
            'metric': metric,
            'threshold': threshold,
            'comparison': comparison,
            'duration': duration
        })
        
    def check_alerts(self, metrics: Dict):
        """Check metrics against alert rules"""
        alerts = []
        for rule in self.alert_rules:
            if self._check_rule(rule, metrics):
                alerts.append(self._create_alert(rule, metrics))
        
        if alerts:
            self._send_alerts(alerts)
```

## Model Health Checks

### 1. Health Check API
```python
from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    checks = {
        'model_status': await check_model_health(),
        'system_resources': await check_system_resources(),
        'data_quality': await check_data_quality(),
        'dependencies': await check_dependencies()
    }
    
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return checks
```

### 2. Automated Testing
```python
class ModelHealthTest:
    def __init__(self, model, test_data: pd.DataFrame):
        self.model = model
        self.test_data = test_data
        
    def run_health_checks(self) -> Dict[str, bool]:
        """Run suite of health checks"""
        results = {
            'prediction_check': self._check_predictions(),
            'performance_check': self._check_performance(),
            'latency_check': self._check_latency(),
            'memory_check': self._check_memory_usage()
        }
        return results
```

## Best Practices

### 1. Monitoring Strategy
- Define KPIs
- Set thresholds
- Implement alerts
- Regular reviews
- Automated responses

### 2. Data Quality
- Validate inputs
- Check distributions
- Monitor missing values
- Track feature correlations
- Detect anomalies

### 3. System Health
- Resource monitoring
- Performance tracking
- Error logging
- Dependency checks
- Security monitoring

## Tools and Technologies

### 1. Monitoring Tools
- Prometheus
- Grafana
- ELK Stack
- DataDog
- New Relic

### 2. Visualization
- Plotly
- Bokeh
- Streamlit
- Dash
- Tableau

### 3. Alerting
- PagerDuty
- OpsGenie
- Slack
- Email
- SMS

## Resources

### Documentation
- Monitoring frameworks
- Alerting systems
- Visualization tools
- Best practices guides

### Tools
- System monitoring
- Data quality checks
- Performance tracking
- Alert management

### Tutorials
- Setting up monitoring
- Implementing alerts
- Creating dashboards
- Drift detection

## Assessment Questions

1. How do you detect data drift effectively?
2. What metrics are crucial for model monitoring?
3. How do you set up alerting thresholds?
4. What are best practices for system monitoring?

## Projects

1. Build a monitoring dashboard
2. Implement drift detection
3. Create an alerting system
4. Set up automated health checks
