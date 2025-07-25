# CI/CD for Machine Learning

This section covers continuous integration and continuous deployment practices for machine learning projects.

## Pipeline Automation

### 1. GitHub Actions Workflow
```yaml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        python -m pytest tests/
        
  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Train model
      run: python train.py
      
    - name: Save model artifacts
      uses: actions/upload-artifact@v2
      with:
        name: model-artifacts
        path: models/

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        python deploy.py
```

### 2. Jenkins Pipeline
```groovy
pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Data Validation') {
            steps {
                sh 'python validate_data.py'
            }
        }
        
        stage('Train Model') {
            steps {
                sh 'python train.py'
            }
        }
        
        stage('Model Validation') {
            steps {
                sh 'python validate_model.py'
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'python deploy.py'
            }
        }
    }
}
```

## Testing Strategies

### 1. Unit Tests
```python
import pytest
from ml_project.preprocessing import DataPreprocessor

def test_preprocessor():
    """Test data preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Test data cleaning
    data = pd.DataFrame({
        'feature1': [1, 2, None, 4],
        'feature2': ['a', 'b', 'c', 'd']
    })
    
    cleaned_data = preprocessor.clean_data(data)
    assert cleaned_data.isnull().sum().sum() == 0
    
    # Test feature engineering
    engineered_data = preprocessor.engineer_features(cleaned_data)
    assert 'feature1_squared' in engineered_data.columns
```

### 2. Integration Tests
```python
class TestMLPipeline:
    def setup_method(self):
        """Setup test environment"""
        self.pipeline = MLPipeline()
        self.test_data = load_test_data()
        
    def test_end_to_end_training(self):
        """Test complete training pipeline"""
        # Train model
        model = self.pipeline.train(self.test_data)
        
        # Validate model artifacts
        assert os.path.exists('models/model.pkl')
        assert os.path.exists('models/scaler.pkl')
        
        # Check model performance
        metrics = self.pipeline.evaluate(model, self.test_data)
        assert metrics['accuracy'] > 0.8
```

### 3. Performance Tests
```python
def test_model_performance():
    """Test model performance requirements"""
    model = load_model('models/model.pkl')
    
    # Test prediction latency
    start_time = time.time()
    prediction = model.predict(test_input)
    latency = time.time() - start_time
    assert latency < 0.1  # 100ms threshold
    
    # Test memory usage
    memory_usage = get_model_memory_usage(model)
    assert memory_usage < 500  # 500MB threshold
```

## Version Control

### 1. Model Versioning
```python
from datetime import datetime
import mlflow

class ModelVersionControl:
    def __init__(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("ml-project")
        
    def log_model_version(self, model, metrics: Dict, params: Dict):
        """Log new model version"""
        with mlflow.start_run():
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log additional artifacts
            mlflow.log_artifacts("artifacts")
```

### 2. Data Versioning
```python
import dvc
import dvc.api

class DataVersionControl:
    def __init__(self):
        self.repo = dvc.api.Repo()
        
    def version_dataset(self, data_path: str, version: str):
        """Version a dataset"""
        # Add data to DVC
        self.repo.add(data_path)
        
        # Create version tag
        self.repo.scm.tag(version)
        
        # Push to remote storage
        self.repo.push()
```

## Artifact Management

### 1. Model Registry
```python
class ModelRegistry:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        
    def register_model(self, model, metadata: Dict):
        """Register new model version"""
        version = self._get_next_version()
        model_path = f"{self.storage_path}/{version}"
        
        # Save model
        self._save_model(model, model_path)
        
        # Save metadata
        self._save_metadata(metadata, model_path)
        
        return version
        
    def promote_to_production(self, version: str):
        """Promote model version to production"""
        source = f"{self.storage_path}/{version}"
        target = f"{self.storage_path}/production"
        
        # Create symbolic link
        if os.path.exists(target):
            os.remove(target)
        os.symlink(source, target)
```

### 2. Artifact Storage
```python
class ArtifactStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path
        
    def save_artifacts(self, artifacts: Dict, version: str):
        """Save training artifacts"""
        artifact_path = f"{self.base_path}/{version}"
        os.makedirs(artifact_path, exist_ok=True)
        
        for name, artifact in artifacts.items():
            path = f"{artifact_path}/{name}"
            self._save_artifact(artifact, path)
            
    def load_artifacts(self, version: str) -> Dict:
        """Load artifacts for specific version"""
        artifact_path = f"{self.base_path}/{version}"
        return self._load_artifacts(artifact_path)
```

## Best Practices

### 1. Pipeline Design
- Modular components
- Clear dependencies
- Error handling
- Logging
- Monitoring

### 2. Testing
- Comprehensive coverage
- Automated testing
- Performance testing
- Integration testing
- Regression testing

### 3. Version Control
- Code versioning
- Data versioning
- Model versioning
- Documentation
- Change tracking

## Resources

### Documentation
- GitHub Actions
- Jenkins
- MLflow
- DVC
- pytest

### Tools
- CI/CD platforms
- Version control systems
- Testing frameworks
- Artifact management
- Monitoring tools

### Tutorials
- Setting up CI/CD
- Implementing tests
- Version control
- Deployment automation

## Assessment Questions

1. How do you design effective ML pipelines?
2. What testing strategies are crucial for ML?
3. How do you manage model versions?
4. What are best practices for CI/CD in ML?

## Projects

1. Set up automated ML pipeline
2. Implement comprehensive testing
3. Create version control system
4. Build artifact management system
