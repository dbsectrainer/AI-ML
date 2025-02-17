# Cloud Platforms for Machine Learning

This section covers major cloud platforms for machine learning operations and deployment.

## Contents

1. [AWS SageMaker](./aws-sagemaker/)
   - Model development
   - Training at scale
   - Deployment options
   - Monitoring tools

2. [Google Cloud AI](./google-cloud-ai/)
   - Vertex AI
   - AutoML
   - AI Platform
   - TPU integration

3. [Azure ML](./azure-ml/)
   - Azure ML Studio
   - Automated ML
   - Model management
   - Deployment services

## Platform Comparison

### 1. Development Experience
| Feature | AWS SageMaker | Google Cloud AI | Azure ML |
|---------|---------------|-----------------|-----------|
| Notebooks | SageMaker Notebooks | Vertex AI Workbench | Azure Notebooks |
| IDEs | Studio | Cloud Code | VS Code Integration |
| SDKs | Python, Java, etc. | Python, Java, etc. | Python, R, etc. |
| AutoML | SageMaker Autopilot | Vertex AI AutoML | Azure AutoML |

### 2. Training Capabilities
| Feature | AWS SageMaker | Google Cloud AI | Azure ML |
|---------|---------------|-----------------|-----------|
| Distributed Training | ✓ | ✓ | ✓ |
| Custom Containers | ✓ | ✓ | ✓ |
| Hyperparameter Tuning | ✓ | ✓ | ✓ |
| Spot Instances | ✓ | ✓ | ✓ |

### 3. Deployment Options
| Feature | AWS SageMaker | Google Cloud AI | Azure ML |
|---------|---------------|-----------------|-----------|
| Real-time Inference | ✓ | ✓ | ✓ |
| Batch Inference | ✓ | ✓ | ✓ |
| Edge Deployment | ✓ | ✓ | ✓ |
| Multi-model Endpoints | ✓ | ✓ | ✓ |

## AWS SageMaker

### 1. Development
```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Initialize estimator
estimator = PyTorch(
    entry_point='train.py',
    role=role,
    framework_version='1.8.1',
    py_version='py36',
    instance_count=1,
    instance_type='ml.p3.2xlarge'
)

# Start training
estimator.fit({'training': 's3://bucket/training-data'})
```

### 2. Deployment
```python
# Deploy model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Make predictions
predictions = predictor.predict(data)
```

## Google Cloud AI

### 1. Model Training
```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='your-project')

# Create training job
job = aiplatform.CustomTrainingJob(
    display_name='training-job',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training'
)

# Start training
model = job.run(
    dataset=dataset,
    model_display_name='my-model'
)
```

### 2. Model Deployment
```python
# Deploy model
endpoint = model.deploy(
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=5
)

# Make predictions
predictions = endpoint.predict(instances=instances)
```

## Azure ML

### 1. Experiment Tracking
```python
from azureml.core import Workspace, Experiment

# Connect to workspace
ws = Workspace.from_config()

# Create experiment
experiment = Experiment(workspace=ws, name='my-experiment')

# Start run
run = experiment.start_logging()
run.log('accuracy', 0.95)
run.complete()
```

### 2. Model Registration
```python
from azureml.core import Model

# Register model
model = Model.register(
    workspace=ws,
    model_path='outputs/model.pkl',
    model_name='my-model',
    description='My trained model'
)
```

## Best Practices

### 1. Development
- Use version control
- Implement CI/CD
- Document everything
- Test thoroughly
- Monitor costs

### 2. Deployment
- Start small
- Scale gradually
- Monitor performance
- Set up alerts
- Plan for rollback

### 3. Security
- Use IAM roles
- Encrypt data
- Secure endpoints
- Monitor access
- Regular audits

## Cost Optimization

### 1. Development
- Use spot instances
- Clean up resources
- Optimize storage
- Right-size instances
- Use auto-scaling

### 2. Production
- Monitor usage
- Set budgets
- Use reserved instances
- Optimize endpoints
- Regular review

## Resources

### Documentation
- [AWS SageMaker Docs](https://docs.aws.amazon.com/sagemaker/)
- [Google Cloud AI Docs](https://cloud.google.com/ai-platform/docs)
- [Azure ML Docs](https://docs.microsoft.com/azure/machine-learning/)

### Tools
- AWS CLI
- Google Cloud SDK
- Azure CLI
- Cloud management tools

### Tutorials
- Getting started guides
- Best practices
- Sample projects
- Video tutorials

## Assessment Questions

1. How do you choose between cloud platforms?
2. What are the key considerations for ML deployment?
3. How do you optimize costs in cloud ML platforms?
4. What are the best practices for security?

## Projects

1. Set up ML pipelines
2. Implement auto-scaling
3. Create monitoring dashboards
4. Develop cost optimization strategies
