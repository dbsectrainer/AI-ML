# Optimization for Machine Learning

This section covers optimization techniques used in machine learning, focusing on methods for minimizing loss functions and training neural networks.

## Fundamentals

### 1. Gradient-Based Methods
- Gradient descent
  * Batch gradient descent
  * Stochastic gradient descent (SGD)
  * Mini-batch gradient descent
- Learning rate scheduling
- Momentum
- Nesterov accelerated gradient

### 2. Advanced Optimizers
- AdaGrad
- RMSprop
- Adam
- AdamW
- LAMB
- RAdam

### 3. Constrained Optimization
- Lagrange multipliers
- Karush-Kuhn-Tucker (KKT) conditions
- Convex optimization
- Linear programming
- Quadratic programming

### 4. Non-Gradient Methods
- Grid search
- Random search
- Bayesian optimization
- Evolutionary algorithms
- Simulated annealing

## Optimization Challenges

### 1. Local Optima
- Local vs global minima
- Saddle points
- Plateau regions
- Escaping local optima

### 2. Convergence Issues
- Learning rate selection
- Vanishing/exploding gradients
- Ill-conditioning
- Batch size effects

### 3. Regularization
- L1 regularization (Lasso)
- L2 regularization (Ridge)
- Elastic Net
- Dropout
- Early stopping

## Code Examples

```python
import numpy as np
from typing import Callable, List, Tuple

class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        self.velocity = self.momentum * self.velocity - self.learning_rate * grads
        return params + self.velocity

class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

## Learning Rate Scheduling

### 1. Step Decay
```python
def step_decay(initial_lr: float, epoch: int, drop_rate: float = 0.5, 
               epochs_drop: int = 10) -> float:
    return initial_lr * np.power(drop_rate, np.floor(epoch/epochs_drop))
```

### 2. Exponential Decay
```python
def exp_decay(initial_lr: float, epoch: int, k: float = 0.1) -> float:
    return initial_lr * np.exp(-k * epoch)
```

### 3. Cosine Annealing
```python
def cosine_annealing(initial_lr: float, epoch: int, total_epochs: int) -> float:
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
```

## Practical Tips

1. Learning Rate Selection
   - Start with a small learning rate (e.g., 0.001)
   - Use learning rate finder techniques
   - Monitor loss curves for oscillation

2. Optimizer Selection
   - SGD: Good for understanding and theory
   - Adam: Good default choice for deep learning
   - AdamW: Better generalization for large models

3. Batch Size Considerations
   - Larger batches: More stable gradients
   - Smaller batches: Better generalization
   - Power of 2 sizes for GPU efficiency

## Resources

### Books
- "Convex Optimization" by Boyd and Vandenberghe
- "Numerical Optimization" by Nocedal and Wright
- "Deep Learning" by Goodfellow et al. (Chapter 8)

### Online Courses
- Stanford CS229: Machine Learning
- Fast.ai: Practical Deep Learning
- Coursera: Deep Learning Specialization

### Papers
- "Adam: A Method for Stochastic Optimization"
- "On the Convergence of Adam and Beyond"
- "Fixing Weight Decay Regularization in Adam"

## Assessment Questions

1. Compare and contrast different optimization algorithms
2. How do learning rate schedules affect model training?
3. What are the trade-offs between batch size and convergence?
4. Explain the benefits and drawbacks of adaptive learning rates

## Projects

1. Implement various optimizers from scratch
2. Create a learning rate finder tool
3. Visualize optimization landscapes
4. Compare optimizer performance on different architectures
