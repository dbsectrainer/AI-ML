# Calculus for Machine Learning

This section covers the calculus concepts essential for understanding machine learning algorithms, particularly in optimization and neural networks.

## Fundamentals

### 1. Derivatives
- Definition and intuition
- Rules of differentiation
  * Power rule
  * Product rule
  * Chain rule
- Partial derivatives
- Directional derivatives
- Gradient vectors

### 2. Integration
- Definite and indefinite integrals
- Integration techniques
- Multiple integrals
- Applications in probability

### 3. Multivariable Calculus
- Functions of multiple variables
- Partial derivatives
- Gradient vectors
- Hessian matrices
- Jacobian matrices

### 4. Vector Calculus
- Vector fields
- Line integrals
- Surface integrals
- Divergence and curl
- Green's theorem

## Applications in Machine Learning

### 1. Optimization
- Gradient descent
- Learning rate
- Local vs global minima
- Saddle points
- Backpropagation

### 2. Neural Networks
- Chain rule in backpropagation
- Activation functions
- Loss function optimization
- Gradient flow

### 3. Information Theory
- Entropy
- Cross-entropy
- KL divergence
- Maximum likelihood estimation

## Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple gradient descent implementation
def gradient_descent(f, df, x0, learning_rate=0.1, n_iterations=100):
    x = x0
    history = [x]
    
    for _ in range(n_iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return x, history

# Example function: f(x) = x^2
f = lambda x: x**2
df = lambda x: 2*x

# Run gradient descent
x_min, history = gradient_descent(f, df, x0=2.0)

# Plotting
x = np.linspace(-2, 2, 100)
plt.plot(x, f(x))
plt.plot(history, [f(x) for x in history], 'ro-')
plt.show()
```

## Exercises

1. Implement gradient descent from scratch
2. Calculate partial derivatives for common ML loss functions
3. Derive backpropagation equations
4. Optimize simple neural networks manually

## Common Derivatives in ML

### Activation Functions
```
1. Sigmoid: σ(x) = 1/(1 + e^(-x))
   σ'(x) = σ(x)(1 - σ(x))

2. tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
   tanh'(x) = 1 - tanh^2(x)

3. ReLU: f(x) = max(0, x)
   f'(x) = 0 if x < 0, 1 if x > 0
```

### Loss Functions
```
1. MSE: L = (1/n)Σ(y - ŷ)²
   ∂L/∂ŷ = (-2/n)(y - ŷ)

2. Cross-Entropy: L = -Σ(y log(ŷ))
   ∂L/∂ŷ = -y/ŷ
```

## Resources

### Books
- "Calculus" by James Stewart
- "Vector Calculus" by Marsden and Tromba
- "Deep Learning" by Goodfellow, Bengio, and Courville (Chapter 4)

### Online Courses
- MIT 18.01 Single Variable Calculus
- MIT 18.02 Multivariable Calculus
- 3Blue1Brown Essence of Calculus

### Tools
- SymPy for symbolic mathematics
- Matplotlib for visualization
- GeoGebra for interactive plots

## Assessment Questions

1. Explain the relationship between derivatives and optimization
2. How does the chain rule apply to neural network backpropagation?
3. Why are second derivatives important in optimization?
4. What role do partial derivatives play in gradient descent?

## Projects

1. Implement various optimization algorithms
2. Visualize gradient descent in 2D and 3D
3. Build a neural network without using frameworks
4. Create an interactive tool for exploring derivatives
