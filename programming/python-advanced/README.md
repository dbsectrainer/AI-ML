# Advanced Python for Machine Learning

This section covers advanced Python concepts and libraries essential for machine learning development.

## Core Python Concepts

### 1. Python Language Features
- Decorators and metaclasses
- Generators and iterators
- Context managers
- Type hints and annotations
- Lambda functions
- List/dictionary comprehensions

### 2. Object-Oriented Programming
- Classes and inheritance
- Abstract base classes
- Properties and descriptors
- Magic methods
- Design patterns

### 3. Memory Management
- Memory allocation
- Garbage collection
- Memory profiling
- Memory optimization
- Reference counting

### 4. Concurrency
- Threading vs Multiprocessing
- AsyncIO
- Parallel processing
- Concurrent.futures
- Queue management

## Essential Libraries

### 1. NumPy
```python
import numpy as np

# Array operations
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr_transpose = arr.T
arr_mean = np.mean(arr, axis=0)

# Broadcasting
broadcast_add = arr + np.array([1, 2, 3])

# Linear algebra
eigenvalues, eigenvectors = np.linalg.eig(arr @ arr.T)
```

### 2. Pandas
```python
import pandas as pd

# Data loading and manipulation
df = pd.read_csv('data.csv')
df_grouped = df.groupby('category').agg({
    'value': ['mean', 'std'],
    'count': 'sum'
})

# Time series
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.resample('1D').mean()
```

### 3. Matplotlib/Seaborn
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Basic plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='feature1', y='feature2', hue='target')
plt.title('Feature Relationship')
plt.show()

# Advanced visualization
g = sns.FacetGrid(df, col='category', row='subcategory')
g.map(sns.histplot, 'value')
```

## Performance Optimization

### 1. Vectorization
```python
# Bad: Using loops
def slow_function(x):
    result = []
    for i in x:
        result.append(i ** 2 + 2*i + 1)
    return result

# Good: Vectorized
def fast_function(x):
    return x ** 2 + 2*x + 1
```

### 2. Profiling
```python
import cProfile
import line_profiler

# Function profiling
def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    # Your code here
    profiler.disable()
    profiler.print_stats(sort='cumtime')
```

### 3. Cython Integration
```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fast_ops.pyx"),
)
```

## Testing and Debugging

### 1. Unit Testing
```python
import unittest
import numpy as np

class TestMLFunctions(unittest.TestCase):
    def setUp(self):
        self.X = np.random.randn(100, 10)
        self.y = np.random.randint(0, 2, 100)
    
    def test_normalization(self):
        normalized = (self.X - self.X.mean()) / self.X.std()
        self.assertTrue(np.allclose(normalized.mean(), 0))
        self.assertTrue(np.allclose(normalized.std(), 1))
```

### 2. Debugging Tools
```python
import pdb
import logging

# Setting up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_function():
    logger.debug("Starting function")
    pdb.set_trace()  # Interactive debugging
    # Your code here
    logger.debug("Function completed")
```

## Best Practices

### 1. Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Write docstrings (Google or NumPy style)
- Type hints for better code understanding

### 2. Project Structure
```
project/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── tests/
├── README.md
└── setup.py
```

### 3. Environment Management
```bash
# Creating virtual environment
python -m venv env

# Requirements management
pip freeze > requirements.txt
pip install -r requirements.txt
```

## Resources

### Books
- "Fluent Python" by Luciano Ramalho
- "High Performance Python" by Micha Gorelick
- "Python for Data Analysis" by Wes McKinney

### Online Courses
- Real Python
- Python for Data Science (Coursera)
- Advanced Python Programming (edX)

### Tools
- PyCharm/VSCode with Python extensions
- Jupyter Notebooks/Lab
- Black code formatter
- Pylint/Flake8

## Assessment Questions

1. Explain the difference between NumPy arrays and Python lists
2. How does vectorization improve performance?
3. What are the best practices for handling large datasets?
4. Describe common debugging strategies for ML code

## Projects

1. Build a data processing pipeline
2. Implement ML algorithms from scratch
3. Create a performance monitoring tool
4. Develop a custom data visualization library
