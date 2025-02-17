# Probability and Statistics for Machine Learning

This section covers fundamental concepts in probability theory and statistics that are essential for machine learning.

## Probability Theory

### 1. Basic Probability Concepts
- Sample spaces and events
- Probability axioms
- Conditional probability
- Independence
- Random variables

### 2. Probability Distributions
- Discrete distributions
  * Bernoulli
  * Binomial
  * Poisson
- Continuous distributions
  * Normal (Gaussian)
  * Uniform
  * Exponential
- Joint distributions
- Marginal distributions

### 3. Bayes' Theorem
- Conditional probability
- Prior and posterior probabilities
- Maximum likelihood estimation
- Maximum a posteriori estimation

## Statistics

### 1. Descriptive Statistics
- Measures of central tendency
  * Mean
  * Median
  * Mode
- Measures of dispersion
  * Variance
  * Standard deviation
  * Quartiles and percentiles
- Correlation and covariance

### 2. Inferential Statistics
- Sampling theory
- Central Limit Theorem
- Confidence intervals
- Hypothesis testing
  * Null and alternative hypotheses
  * p-values
  * Type I and Type II errors
- Statistical tests
  * t-test
  * chi-square test
  * F-test
  * ANOVA

## Applications in Machine Learning

### 1. Model Evaluation
- Cross-validation
- Confusion matrices
- ROC curves
- AUC-ROC analysis

### 2. Probabilistic Models
- Naive Bayes classifiers
- Gaussian Mixture Models
- Hidden Markov Models
- Probabilistic Graphical Models

## Code Examples

```python
import numpy as np
from scipy import stats

# Generate random data from normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)

# Descriptive statistics
mean = np.mean(data)
std = np.std(data)
variance = np.var(data)

# Hypothesis testing
t_stat, p_value = stats.ttest_1samp(data, 0)

# Confidence intervals
confidence_interval = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(len(data)))

# Probability distributions
x = np.linspace(-3, 3, 100)
normal_pdf = stats.norm.pdf(x, loc=0, scale=1)
```

## Exercises

1. Implement basic probability calculations
2. Calculate confidence intervals from scratch
3. Perform hypothesis testing on real datasets
4. Build a Naive Bayes classifier

## Resources

### Books
- "Introduction to Probability" by Bertsekas and Tsitsiklis
- "Statistical Inference" by Casella and Berger
- "Think Stats" by Allen Downey

### Online Courses
- MIT 18.05 Introduction to Probability and Statistics
- Khan Academy Statistics and Probability
- Coursera "Statistics with Python" Specialization

### Tools
- Python (NumPy, SciPy, statsmodels)
- R statistical computing
- MATLAB Statistics Toolbox

## Assessment Questions

1. Explain the difference between probability and likelihood
2. How does the Central Limit Theorem apply to machine learning?
3. What is the relationship between confidence intervals and hypothesis testing?
4. When would you use a t-test versus a chi-square test?

## Projects

1. Build a spam classifier using Naive Bayes
2. Analyze A/B test results
3. Implement bootstrap resampling
4. Create a probability distribution visualization tool
