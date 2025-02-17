# Bias Detection in Machine Learning

This section covers techniques and tools for detecting and mitigating bias in machine learning models.

## Types of Bias

### 1. Data Bias
- Sampling bias
- Selection bias
- Reporting bias
- Temporal bias
- Geographical bias

### 2. Model Bias
- Algorithm bias
- Representation bias
- Measurement bias
- Aggregation bias
- Evaluation bias

## Fairness Metrics

### 1. Group Fairness Metrics
```python
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset

def calculate_group_metrics(dataset, privileged_groups, unprivileged_groups):
    metrics = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    return {
        'disparate_impact': metrics.disparate_impact(),
        'statistical_parity_difference': metrics.statistical_parity_difference(),
        'equal_opportunity_difference': metrics.equal_opportunity_difference()
    }
```

### 2. Individual Fairness Metrics
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def individual_fairness(predictions, sensitive_features, distance_matrix):
    """
    Measure individual fairness using the consistency score
    """
    n_samples = len(predictions)
    fairness_score = 0
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pred_diff = abs(predictions[i] - predictions[j])
            feat_dist = distance_matrix[i, j]
            fairness_score += abs(pred_diff - feat_dist)
            
    return fairness_score / (n_samples * (n_samples - 1) / 2)
```

## Bias Detection Techniques

### 1. Statistical Analysis
```python
def analyze_dataset_bias(df, protected_attribute, target):
    """
    Analyze dataset for potential biases
    """
    # Demographics analysis
    demographics = df[protected_attribute].value_counts(normalize=True)
    
    # Outcome analysis by group
    outcomes_by_group = df.groupby(protected_attribute)[target].mean()
    
    # Statistical significance test
    from scipy import stats
    groups = df[protected_attribute].unique()
    group1_outcomes = df[df[protected_attribute] == groups[0]][target]
    group2_outcomes = df[df[protected_attribute] == groups[1]][target]
    
    t_stat, p_value = stats.ttest_ind(group1_outcomes, group2_outcomes)
    
    return {
        'demographics': demographics,
        'outcomes_by_group': outcomes_by_group,
        't_statistic': t_stat,
        'p_value': p_value
    }
```

### 2. Visual Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bias_analysis(df, protected_attribute, target):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution of protected attribute
    sns.countplot(data=df, x=protected_attribute, ax=axes[0,0])
    axes[0,0].set_title('Distribution of Protected Attribute')
    
    # Outcome distribution by group
    sns.boxplot(data=df, x=protected_attribute, y=target, ax=axes[0,1])
    axes[0,1].set_title('Outcome Distribution by Group')
    
    # Correlation heatmap
    sns.heatmap(df.corr(), ax=axes[1,0])
    axes[1,0].set_title('Feature Correlations')
    
    # Outcome probability density by group
    for group in df[protected_attribute].unique():
        sns.kdeplot(
            data=df[df[protected_attribute] == group][target],
            label=str(group),
            ax=axes[1,1]
        )
    axes[1,1].set_title('Outcome Density by Group')
    
    plt.tight_layout()
    return fig
```

## Bias Mitigation Strategies

### 1. Pre-processing Techniques
```python
from aif360.algorithms.preprocessing import Reweighing

def mitigate_bias_preprocessing(dataset, privileged_groups, unprivileged_groups):
    """
    Apply pre-processing bias mitigation
    """
    reweighing = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    return reweighing.fit_transform(dataset)
```

### 2. In-processing Techniques
```python
from aif360.algorithms.inprocessing import AdversarialDebiasing

def train_debiased_model(dataset, privileged_groups, unprivileged_groups):
    """
    Train a model with in-processing debiasing
    """
    debiased_model = AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name='debiased_classifier'
    )
    return debiased_model.fit(dataset)
```

### 3. Post-processing Techniques
```python
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

def apply_postprocessing(predictions, dataset, privileged_groups, unprivileged_groups):
    """
    Apply post-processing bias mitigation
    """
    postprocessing = CalibratedEqOddsPostprocessing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups
    )
    return postprocessing.fit_predict(dataset, predictions)
```

## Testing Framework

### 1. Bias Test Suite
```python
class BiasTestSuite:
    def __init__(self, model, dataset, protected_attributes):
        self.model = model
        self.dataset = dataset
        self.protected_attributes = protected_attributes
        
    def run_tests(self):
        results = {}
        
        # Demographic parity test
        results['demographic_parity'] = self.test_demographic_parity()
        
        # Equal opportunity test
        results['equal_opportunity'] = self.test_equal_opportunity()
        
        # Disparate impact test
        results['disparate_impact'] = self.test_disparate_impact()
        
        return results
        
    def test_demographic_parity(self):
        # Implementation
        pass
        
    def test_equal_opportunity(self):
        # Implementation
        pass
        
    def test_disparate_impact(self):
        # Implementation
        pass
```

## Best Practices

### 1. Data Collection
- Representative sampling
- Balanced datasets
- Documentation of collection process
- Quality assurance
- Bias awareness

### 2. Model Development
- Regular bias testing
- Multiple fairness metrics
- Cross-validation across groups
- Stakeholder feedback
- Continuous monitoring

### 3. Documentation
- Data collection methods
- Bias mitigation decisions
- Model limitations
- Testing results
- Mitigation strategies

## Resources

### Books
- "Fairness and Machine Learning" by Solon Barocas
- "The Ethical Algorithm" by Michael Kearns
- "Weapons of Math Destruction" by Cathy O'Neil

### Tools
- AI Fairness 360 (IBM)
- Fairlearn (Microsoft)
- What-If Tool (Google)
- Aequitas (UChicago)

### Papers
- "A Survey on Bias and Fairness in Machine Learning"
- "Fair Classification and Social Welfare"
- "Equality of Opportunity in Supervised Learning"

## Assessment Questions

1. How do you identify different types of bias in datasets?
2. What are the trade-offs between different fairness metrics?
3. How do you choose appropriate bias mitigation strategies?
4. What are the best practices for continuous bias monitoring?

## Projects

1. Build a bias detection pipeline
2. Implement multiple fairness metrics
3. Create a bias visualization dashboard
4. Develop a comprehensive testing framework
