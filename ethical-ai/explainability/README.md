# Model Explainability in Machine Learning

This section covers techniques and tools for making machine learning models interpretable and their decisions explainable.

## Explainability Techniques

### 1. SHAP (SHapley Additive exPlanations)
```python
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def explain_with_shap(model, X_train, X_test):
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Feature importance
    feature_importance = np.abs(shap_values).mean(0)
    
    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance
    }

# Visualization
def plot_shap_summary(shap_values, features):
    shap.summary_plot(shap_values, features)
```

### 2. LIME (Local Interpretable Model-agnostic Explanations)
```python
from lime import lime_tabular

def explain_with_lime(model, X_train, X_test, feature_names):
    # Initialize LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        mode='classification'
    )
    
    # Explain a single prediction
    def explain_instance(instance):
        exp = explainer.explain_instance(
            instance, 
            model.predict_proba,
            num_features=len(feature_names)
        )
        return exp
    
    return explain_instance
```

### 3. Feature Importance
```python
from sklearn.inspection import permutation_importance

def analyze_feature_importance(model, X, y):
    # Built-in feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    
    # Permutation importance
    result = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=42
    )
    
    return {
        'feature_importances': importance,
        'permutation_importance': result.importances_mean
    }
```

## Model-Specific Techniques

### 1. Decision Trees
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def visualize_decision_tree(model, feature_names, class_names):
    plt.figure(figsize=(20,10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True
    )
    plt.show()
```

### 2. Linear Models
```python
import eli5
from eli5.sklearn import PermutationImportance

def explain_linear_model(model, X, feature_names):
    # Get coefficients
    coefficients = pd.DataFrame(
        model.coef_,
        columns=feature_names
    )
    
    # Permutation importance
    perm = PermutationImportance(model).fit(X, y)
    
    # Generate HTML explanation
    html_exp = eli5.show_weights(
        model,
        feature_names=feature_names
    )
    
    return {
        'coefficients': coefficients,
        'permutation': perm,
        'html_explanation': html_exp
    }
```

## Global Interpretability

### 1. Partial Dependence Plots
```python
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

def plot_partial_dependence(model, X, features):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pdp = partial_dependence(
        model, X,
        features=[features],
        kind='average'
    )
    
    plt.plot(pdp[1][0], pdp[0][0])
    plt.xlabel(features)
    plt.ylabel('Partial dependence')
    return fig
```

### 2. Global Surrogate Models
```python
from sklearn.tree import DecisionTreeClassifier

def create_surrogate_model(complex_model, X, feature_names):
    # Get predictions from complex model
    y_pred = complex_model.predict(X)
    
    # Train interpretable surrogate model
    surrogate = DecisionTreeClassifier(max_depth=3)
    surrogate.fit(X, y_pred)
    
    # Visualize surrogate model
    visualize_decision_tree(
        surrogate,
        feature_names=feature_names,
        class_names=['0', '1']
    )
    
    return surrogate
```

## Local Interpretability

### 1. Individual Prediction Explanation
```python
def explain_prediction(model, instance, explainers):
    explanations = {}
    
    # SHAP explanation
    if 'shap' in explainers:
        shap_values = explainers['shap'].shap_values(instance)
        explanations['shap'] = shap_values
    
    # LIME explanation
    if 'lime' in explainers:
        lime_exp = explainers['lime'].explain_instance(
            instance,
            model.predict_proba
        )
        explanations['lime'] = lime_exp
    
    return explanations
```

### 2. Counterfactual Explanations
```python
from alibi.explainers import CounterfactualProto

def generate_counterfactuals(model, instance, target_class):
    # Initialize explainer
    cf = CounterfactualProto(
        model,
        shape=instance.shape,
        target_class=target_class
    )
    
    # Generate explanation
    explanation = cf.explain(instance)
    
    return {
        'counterfactual': explanation.cf,
        'distance': explanation.distance,
        'target_class': target_class
    }
```

## Visualization Tools

### 1. Interactive Dashboards
```python
import dash
from dash import dcc, html

def create_explanation_dashboard(model, data, explanations):
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        dcc.Graph(
            figure=create_feature_importance_plot(explanations)
        ),
        dcc.Graph(
            figure=create_prediction_explanation_plot(explanations)
        )
    ])
    
    return app
```

### 2. Report Generation
```python
def generate_explanation_report(model, data, explanations):
    """
    Generate a comprehensive explanation report
    """
    report = {
        'model_summary': {
            'type': type(model).__name__,
            'performance_metrics': calculate_metrics(model, data)
        },
        'global_explanations': {
            'feature_importance': analyze_feature_importance(model, data),
            'model_behavior': analyze_model_behavior(model, data)
        },
        'local_explanations': {
            'example_cases': generate_example_explanations(model, data)
        }
    }
    return report
```

## Best Practices

### 1. Model Development
- Start with interpretable models
- Document model assumptions
- Validate explanations
- Test with stakeholders
- Maintain simplicity

### 2. Explanation Methods
- Use multiple techniques
- Consider audience needs
- Validate explanations
- Document limitations
- Regular updates

### 3. Documentation
- Model cards
- Explanation methods
- Limitations
- Use cases
- Version control

## Resources

### Books
- "Interpretable Machine Learning" by Christoph Molnar
- "Explanatory Model Analysis" by Przemyslaw Biecek
- "Machine Learning Interpretability with Python" by Dipanjan Sarkar

### Tools
- SHAP
- LIME
- ELI5
- InterpretML
- AIX360

### Papers
- "A Unified Approach to Interpreting Model Predictions" (SHAP)
- "Why Should I Trust You?: Explaining the Predictions of Any Classifier" (LIME)
- "The Mythos of Model Interpretability"

## Assessment Questions

1. How do you choose between different explanation methods?
2. What are the trade-offs between accuracy and interpretability?
3. How do you validate the quality of explanations?
4. What are the best practices for explaining models to different audiences?

## Projects

1. Build an interpretable ML pipeline
2. Create an explanation dashboard
3. Implement multiple explanation methods
4. Develop a model card generator
