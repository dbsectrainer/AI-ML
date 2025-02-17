# Data Preprocessing for Machine Learning

This section covers essential techniques for preparing data for machine learning models.

## Data Cleaning

### 1. Handling Missing Values
```python
import pandas as pd
import numpy as np

def handle_missing_values(df):
    # Numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df
```

### 2. Outlier Detection
```python
def detect_outliers(data, n_std=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(y - mean) / std for y in data]
    return np.abs(z_scores) > n_std

# IQR method
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    outlier_mask = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    return outlier_mask
```

### 3. Data Validation
```python
from typing import Dict, Any
import pandas as pd

def validate_data(df: pd.DataFrame, rules: Dict[str, Any]) -> bool:
    """
    Validate dataframe against a set of rules
    
    rules = {
        'age': {'type': 'int', 'min': 0, 'max': 120},
        'email': {'type': 'str', 'pattern': r'[^@]+@[^@]+\.[^@]+'},
        'income': {'type': 'float', 'min': 0}
    }
    """
    for column, rule in rules.items():
        if column not in df.columns:
            return False
            
        if rule.get('type') == 'int':
            if not pd.api.types.is_integer_dtype(df[column]):
                return False
        
        if 'min' in rule:
            if df[column].min() < rule['min']:
                return False
                
        if 'max' in rule:
            if df[column].max() > rule['max']:
                return False
                
    return True
```

## Feature Engineering

### 1. Numerical Features
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def create_polynomial_features(X, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)
```

### 2. Categorical Features
```python
def encode_categorical_features(df, columns):
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=columns)
    
    # Label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in columns:
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        
    return df_encoded, df
```

### 3. Text Features
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def process_text_features(texts, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    else:
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english'
        )
    
    return vectorizer.fit_transform(texts)
```

## Time Series Processing

### 1. Feature Creation
```python
def create_time_features(df, date_column):
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['hour'] = df[date_column].dt.hour
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    return df
```

### 2. Lag Features
```python
def create_lag_features(df, column, lags):
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

def create_rolling_features(df, column, windows):
    for window in windows:
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window).std()
    return df
```

## Data Pipeline

### 1. Pipeline Creation
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_preprocessing_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
```

### 2. Custom Transformers
```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_params):
        self.feature_params = feature_params
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        # Custom transformation logic
        return X_transformed
```

## Best Practices

### 1. Data Quality Checks
- Check for missing values
- Validate data types
- Ensure feature ranges
- Check for duplicates
- Verify data consistency

### 2. Performance Optimization
- Use efficient data structures
- Implement batch processing
- Utilize parallel processing
- Optimize memory usage
- Cache intermediate results

### 3. Documentation
- Document preprocessing steps
- Track feature engineering decisions
- Maintain transformation history
- Version control preprocessing code
- Document validation rules

## Resources

### Books
- "Feature Engineering for Machine Learning" by Alice Zheng
- "Python for Data Analysis" by Wes McKinney
- "Hands-on Machine Learning" by Aurélien Géron

### Online Resources
- Scikit-learn Documentation
- Pandas Documentation
- PyData Tutorials
- Kaggle Competitions

### Tools
- Pandas
- Scikit-learn
- NumPy
- Feature-engine
- Category Encoders

## Assessment Questions

1. When should you use different scaling methods?
2. How do you handle categorical variables with high cardinality?
3. What are the best practices for handling missing data?
4. How do you detect and handle outliers effectively?

## Projects

1. Build an automated data cleaning pipeline
2. Create a feature engineering framework
3. Implement a data validation system
4. Develop a custom preprocessing library
