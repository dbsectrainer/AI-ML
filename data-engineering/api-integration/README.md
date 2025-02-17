# API Integration for Machine Learning

This section covers techniques and best practices for integrating APIs into machine learning workflows.

## RESTful API Integration

### 1. Basic HTTP Requests
```python
import requests
from typing import Dict, Any

def make_api_request(
    url: str,
    method: str = 'GET',
    params: Dict[str, Any] = None,
    headers: Dict[str, Any] = None,
    data: Dict[str, Any] = None,
    timeout: int = 30
) -> requests.Response:
    """
    Make an HTTP request to an API endpoint
    """
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            headers=headers,
            json=data,
            timeout=timeout
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {str(e)}")
        raise
```

### 2. Rate Limiting
```python
import time
from functools import wraps

class RateLimiter:
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                time.sleep(self.min_interval - time_since_last_call)
            
            self.last_call_time = time.time()
            return func(*args, **kwargs)
        return wrapper

@RateLimiter(calls_per_second=2)
def fetch_data_from_api():
    # API call implementation
    pass
```

### 3. Authentication
```python
class APIAuthenticator:
    def __init__(self, api_key: str, auth_url: str):
        self.api_key = api_key
        self.auth_url = auth_url
        self.token = None
        self.token_expiry = None

    def get_token(self) -> str:
        if self._is_token_valid():
            return self.token
            
        response = requests.post(
            self.auth_url,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        response.raise_for_status()
        
        auth_data = response.json()
        self.token = auth_data['access_token']
        self.token_expiry = time.time() + auth_data['expires_in']
        
        return self.token

    def _is_token_valid(self) -> bool:
        return (
            self.token is not None and
            self.token_expiry is not None and
            time.time() < self.token_expiry
        )
```

## GraphQL Integration

### 1. Basic Query
```python
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

def setup_graphql_client(url: str, headers: Dict[str, str] = None) -> Client:
    transport = RequestsHTTPTransport(
        url=url,
        headers=headers,
        use_json=True,
    )
    return Client(transport=transport, fetch_schema_from_transport=True)

def execute_graphql_query(client: Client, query: str, variables: Dict = None):
    try:
        return client.execute(gql(query), variable_values=variables)
    except Exception as e:
        logging.error(f"GraphQL query failed: {str(e)}")
        raise
```

### 2. Query Builder
```python
class GraphQLQueryBuilder:
    def __init__(self):
        self.query_parts = []
        self.variables = {}

    def add_field(self, field_name: str, subfields: List[str] = None):
        if subfields:
            field_str = f"{field_name} {{ {' '.join(subfields)} }}"
        else:
            field_str = field_name
        self.query_parts.append(field_str)
        return self

    def build(self) -> str:
        return "query { " + " ".join(self.query_parts) + " }"
```

## Webhook Implementation

### 1. Webhook Server
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
        
    data = request.get_json()
    
    # Process webhook data
    try:
        process_webhook_data(data)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_webhook_data(data: Dict[str, Any]):
    # Implementation of webhook data processing
    pass
```

### 2. Webhook Client
```python
class WebhookSender:
    def __init__(self, webhook_url: str, secret: str = None):
        self.webhook_url = webhook_url
        self.secret = secret

    def send_event(self, event_type: str, payload: Dict[str, Any]):
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.secret:
            headers['X-Webhook-Secret'] = self.secret

        data = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'payload': payload
        }

        response = requests.post(
            self.webhook_url,
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response
```

## Error Handling and Retry Logic

### 1. Retry Decorator
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def resilient_api_call(url: str, **kwargs):
    return requests.get(url, **kwargs)
```

### 2. Error Handler
```python
class APIError(Exception):
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

def handle_api_error(error: requests.exceptions.RequestException) -> None:
    if isinstance(error, requests.exceptions.HTTPError):
        status_code = error.response.status_code
        if status_code == 429:
            raise APIError("Rate limit exceeded", status_code)
        elif status_code == 401:
            raise APIError("Authentication failed", status_code)
        elif status_code == 403:
            raise APIError("Permission denied", status_code)
    elif isinstance(error, requests.exceptions.ConnectionError):
        raise APIError("Connection failed")
    elif isinstance(error, requests.exceptions.Timeout):
        raise APIError("Request timed out")
```

## Data Validation

### 1. Schema Validation
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class DataPoint(BaseModel):
    id: str
    value: float
    timestamp: datetime
    tags: Optional[List[str]] = Field(default_factory=list)

def validate_api_response(response_data: Dict) -> List[DataPoint]:
    try:
        return [DataPoint(**item) for item in response_data]
    except ValidationError as e:
        logging.error(f"Data validation failed: {str(e)}")
        raise
```

## Best Practices

### 1. Security
- Use HTTPS for all API calls
- Implement proper authentication
- Validate all input data
- Handle sensitive data securely
- Use API keys and tokens safely

### 2. Performance
- Implement caching
- Use connection pooling
- Handle rate limiting
- Optimize payload size
- Use async calls when appropriate

### 3. Reliability
- Implement retry logic
- Handle errors gracefully
- Log all API interactions
- Monitor API health
- Implement circuit breakers

## Resources

### Books
- "RESTful Web APIs" by Leonard Richardson
- "GraphQL in Action" by Samer Buna
- "Designing Web APIs" by Brenda Jin

### Online Resources
- API Documentation Best Practices
- OpenAPI Specification
- GraphQL Documentation
- Webhook Security Guide

### Tools
- Postman
- Insomnia
- GraphiQL
- Charles Proxy

## Assessment Questions

1. How do you handle API rate limiting effectively?
2. What are the best practices for API authentication?
3. How do you implement robust error handling?
4. What are the advantages of GraphQL over REST?

## Projects

1. Build a data collection pipeline
2. Create a webhook processing system
3. Implement a GraphQL client
4. Develop an API monitoring tool
