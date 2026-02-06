# EdgeML Python SDK

Federated Learning orchestration and client SDK for Python.

## Installation

```bash
pip install edgeml-sdk
```

## Quick Start

```python
from edgeml import EdgeML

client = EdgeML(api_key="your-api-key")

# Create rollout
rollout = client.rollouts.create(
    model_id="model-123",
    version="2.0.0",
    rollout_percentage=10
)
```

## Documentation

https://docs.edgeml.io/sdks/python
