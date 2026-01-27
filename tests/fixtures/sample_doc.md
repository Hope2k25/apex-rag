# Test Document: FastAPI Guide

This is a sample document for testing the Apex RAG ingestion pipeline.

## Introduction

FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Key Features

### High Performance

- One of the fastest Python frameworks available
- On par with NodeJS and Go
- Based on Starlette and Pydantic

### Easy to Use

FastAPI provides automatic documentation, validation, and more.

## Installation

```bash
pip install fastapi
pip install uvicorn
```

## Quick Start

Here's a minimal FastAPI application:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}
```

## Error Handling

Common errors you might encounter:

### ValueError: Prefix must start with /

This error occurs when you create an APIRouter with an invalid prefix.

**Solution:** Ensure your prefix starts with a forward slash.

```python
# Wrong
router = APIRouter(prefix="api")

# Correct
router = APIRouter(prefix="/api")
```

### ValidationError

Pydantic validation errors occur when request data doesn't match the expected schema.

## Conclusion

FastAPI is an excellent choice for building modern Python APIs.
