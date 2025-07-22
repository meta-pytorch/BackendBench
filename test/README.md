# BackendBench Test Suite

This directory contains the pytest test suite for BackendBench.

## Running Tests

First, install the required dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

To run specific test files:
```bash
pytest test/test_smoke.py
pytest test/test_eval.py
pytest test/test_backends.py
pytest test/test_suite.py
```

To run with coverage:
```bash
pytest --cov=BackendBench
```

## Test Structure

- `test_smoke.py` - Smoke tests that verify basic functionality with the Aten backend
- `test_backends.py` - Unit tests for all backend implementations
- `test_eval.py` - Unit tests for evaluation functions
- `test_suite.py` - Unit tests for test suite classes
- `conftest.py` - Shared fixtures and test configuration

## Test Markers

Tests are marked with the following pytest markers:
- `@pytest.mark.smoke` - Basic smoke tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_cuda` - Tests requiring GPU
- `@pytest.mark.requires_api_key` - Tests requiring API keys

Run tests by marker:
```bash
pytest -m smoke
pytest -m "not slow"
pytest -m "not requires_api_key"
```