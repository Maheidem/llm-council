# LLM Council - Project Instructions

## Testing Policy: NO MOCKED API TESTS

**MANDATORY**: All API tests MUST reach actual API endpoints. Mocking API responses is strictly prohibited.

### Rules

1. **No Mock Providers** - Do not use `MockProvider`, `MagicMock`, or `@patch` for LLM API calls
2. **Real API Validation** - Tests must validate actual API responses, not simulated ones
3. **Local Server Required** - Use LM Studio (http://localhost:1234/v1) as the test endpoint to avoid costs
4. **CI Requirement** - LM Studio must be running in CI/CD - tests fail if unavailable

### Rationale

- Mock tests hide integration bugs that only surface in production
- Real API tests catch serialization issues, timeout handling, and response parsing errors
- Local LM Studio provides cost-free testing with real LLM behavior

### Exceptions

- `test_testing.py` - Tests the mock framework itself, mocks allowed
- Pure logic tests (models, schemas, assertions) - No API involvement, no mocks needed
- Environment variable patching (`patch.dict(os.environ)`) - Acceptable for config testing

### Test Infrastructure

```python
# Use the lmstudio_provider fixture from conftest.py
def test_something(lmstudio_provider):
    result = lmstudio_provider.complete("system", "user")
    assert len(result) > 0  # Real response validation

# Tests will fail fast if LM Studio is not running
```

### Running Tests

```bash
# Ensure LM Studio is running on localhost:1234
uv run pytest tests/ -v

# Run only API tests
uv run pytest tests/ -v -m api
```

## Development Guidelines

- Entry points must be synchronous functions (wrap async with `asyncio.run()`)
- Use UV for dependency management, not pip/venv directly
- Test actual entry point commands after packaging, not just module imports
