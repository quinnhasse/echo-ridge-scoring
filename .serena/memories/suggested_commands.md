# Suggested Commands for Echo Ridge Scoring

## Testing
- Run all tests: `poetry run pytest tests/ -v`
- Run specific test file: `poetry run pytest tests/test_batch.py -v`
- Run with coverage: `poetry run pytest tests/ --cov=src --cov-report=html`

## Code Quality
- Format code: `poetry run black src/ tests/`
- Type checking: `poetry run mypy src/`
- Linting: `poetry run ruff check src/ tests/`

## CLI Operations
- Score batch: `poetry run python cli.py score-batch input.jsonl results.jsonl`
- Validate deterministic: `poetry run python cli.py validate-deterministic context_id`
- List contexts: `poetry run python cli.py list-contexts`

## Development
- Install dependencies: `poetry install`
- Add dependency: `poetry add package_name`
- Shell with venv: `poetry shell`
- Run Python: `poetry run python`