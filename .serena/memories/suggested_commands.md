# Suggested Commands for Echo Ridge Scoring Development

## Environment Setup
```bash
# Install dependencies (Poetry recommended)
poetry install
poetry shell

# Alternative with pip
pip install -e .
```

## Development Commands
```bash
# Run the main demo/example
python example_usage.py

# Run tests (when tests are implemented)
pytest

# Format code with black
black src/ example_usage.py

# Check code style
black --check src/ example_usage.py

# Type checking (if mypy added)
mypy src/

# Install in development mode
pip install -e .
```

## Virtual Environment
- **Important**: Use `venv_linux` virtual environment for Python commands as specified in CLAUDE.md
- Note: No venv_linux found in current directory - may need to be created

## Poetry Commands
```bash
# Add new dependency
poetry add package_name

# Add development dependency  
poetry add --group dev package_name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Build package
poetry build
```

## Git Commands (Darwin-specific)
```bash
# Standard git operations work on macOS
git status
git add .
git commit -m "message"
git push origin main
```

## File System Commands (Darwin/macOS)
```bash
# List files
ls -la

# Find files
find . -name "*.py"

# Search in files (use grep or ripgrep)
grep -r "pattern" src/

# Navigate directories
cd src/
pwd
```

## Testing and Quality Commands
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_specific.py

# Run tests in verbose mode
pytest -v
```

## Project-Specific Commands
```bash
# Demonstrate scoring system
python example_usage.py

# Validate schema (when implemented)
python -c "from src.schema import CompanySchema; print('Schema valid')"

# Test normalization (when tests exist)
pytest tests/test_normalization.py

# Test scoring (when tests exist)  
pytest tests/test_scoring.py
```

## Notes
- **Missing**: Currently no venv_linux directory found - may need setup
- **Missing**: PLANNING.md and TASK.md files referenced in CLAUDE.md don't exist yet
- **Poetry**: Available at /opt/homebrew/bin/poetry
- **Python**: Available at /Users/quinnhasse/.cache/uv/archive-v0/D2WTLFhJOxENSN01-Isme/bin/python3