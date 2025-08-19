# Task Completion Guidelines

## Required Actions When Task is Complete

### 1. Code Quality Checks
```bash
# Format code with black
black src/ tests/ example_usage.py

# Run linting (when configured)
# Note: No specific linter configured yet, consider adding flake8 or ruff

# Type checking (when mypy is added)
# mypy src/
```

### 2. Testing Requirements
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Ensure all new features have tests:
# - 1 test for expected use
# - 1 edge case test  
# - 1 failure case test
```

### 3. Documentation Updates
- **Update README.md** when:
  - New features are added
  - Dependencies change
  - Setup steps are modified
- **Update docstrings** for all new/modified functions using Google style
- **Add inline comments** for complex logic with `# Reason:` explanations

### 4. Task Management
- **Mark completed tasks in TASK.md** immediately after finishing
  - Note: TASK.md doesn't exist yet - should be created when first task is added
- **Add discovered sub-tasks** to TASK.md under "Discovered During Work" section
- **Check PLANNING.md** before starting new tasks
  - Note: PLANNING.md doesn't exist yet - should be created with architecture guidelines

### 5. Code Structure Validation
- **Verify file length** - no files should exceed 500 lines
- **Check module organization** - ensure clear separation by feature/responsibility
- **Validate imports** - prefer relative imports within packages
- **Ensure type hints** are present for all functions

### 6. Environment and Dependencies
- **Use venv_linux** virtual environment for all Python commands
  - Note: venv_linux not found - may need to be created
- **Update poetry.lock** if dependencies changed
- **Test example usage** still works: `python example_usage.py`

### 7. Project-Specific Checks
- **Validate Pydantic schemas** still work correctly
- **Test scoring calculations** produce expected results
- **Verify confidence metrics** are within expected ranges
- **Check natural language explanations** are generated properly

### 8. Git Best Practices
- **Commit with clear messages** describing what was accomplished
- **Don't commit** unless explicitly requested by user
- **Ensure no secrets** or keys are committed

## Missing Components to Address
1. **Create TASK.md** for task tracking
2. **Create PLANNING.md** for architecture documentation  
3. **Setup venv_linux** virtual environment as specified in CLAUDE.md
4. **Add linting configuration** (ruff, flake8, or similar)
5. **Add mypy configuration** for type checking
6. **Create tests directory** with initial test files

## Quality Gates
Before marking any task complete:
- [ ] Code is formatted with black
- [ ] All functions have type hints
- [ ] All functions have Google-style docstrings
- [ ] Tests exist and pass
- [ ] Documentation is updated
- [ ] Example usage still works
- [ ] No files exceed 500 lines
- [ ] Task is marked in TASK.md