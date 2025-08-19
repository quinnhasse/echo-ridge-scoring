# Coding Style and Conventions

## Code Style Requirements
- **Follow PEP8** strictly
- **Use type hints** for all function parameters and return values
- **Format with `black`** for consistent code formatting
- **Use `pydantic` for data validation** - already established pattern
- **File Length Limit**: Never create files longer than 500 lines - refactor into modules
- **Imports**: Prefer relative imports within packages

## Documentation Standards
- **Google-style docstrings** for every function:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

## Code Organization Patterns
- **Agent-based architecture** pattern used:
  - `agent.py` - Main agent definition and execution logic
  - `tools.py` - Tool functions used by the agent  
  - `prompts.py` - System prompts
- **Clear module separation** by feature/responsibility
- **Environment variables**: Use `python_dotenv` and `load_env()`

## API and ORM Preferences
- **FastAPI** for APIs (already configured)
- **SQLAlchemy or SQLModel** for ORM if database needed
- **Pydantic** for data validation (extensively used)

## Code Quality Rules
- **Comment non-obvious code** for mid-level developer understanding
- **Add inline `# Reason:` comments** explaining why, not just what
- **No assumptions**: Ask questions if uncertain rather than guessing
- **No hallucination**: Only use known, verified Python packages
- **Verify paths**: Always confirm file paths and module names exist

## Testing Requirements
- **Pytest** for all testing (configured)
- **Test structure**: `/tests` folder mirroring main app structure
- **Minimum test coverage**: 
  - 1 test for expected use
  - 1 edge case test
  - 1 failure case test
- **Update tests** when logic changes

## Project-Specific Patterns
- **Statistical approach**: Z-score normalization with confidence thresholds
- **Explainable results**: Always provide natural language explanations
- **Type safety**: Comprehensive Pydantic validation throughout
- **Modular scoring**: Separate subscore calculation from final aggregation