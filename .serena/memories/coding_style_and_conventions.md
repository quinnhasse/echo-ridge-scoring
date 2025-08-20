# Coding Style and Conventions

## Language and Formatting
- **Primary Language**: Python 3.11+
- **Code Style**: PEP8 compliance
- **Formatter**: Black (mentioned in CLAUDE.md)
- **Type Hints**: Required for all functions and methods
- **Data Validation**: Pydantic models for all data structures

## File Size and Organization
- **Maximum file size**: 500 lines of code per file
- **Module organization**: Split by feature/responsibility when approaching limit
- **Import style**: Prefer relative imports within packages
- **Environment variables**: Use python_dotenv and load_env()

## Documentation Standards
- **Docstrings**: Required for every function using Google style:
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
- **Comments**: Non-obvious code requires explanation
- **Reasoning comments**: Use `# Reason:` to explain why, not what
- **Inline documentation**: Ensure mid-level developer comprehension

## Testing Requirements
- **Framework**: Pytest
- **Test location**: `/tests` folder mirroring main app structure
- **Minimum coverage**: Each feature requires:
  - 1 test for expected use case
  - 1 edge case test
  - 1 failure case test
- **Test execution**: Use venv_linux virtual environment

## Naming and Structure
- **Agent organization pattern**:
  - `agent.py` - Main agent definition and execution logic
  - `tools.py` - Tool functions used by the agent  
  - `prompts.py` - System prompts
- **Clear module separation**: Group by feature or responsibility
- **Consistent patterns**: Follow existing architectural patterns

## Error Handling and Reliability
- **Never assume missing context**: Ask questions if uncertain
- **No hallucination**: Only use verified Python packages
- **Path confirmation**: Always confirm file paths and module names exist
- **Preserve existing code**: Never delete/overwrite unless explicitly instructed