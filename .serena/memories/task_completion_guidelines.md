# Task Completion Guidelines

## Task Management Process
1. **Always read `PLANNING.md`** at start (note: currently missing, documented in CLAUDE.md)
2. **Check `TASK.md`** before starting new tasks
3. **Add new tasks** to TASK.md with brief description and today's date
4. **Mark completed tasks** in TASK.md immediately after finishing
5. **Add discovered sub-tasks** to "Discovered During Work" section in TASK.md

## Before Task Completion Checklist
- [ ] **Run tests**: Use `pytest` to ensure all tests pass
- [ ] **Check existing tests**: Update any tests affected by logic changes
- [ ] **Create new tests**: Add unit tests for new features (expected use + edge case + failure case)
- [ ] **Lint/Format**: Run formatting tools (black) if available
- [ ] **Type checking**: Verify type hints are correct
- [ ] **Documentation**: Update README.md if features/dependencies changed
- [ ] **Virtual environment**: Ensure venv_linux was used for all Python commands

## Code Quality Standards
- **File size limit**: Never create files longer than 500 lines
- **Refactor when needed**: Split large files into modules
- **Test coverage**: Each new feature needs comprehensive tests
- **Documentation**: Update docstrings and inline comments
- **Type safety**: All functions must have type hints

## Integration Requirements
- **Database writes**: Verify persistence works correctly
- **CLI functionality**: Test all relevant CLI commands
- **API endpoints**: Test FastAPI endpoints if modified
- **Batch processing**: Validate deterministic outputs
- **Error handling**: Ensure graceful failure modes

## Documentation Updates
- **README.md**: Update when features/setup changes
- **CLAUDE.md**: May need updates for new conventions
- **Docstrings**: Google style for all functions
- **Comments**: Explain complex logic with `# Reason:` comments

## Verification Steps
1. **Local testing**: All tests must pass
2. **Integration testing**: CLI and API functionality verified  
3. **Deterministic validation**: Same inputs produce identical outputs
4. **Performance check**: No significant performance regressions
5. **Database integrity**: Persistence layer works correctly

## Current Active Task
- **2025-08-19**: Implement Echo Ridge Phases 4-10 (In Progress)
- **PRP File**: PRPs/echo-ridge-phases-4-10.md
- Transform Phase 3 scoring system into production-ready service