```markdown
# OP Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill introduces the development patterns and conventions used in the OP Python codebase. It covers file naming, import/export styles, commit message conventions, and testing patterns. This guide helps maintain consistency and efficiency when contributing to the OP repository.

## Coding Conventions

### File Naming
- Use **camelCase** for file names.
  - Example: `myModule.py`, `dataProcessor.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import helperFunction
    ```

### Export Style
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    def usefulFunction():
        pass

    __all__ = ['usefulFunction']
    ```

### Commit Messages
- Follow **conventional commit** style.
- Use the `feat` prefix for new features.
  - Example:  
    ```
    feat: add support for user authentication
    ```
- Average commit message length: ~67 characters.

## Workflows

### Feature Development
**Trigger:** When adding a new feature  
**Command:** `/feature-development`

1. Create a new branch for your feature.
2. Implement the feature using camelCase file naming and relative imports.
3. Add or update tests in files matching `*.test.*`.
4. Use a conventional commit message with the `feat` prefix.
5. Push your branch and open a pull request.

### Testing
**Trigger:** When verifying code functionality  
**Command:** `/run-tests`

1. Identify or create test files following the `*.test.*` pattern.
2. Run your tests using your preferred Python test runner (e.g., `pytest`, `unittest`).
3. Ensure all tests pass before merging or deploying changes.

## Testing Patterns

- Test files are named using the pattern `*.test.*` (e.g., `user.test.py`).
- The testing framework is not specified; use your preferred Python testing tool.
- Place test files alongside the modules they test or in a dedicated test directory.
- Example test file:
  ```python
  # user.test.py

  from .user import getUser

  def test_getUser_returns_valid_user():
      user = getUser(1)
      assert user is not None
  ```

## Commands
| Command              | Purpose                                   |
|----------------------|-------------------------------------------|
| /feature-development | Start the feature development workflow    |
| /run-tests           | Run all tests in the codebase             |
```