### Testing & Quality

- **SOLID principles** applied throughout refactored modules

## Code Conventions

- **Function length**: Target ≤20 lines for new code
- **No magic numbers**: Use named constants
- **SOLID principles**: Single responsibility, open/closed, dependency inversion
- **TDD (Test-Driven Development)**: MANDATORY for all new features and bug fixes
  - Write tests BEFORE writing implementation code
  - Red-Green-Refactor cycle: Write failing test → Make it pass → Refactor
  - No code commits without corresponding tests
  - Tests must be written first to define expected behavior

## Git Conventions

- **Messages**: Conventional Commits (feat:, fix:, docs:, test:, refactor:)
- **Branches**: feature/, bugfix/, hotfix/
- **Co-authorship**: Do NOT add Claude as a co-author in commit messages

## Tests

- **Minimum coverage**: 80% (currently 92%)
- **Naming**: test_<module>_<function>_<scenario>

### TDD Workflow (MANDATORY)

**All new features and bug fixes MUST follow Test-Driven Development:**

1. **Red Phase**: Write a failing test first
   - Define expected behavior in test
   - Run test to confirm it fails
   - Example: `...` (should fail)

2. **Green Phase**: Write minimal code to pass the test
   - Implement only what's needed to make test pass
   - Run test to confirm it passes
   - Example: `...` (should pass)

3. **Refactor Phase**: Improve code quality
   - Apply SOLID principles
   - Remove duplication
   - Keep functions ≤20 lines
   - Ensure all tests still pass

4. **Commit**: Only commit when all tests pass
   - Run `make test` before committing
   - Run `make check-all` to verify style/types
   - Follow Conventional Commits format

**Example TDD Session:**
```bash
# 1. Write test first
vim ...
make test    # Should FAIL (Red)

# 2. Write implementation
vim ...
make test    # Should PASS (Green)

# 3. Refactor if needed
vim ...
make test    # Should still PASS

# 4. Run full test suite
make test

# 5. Commit
git add ...
git commit -m "feat: add new feature with TDD"
```

## Documentation

- **README**: Always up-to-date with installation commands

## Commands

### Development Workflow

```bash
# Setup
make install-dev        # Install with dev dependencies

# Testing
make test              # Run tests
make test-verbose      # Run tests with verbose output
make test-cov          # Run tests with coverage (requires 80%)
make test-cov-html     # Generate HTML coverage report

# Code Quality
make lint              # Check code style with ruff
make format            # Check formatting with black
make format-fix        # Auto-format code with black
make type-check        # Run type checking with mypy
make check-all         # Run all checks (lint, format, type, test)

# Development
make run               # Run the application with auto-reload
make clean             # Remove cache and coverage files
make help              # Show all available commands
```

## Notes for Development

- **TDD is MANDATORY**: Always write tests BEFORE implementation code (see TDD Workflow above)
- Run `make test` to ensure all tests pass before committing
- Run `make check-all` before committing to verify style, types, and tests
- Follow Conventional Commits format
- Maintain 80%+ test coverage on all new code
- Keep functions ≤20 lines for maintainability
- Use named constants instead of magic numbers
- Apply SOLID principles to new code
- No commits without corresponding tests
