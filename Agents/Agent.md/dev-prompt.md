```
Enhance the existing ./CLAUDE.md file with following conventions: (1) follow the SOLID principles (2) Strictly implement with TDD (Test-Driven Development) (3) Use conventional commits (4) Do NOT add Claude as a co-author in commit messages (5) ensure all tests pass before committing (6) use context7 before using external crates to avoid hallucination
```

```
update the project memory ./CLAUDE.md, save the current progress, which allows you to resume in the next session.
```


```
Enhance the existing CLAUDE.md file with our team conventions:

## Code Conventions
- Maximum length: 88 characters (Black)

## Git Conventions
- Messages: Conventional Commits (feat:, fix:, docs:, test:, refactor:)
- Branches: feature/, bugfix/, hotfix/

## Tests
- Minimum coverage: 80%
- Naming: test_<module>_<function>_<scenario>

## Documentation
- README: Always up-to-date with installation commands

Retains previously detected information and structures the file clearly.
```

```
<context>
TaskFlow Project - REST API for task management
Stack: Python 3.11, FastAPI, SQLite, Pydantic
</context>

<task>
Creates the Pydantic models for the TaskFlow API
</task>

<constraints>
- Task model with: id, title, description, status, priority, due_date, created_at, updated_at
- Enum for status: TODO, IN_PROGRESS, DONE
- Enum for priority: LOW, MEDIUM, HIGH
- Date validation (due_date >= today for creation)
- Type hints required
</constraints>

<output_format>
File src/taskflow/models/task.py with docstrings
</output_format>
```

```
Step-by-step analysis of what's needed to create CRUD endpoints for the TaskFlow API:

1. First, identify the necessary endpoints (GET, POST, PUT, DELETE)
2. Next, define the structure of each endpoint
3. Then, implement the error-handling code
4. Finally, add the OpenAPI documentation

Target file: src/taskflow/routers/tasks.py
```

```
Use Context7 to search for the official FastAPI documentation on dependencies.
Then apply these best practices to refactor our dependency injection system in TaskFlow.
```
