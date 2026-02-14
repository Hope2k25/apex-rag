# Pre-Commit Hooks Setup

> **Created**: 2026-01-28
> **Purpose**: Automated code quality, type checking, and security enforcement

---

## Overview

This project uses pre-commit hooks to enforce code quality, type checking, and security before every commit. The hooks are configured in [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) and run automatically before each commit.

## MCP Usage Documentation

All tools were selected based on research using MCP servers:

- **Ruff**: `/astral-sh/ruff` - High reputation, 6955 code snippets
- **Mypy**: `/python/mypy` - High reputation, 733 code snippets, Benchmark Score 91.8
- **TruffleHog**: `/trufflesecurity/trufflehog` - High reputation, 304 code snippets, Benchmark Score 88.3

## Installed Tools

| Tool | Version | Purpose |
|------|---------|---------|
| **Ruff** | v0.14.14 | Fast Python linter and formatter |
| **Mypy** | v1.19.1 | Static type checking |
| **TruffleHog** | v2.2.1 | Secrets detection |
| **Pre-commit** | v4.5.1 | Git hook management |

## Configuration

The pre-commit configuration includes the following hooks:

### Ruff (Linter & Formatter)
- **ruff-check**: Runs linter with auto-fix (`--fix`)
- **ruff-format**: Formats Python code
- **Configuration**: Defined in [`pyproject.toml`](../pyproject.toml) under `[tool.ruff]`

### Mypy (Type Checking)
- Runs static type checking
- **Configuration**: Defined in [`pyproject.toml`](../pyproject.toml) under `[tool.mypy]`
- **Note**: Type checking is disabled for external dependencies (sentence-transformers, torch, tree-sitter, neo4j, asyncpg) via overrides

### TruffleHog (Secrets Detection)
- Scans for secrets, API keys, and credentials
- **Arguments**: `--results=verified,unknown --fail`
- **Behavior**: Blocks commits if secrets are detected

### General File Checks
- **check-added-large-files**: Warns about files > 1MB
- **check-merge-conflict**: Detects merge conflict markers
- **detect-private-key**: Detects private keys
- **end-of-file-fixer**: Ensures proper file endings
- **trailing-whitespace**: Removes trailing whitespace
- **check-yaml**: Validates YAML syntax
- **check-json**: Validates JSON syntax
- **check-toml**: Validates TOML syntax

## Usage

### Automatic Execution

Pre-commit hooks run automatically when you execute `git commit`. No manual intervention is required.

### Manual Execution

To run hooks manually without committing:

```bash
# Run all hooks
pre-commit run --all-files

# Run specific hooks
pre-commit run ruff-check
pre-commit run mypy
pre-commit run trufflehog
```

### Skipping Hooks

To skip hooks for a specific commit:

```bash
# Skip all hooks
git commit --no-verify -m "Your message"

# Skip specific hooks
SKIP=mypy git commit -m "Your message"
```

## Troubleshooting

### Pre-commit not running

If hooks don't run, verify:

```bash
# Check if hooks are installed
ls .git/hooks/

# Reinstall hooks
pre-commit install
```

### TruffleHog false positives

If TruffleHog detects false positives:

1. Add `trufflehog:ignore` comments on lines with known false positives
2. Or use `--results=verified` to only show verified secrets

### Mypy errors

For external dependencies that lack type stubs, mypy is configured to ignore missing imports:

```toml
[[tool.mypy.overrides]]
module = ["sentence_transformers.*", "torch.*", "tree_sitter.*", "neo4j.*", "asyncpg.*"]
ignore_missing_imports = true
```

## Known Issues

### FlashRank Dependency

The `flashrank` package is temporarily commented out due to Windows compatibility issues with `onnxruntime` dependency. This will be addressed in a future update.

### Windows PATH

On Windows, use `python -m pre_commit` instead of `pre-commit` if the command is not recognized.

## Development Workflow

Recommended workflow:

1. **Write code**: Make changes to your files
2. **Stage changes**: `git add <files>`
3. **Auto-fix**: Pre-commit runs ruff-check with `--fix` to auto-fix linting issues
4. **Format**: Pre-commit runs ruff-format to format code
5. **Type check**: Pre-commit runs mypy to verify type hints
6. **Security scan**: Pre-commit runs TruffleHog to detect secrets
7. **Commit**: If all checks pass, commit proceeds

## Updating Hooks

To update hooks after modifying `.pre-commit-config.yaml`:

```bash
pre-commit install
```

To update to latest versions:

```bash
pre-commit autoupdate
```

## References

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [TruffleHog Documentation](https://docs.trufflesecurity.com/)
- [Project pyproject.toml](../pyproject.toml)
- [Pre-commit config](../.pre-commit-config.yaml)
