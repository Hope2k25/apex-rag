# Credits and Licenses

This project uses the following Free and Open Source Software (FOSS):

## Container Runtime

### Podman
- **Project**: Podman
- **License**: Apache License 2.0
- **Website**: https://podman.io/
- **Purpose**: FOSS alternative to Docker for container management
- **Usage**: Runs PostgreSQL and Neo4j containers

## Database Software

### PostgreSQL + pgvector
- **Image**: pgvector/pgvector:pg16
- **PostgreSQL License**: PostgreSQL License (similar to MIT, permissive)
- **pgvector Extension**: Apache License 2.0
- **Website**: https://github.com/pgvector/pgvector
- **Purpose**: Vector similarity search for PostgreSQL

### Neo4j Community Edition
- **Project**: Neo4j
- **License**: GPLv3 (Community Edition)
- **Website**: https://neo4j.com/
- **Purpose**: Graph database for knowledge storage
- **Note**: Neo4j Community Edition is free and open-source

## Python Dependencies

### Python
- **License**: Python Software Foundation License (PSFL)
- **Website**: https://www.python.org/

### uv Package Manager
- **License**: Apache License 2.0 OR MIT
- **Website**: https://github.com/astral-sh/uv
- **Purpose**: Fast Python package installer

### Key Python Libraries

| Library | License | Website |
|----------|---------|---------|
| asyncpg | PostgreSQL License | https://github.com/MagicStack/asyncpg |
| pgvector | Apache 2.0 | https://github.com/pgvector/pgvector |
| neo4j | Apache 2.0 | https://github.com/neo4j/neo4j-python-driver |
| sentence-transformers | Apache 2.0 | https://github.com/UKPLab/sentence-transformers |
| torch | BSD 3-Clause | https://github.com/pytorch/pytorch |
| numpy | BSD License | https://numpy.org/ |
| pydantic | MIT License | https://github.com/pydantic/pydantic |

## Development Tools

### pytest
- **License**: MIT License
- **Website**: https://github.com/pytest-dev/pytest/

### ruff (Linting & Formatting)
- **License**: Apache License 2.0 / MIT
- **Website**: https://github.com/astral-sh/ruff

### mypy (Static Type Checking)
- **License**: MIT License
- **Website**: https://github.com/python/mypy

## Build Tools

### hatchling
- **License**: MIT License
- **Website**: https://github.com/pypa/hatch

## Security Tools

### trufflehog (Secrets Detection)
- **License**: AGPL-3.0
- **Website**: https://github.com/trufflesecurity/trufflehog

---

## License Compatibility Notes

This project is licensed under the MIT License. The use of the following FOSS components is compatible:

- **Apache 2.0** (Podman, pgvector, neo4j driver, sentence-transformers, ruff) - Permissive, MIT-compatible
- **GPLv3** (Neo4j Community Edition) - Copyleft, but used as a service, not linked into the project
- **BSD 3-Clause** (PyTorch) - Permissive, MIT-compatible
- **AGPL-3.0** (trufflehog) - Strong copyleft, used as development tool only

## Attribution

This project incorporates or uses software from the projects listed above. Each project retains its own copyright and license.
