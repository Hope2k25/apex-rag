# Apex RAG - Next Steps

## 1. Start Neo4j (Required for Graph Features)
The system is currently running in **Postgres-Only Mode**.
To enable the full Knowledge Graph:
1. Open **Neo4j Desktop**.
2. Start your active database (Version 5.x).
3. Ensure it is listening on `bolt://localhost:7687` with password `neo4j_dev_password_2026` (or update `.env`).

## 2. Verify Infrastructure
Run the simple test script until both checks pass:
```powershell
uv run python simple_test.py
```

## 3. Ingest Documentation (Available Now!)
You can already ingest documents into Postgres (Vector Search) even without Neo4j:
```powershell
uv run python -m src.ingestion.pipeline --directory ./docs --pattern "*.md"
```

## 4. Start the MCP Server
Once everything is ready, start the server for your AI coding agent:
```powershell
uv run python -m src.server
```
