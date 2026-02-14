"""
End-to-end tests for Apex RAG system.

These tests use real database connections to verify the complete system
works together as expected.

Run with: uv run pytest tests/e2e/ -v

Prerequisites:
1. Docker containers running: docker-compose up -d
2. Environment variables configured in .env
3. PostgreSQL accessible at localhost:5432
4. Neo4j accessible at localhost:7687
"""
