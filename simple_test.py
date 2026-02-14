"""
Simple live test to verify database connections.
Tests PostgreSQL and Neo4j connectivity.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

from src.storage.postgres_client import PostgresClient, PostgresConfig
from src.storage.neo4j_client import Neo4jClient, Neo4jConfig

# Explicitly load .env file
load_dotenv()



async def test_postgres_connection():
    """Test PostgreSQL connection and basic operations."""
    print("=" * 60)
    print("Testing PostgreSQL Connection")
    print("=" * 60)

    config = PostgresConfig.from_env()
    print(f"Host: {config.host}:{config.port}")
    print(f"Database: {config.database}")
    print(f"User: {config.user}")

    client = PostgresClient(config)

    try:
        await client.connect()
        print("[OK] PostgreSQL connection successful!")

        # Test a simple query
        async with client.acquire() as conn:
            result = await conn.fetchval("SELECT version()")
            print(f"[OK] PostgreSQL version: {result[:50]}...")

            # Check if tables exist
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)
            print(f"[OK] Found {len(tables)} tables: {[t['tablename'] for t in tables]}")

        await client.disconnect()
        print("[OK] PostgreSQL disconnected cleanly")
        return True

    except Exception as e:
        print(f"[FAIL] PostgreSQL connection failed: {e}")
        return False


async def test_neo4j_connection():
    """Test Neo4j connection and basic operations."""
    print("\n" + "=" * 60)
    print("Testing Neo4j Connection")
    print("=" * 60)

    config = Neo4jConfig.from_env()
    print(f"URI: {config.uri}")
    print(f"Database: {config.database}")
    print(f"User: {config.user}")

    client = Neo4jClient(config)

    try:
        await client.connect()
        print("[OK] Neo4j connection successful!")

        # Test a simple query
        async with client.session() as session:
            result = await session.run("RETURN 'Hello Neo4j!' as message")
            record = await result.single()
            print(f"[OK] Query result: {record['message']}")

            # Get database version
            result = await session.run("CALL dbms.components() YIELD name, versions, edition")
            records = await result.data()
            print(f"[OK] Neo4j components: {records}")

            # Get node counts
            result = await session.run("MATCH (n) RETURN labels(n) as label, count(*) as count")
            records = await result.data()
            print(f"[OK] Current nodes: {records}")

        await client.disconnect()
        print("[OK] Neo4j disconnected cleanly")
        return True

    except Exception as e:
        print(f"[FAIL] Neo4j connection failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("APEX RAG - Database Connection Test")
    print("=" * 60)

    # Test PostgreSQL
    postgres_ok = await test_postgres_connection()

    # Test Neo4j
    neo4j_ok = await test_neo4j_connection()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"PostgreSQL: {'[PASS]' if postgres_ok else '[FAIL]'}")
    print(f"Neo4j:      {'[PASS]' if neo4j_ok else '[FAIL]'}")

    if postgres_ok and neo4j_ok:
        print("\n[OK] All tests passed! Databases are ready.")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
