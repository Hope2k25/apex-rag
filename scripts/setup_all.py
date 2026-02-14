"""
Setup script for Apex RAG infrastructure.
Initializes PostgreSQL schema and Neo4j constraints.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.postgres_client import PostgresClient
from src.storage.neo4j_client import Neo4jClient

async def setup_postgres():
    print("🐘 Setting up PostgreSQL...")
    try:
        client = PostgresClient()
        await client.connect()
        
        # Read init.sql
        sql_path = Path(__file__).parent.parent / "sql" / "init.sql"
        if not sql_path.exists():
            print(f"❌ Could not find sql/init.sql at {sql_path}")
            return
            
        print(f"   Reading schema from {sql_path.name}...")
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_script = f.read()
            
        # Execute script
        print("   Executing schema script...")
        async with client.acquire() as conn:
             await conn.execute(sql_script)
             
        print("✅ PostgreSQL Schema Initialized")
        await client.disconnect()
    except Exception as e:
        print(f"❌ PostgreSQL Setup Failed: {e}")

async def setup_neo4j():
    print("\n🕸️ Setting up Neo4j...")
    try:
        client = Neo4jClient()
        await client.connect()
        print("   Connected to Neo4j, applying schema...")
        await client.setup_schema()
        print("✅ Neo4j Schema/Constraints Initialized")
        await client.disconnect()
    except Exception as e:
        print(f"❌ Neo4j Setup Failed: {e}")

async def main():
    print("🚀 Starting Apex RAG Infrastructure Setup")
    print("=========================================")
    await setup_postgres()
    await setup_neo4j()
    print("\n✨ Setup Complete")

if __name__ == "__main__":
    # Load env vars if needed (dotenv)
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        print("Warning: python-dotenv not installed, assuming environment variables are set.")
    
    # Check for asyncpg/neo4j/pgvector dependencies
    # (Implicitly checked by imports above, but good to catch early)
    
    asyncio.run(main())
