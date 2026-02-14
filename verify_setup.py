import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.storage.postgres_client import PostgresClient
from src.storage.neo4j_client import Neo4jClient

async def check_postgres():
    print("\n🐘 Checking PostgreSQL...")
    try:
        client = PostgresClient()
        await client.connect()
        async with client.acquire() as conn:
            # Check connection
            version = await conn.fetchval("SELECT version();")
            print(f"✅ Connected: {version}")
            
            # Check for pgvector (WARN if missing)
            has_vector = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            if has_vector:
                print("✅ pgvector extension is active")
            else:
                print("⚠️  pgvector extension NOT found (Using degraded mode for Windows)")

            # Check Schema
            has_chunks = await conn.fetchval("SELECT to_regclass('public.semantic_chunks') IS NOT NULL")
            if has_chunks:
                print("✅ Schema initialized: 'semantic_chunks' table exists")
                
                # Check for vector vs float8
                col_type = await conn.fetchval("SELECT data_type FROM information_schema.columns WHERE table_name='semantic_chunks' AND column_name='embedding'")
                print(f"ℹ️  Embedding column type: {col_type}")
            else:
                print("❌ Schema NOT initialized (semantic_chunks missing)")
                
        await client.disconnect()
        return True
    except Exception as e:
        print(f"❌ PostgreSQL Check Failed: {e}")
        return False

async def check_neo4j():
    print("\n🕸️ Checking Neo4j...")
    try:
        client = Neo4jClient()
        await client.connect()
        print("✅ Connected to Neo4j")
        # Check constraints if possible (client.verify_connectivity is implicit in connect)
        await client.disconnect()
        return True
    except Exception as e:
        print(f"❌ Neo4j Check Failed: {e}")
        print("   -> Please ensure Neo4j Desktop is running and the database is started.")
        print("   -> Check port 7687.")
        return False

async def main():
    print("🚀 Apex RAG Verification")
    print("========================")
    
    pg_ok = await check_postgres()
    neo_ok = await check_neo4j()
    
    print("\n========================")
    if pg_ok and neo_ok:
        print("✅ SYSTEM READY")
    elif pg_ok:
        print("⚠️  PARTIAL SYSTEM READY (Postgres OK, Neo4j Missing)")
    else:
        print("❌ SYSTEM NOT READY")

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    asyncio.run(main())
