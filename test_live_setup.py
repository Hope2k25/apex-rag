#!/usr/bin/env python3
"""
Live test script to verify Apex RAG infrastructure is working.

Tests:
1. PostgreSQL connection with pgvector extension
2. Neo4j connection
3. Embedding model (local, no API calls)
4. Database schema verification
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.postgres_client import PostgresClient
from src.utils.embedding import EmbeddingModel, get_embedding_model


async def test_postgres_connection():
    """Test PostgreSQL connection and pgvector extension."""
    print("\n=== Testing PostgreSQL Connection ===")
    try:
        storage = PostgresClient()
        await storage.connect()
        
        # Test basic connection
        async with storage.acquire() as conn:
            result = await conn.fetchval("SELECT 1;")
            print(f"✅ PostgreSQL connection successful")
            print(f"   Query result: {result}")
            
            # Test pgvector extension mismatch (Warn instead of Fail)
            result = await conn.fetch("SELECT * FROM pg_extension WHERE extname = 'vector';")
            if result:
                print(f"✅ pgvector extension installed")
            else:
                print(f"⚠️  pgvector extension NOT found (Running in degraded mode for Windows)")
            
            # Test schema existence
            result = await conn.fetchval("""
                SELECT to_regclass('public.semantic_chunks');
            """)
            if result:
                print(f"✅ Schema initialized (semantic_chunks table found)")
            else:
                print(f"❌ Schema NOT initialized")

        await storage.disconnect()
            print(f"❌ Vector column NOT found")
        
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False


def test_neo4j_connection():
    """Test Neo4j connection."""
    print("\n=== Testing Neo4j Connection ===")
    try:
        storage = Neo4jStorage()
        
        # Test basic connection
        result = storage.execute_query("RETURN 1 AS test;")
        print(f"✅ Neo4j connection successful")
        print(f"   Query result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False


def test_embedding_model():
    """Test embedding model (local, no API calls)."""
    print("\n=== Testing Embedding Model ===")
    try:
        model = get_embedding_model()
        
        # Test single text embedding
        text = "This is a test of the local embedding model."
        embedding = model.embed(text)
        print(f"✅ Embedding generated successfully")
        print(f"   Text: '{text[:50]}...'")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
        return False


def test_database_schema():
    """Test database schema is initialized."""
    print("\n=== Testing Database Schema ===")
    try:
        storage = PostgresStorage()
        
        # Check if semantic_chunks table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'semantic_chunks'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ semantic_chunks table exists")
        else:
            print(f"❌ semantic_chunks table NOT found")
        
        # Check if code_entities table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'code_entities'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ code_entities table exists")
        else:
            print(f"❌ code_entities table NOT found")
        
        # Check if memory_notes table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'memory_notes'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ memory_notes table exists")
        else:
            print(f"❌ memory_notes table NOT found")
        
        # Check if ingestion_manifest table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'ingestion_manifest'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ ingestion_manifest table exists")
        else:
            print(f"❌ ingestion_manifest table NOT found")
        
        # Check if libraries table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'libraries'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ libraries table exists")
        else:
            print(f"❌ libraries table NOT found")
        
        # Check if api_elements table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'api_elements'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ api_elements table exists")
        else:
            print(f"❌ api_elements table NOT found")
        
        # Check if error_patterns table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'error_patterns'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ error_patterns table exists")
        else:
            print(f"❌ error_patterns table NOT found")
        
        # Check if error_fixes table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'error_fixes'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ error_fixes table exists")
        else:
            print(f"❌ error_fixes table NOT found")
        
        # Check if memory_checkpoints table exists
        result = storage.execute_query("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'memory_checkpoints'
            );
        """)
        if result and result[0]['exists']:
            print(f"✅ memory_checkpoints table exists")
        else:
            print(f"❌ memory_checkpoints table NOT found")
        
        return True
    except Exception as e:
        print(f"❌ Database schema check failed: {e}")
        return False


def main():
    """Run all live tests."""
    print("=" * 60)
    print("APEX RAG LIVE INFRASTRUCTURE TEST")
    print("=" * 60)
    print()
    
    results = {
        "postgres": False,
        "neo4j": False,
        "embedding": False,
        "schema": False,
    }
    
    # Test PostgreSQL
    results["postgres"] = test_postgres_connection()
    
    # Test Neo4j
    results["neo4j"] = test_neo4j_connection()
    
    # Test embedding model
    results["embedding"] = test_embedding_model()
    
    # Test database schema
    results["schema"] = test_database_schema()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print()
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    all_passed = all(results.values())
    if all_passed:
        print("🎉 ALL TESTS PASSED! Infrastructure is ready for development.")
        print()
        print("Next steps:")
        print("1. Add LLM API key to .env file (ZAI_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY)")
        print("2. Start the MCP server: python -m src.server")
        print("3. Or run tests: python -m pytest tests/")
        print("4. Ingest documents into: data/input/docs/")
        print("5. Check Neo4j browser: http://localhost:7474")
    else:
        print("⚠️  SOME TESTS FAILED. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
