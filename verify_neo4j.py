import asyncio
import os
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv()

async def check_neo4j():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j_dev_password_2026")
    
    print(f"Connecting to {uri} as {user}...")
    
    try:
        # Try tuple auth first (what the code uses)
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        await driver.verify_connectivity()
        print("✅ Neo4j Connection Successful (Tuple Auth)")
        await driver.close()
    except Exception as e:
        print(f"❌ Neo4j Tuple Auth Failed: {e}")
        
    try:
        # Try explicit basic_auth
        from neo4j import basic_auth
        driver = AsyncGraphDatabase.driver(uri, auth=basic_auth(user, password))
        await driver.verify_connectivity()
        print("✅ Neo4j Connection Successful (basic_auth)")
        await driver.close()
    except Exception as e:
        print(f"❌ Neo4j basic_auth Failed: {e}")

asyncio.run(check_neo4j())
