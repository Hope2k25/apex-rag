import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def check_postgres():
    try:
        conn = await asyncpg.connect(
            user=os.getenv("POSTGRES_USER", "apex"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            database=os.getenv("POSTGRES_DB", "apex_rag"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432)
        )
        print("✅ Postgres Connection Successful")
        await conn.close()
    except Exception as e:
        print(f"❌ Postgres Connection Failed: {e}")

asyncio.run(check_postgres())
