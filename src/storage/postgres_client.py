"""
PostgreSQL client for Apex RAG.

Handles all PostgreSQL operations:
- Semantic chunks (with pgvector embeddings)
- Code entities
- Memory notes with checkpoint/rollback
- Hybrid search (vector + BM25)

Self-hosted - runs locally.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from uuid import UUID

import asyncpg
from asyncpg import Pool

from .schemas import (
    SemanticChunk,
    SemanticChunkCreate,
    CodeEntity,
    CodeEntityCreate,
    MemoryNote,
    MemoryNoteCreate,
    MemoryCheckpoint,
    MemorySnapshot,
    SearchResult,
    IngestionManifestEntry,
    IngestionStatus,
)


class PostgresConfig:
    """PostgreSQL connection configuration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "apex_rag",
        user: str = "apex",
        password: str = "",
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "apex_rag"),
            user=os.getenv("POSTGRES_USER", "apex"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            min_connections=int(os.getenv("POSTGRES_MIN_CONNECTIONS", "2")),
            max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "10")),
        )

    @property
    def dsn(self) -> str:
        """Get connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class PostgresClient:
    """
    Async PostgreSQL client for Apex RAG.

    Manages connection pool and provides CRUD operations
    for all stored entities.
    """

    def __init__(self, config: Optional[PostgresConfig] = None):
        """Initialize the client with configuration."""
        self.config = config or PostgresConfig.from_env()
        self._pool: Optional[Pool] = None

    async def connect(self):
        """Establish connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
            )

    async def disconnect(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self._pool:
            await self.connect()
        async with self._pool.acquire() as conn:
            yield conn

    # ========================================
    # SEMANTIC CHUNKS
    # ========================================

    async def create_chunk(self, chunk: SemanticChunkCreate) -> SemanticChunk:
        """Create a new semantic chunk."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO semantic_chunks 
                    (source_file, chunk_index, header_path, content, embedding, metadata, content_hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (source_file, chunk_index) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    content_hash = EXCLUDED.content_hash,
                    updated_at = NOW()
                RETURNING *
                """,
                chunk.source_file,
                chunk.chunk_index,
                chunk.header_path,
                chunk.content,
                chunk.embedding,
                chunk.metadata,
                chunk.content_hash,
            )
            return SemanticChunk(**dict(row))

    async def get_chunk(self, chunk_id: UUID) -> Optional[SemanticChunk]:
        """Get a chunk by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM semantic_chunks WHERE id = $1",
                chunk_id,
            )
            return SemanticChunk(**dict(row)) if row else None

    async def get_chunks_by_file(self, source_file: str) -> list[SemanticChunk]:
        """Get all chunks for a source file."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM semantic_chunks 
                WHERE source_file = $1 
                ORDER BY chunk_index
                """,
                source_file,
            )
            return [SemanticChunk(**dict(row)) for row in rows]

    async def delete_chunks_by_file(self, source_file: str) -> int:
        """Delete all chunks for a source file. Returns count deleted."""
        async with self.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM semantic_chunks WHERE source_file = $1",
                source_file,
            )
            return int(result.split()[-1])

    # ========================================
    # HYBRID SEARCH
    # ========================================

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        alpha: float = 0.7,
        limit: int = 20,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining vector and BM25.

        Args:
            query_text: Text query for BM25
            query_embedding: Vector for similarity search
            alpha: Weight for dense vs sparse (0.7 = 70% vector)
            limit: Maximum results to return

        Returns:
            List of SearchResult ordered by combined score
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM hybrid_search($1, $2, $3, $4)
                """,
                query_text,
                query_embedding,
                alpha,
                limit,
            )

            return [
                SearchResult(
                    id=row["id"],
                    content=row["content"],
                    source_file=row["source_file"],
                    header_path=row["header_path"],
                    dense_score=row["dense_score"],
                    sparse_score=row["sparse_score"],
                    combined_score=row["combined_score"],
                )
                for row in rows
            ]

    async def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 20,
        min_similarity: float = 0.5,
    ) -> list[SearchResult]:
        """
        Pure vector similarity search.

        Args:
            query_embedding: Query vector
            limit: Maximum results
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of SearchResult ordered by similarity
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    id, content, source_file, header_path, metadata,
                    1 - (embedding <=> $1) as similarity
                FROM semantic_chunks
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> $1) >= $2
                ORDER BY embedding <=> $1
                LIMIT $3
                """,
                query_embedding,
                min_similarity,
                limit,
            )

            return [
                SearchResult(
                    id=row["id"],
                    content=row["content"],
                    source_file=row["source_file"],
                    header_path=row["header_path"],
                    dense_score=row["similarity"],
                    sparse_score=0.0,
                    combined_score=row["similarity"],
                    metadata=row["metadata"],
                )
                for row in rows
            ]

    # ========================================
    # MEMORY NOTES
    # ========================================

    async def create_memory(self, memory: MemoryNoteCreate) -> MemoryNote:
        """Create a new memory note."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO memory_notes 
                    (agent_id, content, memory_type, context, keywords, source_ref, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
                """,
                memory.agent_id,
                memory.content,
                memory.memory_type.value,
                memory.context,
                memory.keywords,
                memory.source_ref,
                memory.embedding,
            )
            return MemoryNote(**dict(row))

    async def get_memory(self, memory_id: UUID) -> Optional[MemoryNote]:
        """Get a memory note by ID."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memory_notes WHERE id = $1",
                memory_id,
            )
            return MemoryNote(**dict(row)) if row else None

    async def update_memory(
        self,
        memory_id: UUID,
        content: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        embedding: Optional[list[float]] = None,
    ) -> Optional[MemoryNote]:
        """Update a memory note."""
        async with self.acquire() as conn:
            # Build dynamic update
            updates = []
            params = [memory_id]
            param_idx = 2

            if content is not None:
                updates.append(f"content = ${param_idx}")
                params.append(content)
                param_idx += 1

            if keywords is not None:
                updates.append(f"keywords = ${param_idx}")
                params.append(keywords)
                param_idx += 1

            if embedding is not None:
                updates.append(f"embedding = ${param_idx}")
                params.append(embedding)
                param_idx += 1

            if not updates:
                return await self.get_memory(memory_id)

            updates.append("updated_at = NOW()")

            row = await conn.fetchrow(
                f"""
                UPDATE memory_notes 
                SET {', '.join(updates)}
                WHERE id = $1
                RETURNING *
                """,
                *params,
            )
            return MemoryNote(**dict(row)) if row else None

    async def soft_delete_memory(
        self,
        memory_id: UUID,
        reason: str = "User requested deletion",
    ) -> bool:
        """Soft delete a memory note."""
        async with self.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE memory_notes 
                SET is_active = FALSE, deleted_reason = $2, updated_at = NOW()
                WHERE id = $1
                """,
                memory_id,
                reason,
            )
            return result != "UPDATE 0"

    async def search_memories(
        self,
        query_embedding: list[float],
        agent_id: str = "default",
        limit: int = 10,
        include_inactive: bool = False,
    ) -> list[MemoryNote]:
        """Search memories by similarity."""
        async with self.acquire() as conn:
            active_filter = "" if include_inactive else "AND is_active = TRUE"
            rows = await conn.fetch(
                f"""
                SELECT *, 1 - (embedding <=> $1) as similarity
                FROM memory_notes
                WHERE agent_id = $2 
                  AND embedding IS NOT NULL
                  {active_filter}
                ORDER BY embedding <=> $1
                LIMIT $3
                """,
                query_embedding,
                agent_id,
                limit,
            )
            return [MemoryNote(**dict(row)) for row in rows]

    # ========================================
    # MEMORY CHECKPOINTS
    # ========================================

    async def create_checkpoint(
        self,
        name: str,
        agent_id: str = "default",
        reason: Optional[str] = None,
    ) -> MemoryCheckpoint:
        """Create a memory checkpoint (snapshot current state)."""
        async with self.acquire() as conn:
            # Get current memories
            memories = await conn.fetch(
                """
                SELECT * FROM memory_notes 
                WHERE agent_id = $1 AND is_active = TRUE
                """,
                agent_id,
            )

            snapshot = MemorySnapshot(
                memories=[dict(row) for row in memories],
                memory_links=[],  # TODO: Include links if we have them
                timestamp=datetime.utcnow(),
                memory_count=len(memories),
            )

            # Create checkpoint
            row = await conn.fetchrow(
                """
                INSERT INTO memory_checkpoints 
                    (checkpoint_name, created_by, reason, memory_snapshot)
                VALUES ($1, $2, $3, $4)
                RETURNING *
                """,
                name,
                agent_id,
                reason,
                snapshot.model_dump_json(),
            )

            return MemoryCheckpoint(
                checkpoint_id=row["id"],
                checkpoint_name=row["checkpoint_name"],
                created_at=row["created_at"],
                created_by=row["created_by"],
                reason=row["reason"],
                memory_snapshot=snapshot,
            )

    async def rollback_to_checkpoint(
        self,
        checkpoint_id: UUID,
        agent_id: str = "default",
    ) -> bool:
        """Rollback memories to a checkpoint state."""
        async with self.acquire() as conn:
            # Get checkpoint
            row = await conn.fetchrow(
                "SELECT * FROM memory_checkpoints WHERE id = $1",
                checkpoint_id,
            )

            if not row:
                return False

            import json
            snapshot_data = json.loads(row["memory_snapshot"])
            snapshot = MemorySnapshot(**snapshot_data)

            async with conn.transaction():
                # Soft delete current memories
                await conn.execute(
                    """
                    UPDATE memory_notes 
                    SET is_active = FALSE, 
                        deleted_reason = 'Rolled back to checkpoint: ' || $2
                    WHERE agent_id = $1 AND is_active = TRUE
                    """,
                    agent_id,
                    str(checkpoint_id),
                )

                # Restore from snapshot
                for mem_data in snapshot.memories:
                    await conn.execute(
                        """
                        INSERT INTO memory_notes 
                            (agent_id, content, memory_type, context, keywords, 
                             source_ref, embedding, is_active, restored_from)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE, $8)
                        """,
                        agent_id,
                        mem_data["content"],
                        mem_data["memory_type"],
                        mem_data.get("context"),
                        mem_data.get("keywords", []),
                        mem_data.get("source_ref"),
                        mem_data.get("embedding"),
                        checkpoint_id,
                    )

            return True

    async def list_checkpoints(
        self,
        agent_id: str = "default",
        limit: int = 20,
    ) -> list[MemoryCheckpoint]:
        """List available checkpoints."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM memory_checkpoints 
                WHERE created_by = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                agent_id,
                limit,
            )

            checkpoints = []
            for row in rows:
                import json
                snapshot_data = json.loads(row["memory_snapshot"])
                checkpoints.append(
                    MemoryCheckpoint(
                        checkpoint_id=row["id"],
                        checkpoint_name=row["checkpoint_name"],
                        created_at=row["created_at"],
                        created_by=row["created_by"],
                        reason=row["reason"],
                        memory_snapshot=MemorySnapshot(**snapshot_data),
                    )
                )

            return checkpoints

    # ========================================
    # INGESTION MANIFEST
    # ========================================

    async def create_manifest_entry(
        self,
        source_file: str,
        file_type: str,
        knowledge_type: Optional[str] = None,
    ) -> IngestionManifestEntry:
        """Create a new ingestion manifest entry."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO ingestion_manifest 
                    (source_file, file_type, knowledge_type, status)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (source_file) DO UPDATE SET
                    file_type = EXCLUDED.file_type,
                    knowledge_type = EXCLUDED.knowledge_type,
                    status = 'pending',
                    updated_at = NOW()
                RETURNING *
                """,
                source_file,
                file_type,
                knowledge_type,
                IngestionStatus.PENDING.value,
            )
            return IngestionManifestEntry(**dict(row))

    async def update_manifest_status(
        self,
        source_file: str,
        status: IngestionStatus,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> Optional[IngestionManifestEntry]:
        """Update manifest entry status."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE ingestion_manifest 
                SET status = $2, 
                    chunk_count = $3, 
                    error_message = $4,
                    content_hash = $5,
                    ingested_at = CASE WHEN $2 = 'completed' THEN NOW() ELSE ingested_at END,
                    updated_at = NOW()
                WHERE source_file = $1
                RETURNING *
                """,
                source_file,
                status.value,
                chunk_count,
                error_message,
                content_hash,
            )
            return IngestionManifestEntry(**dict(row)) if row else None

    async def get_manifest_entry(
        self,
        source_file: str,
    ) -> Optional[IngestionManifestEntry]:
        """Get manifest entry for a file."""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ingestion_manifest WHERE source_file = $1",
                source_file,
            )
            return IngestionManifestEntry(**dict(row)) if row else None

    async def needs_reingestion(
        self,
        source_file: str,
        content_hash: str,
    ) -> bool:
        """Check if a file needs to be re-ingested."""
        entry = await self.get_manifest_entry(source_file)
        if not entry:
            return True
        if entry.status != IngestionStatus.COMPLETED:
            return True
        if entry.content_hash != content_hash:
            return True
        return False
