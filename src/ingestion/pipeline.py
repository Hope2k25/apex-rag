"""
Ingestion Pipeline - Manual-control orchestrator for Apex RAG.

This module provides the main ingestion pipeline that:
1. Parses documents (HTML, Markdown, RST, PDF)
2. Chunks them into semantic units
3. Generates embeddings
4. Stores in PostgreSQL with pgvector

IMPORTANT: This pipeline does NOT auto-ingest. All ingestion
must be explicitly triggered by the user via:
- CLI commands
- MCP tools  
- Direct API calls

The pipeline is designed for manual control over what gets ingested.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import json

from .document_parser import DocumentParser, ParsedDocument, DocumentType
from .chunker import Chunker, ChunkConfig, Chunk
from .code_indexer import CodeIndexer, index_codebase, IndexedCodebase


class IngestionStatus(str, Enum):
    """Status of an ingestion job."""
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionResult:
    """Result of ingesting a single file."""
    source_file: str
    status: IngestionStatus
    document_type: Optional[str] = None
    chunk_count: int = 0
    content_hash: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0
    llm_enrichment_pending: bool = True


@dataclass
class BatchIngestionResult:
    """Result of ingesting multiple files."""
    total_files: int
    successful: int
    failed: int
    results: list[IngestionResult] = field(default_factory=list)
    processing_time_ms: float = 0


class IngestionPipeline:
    """
    Manual-control ingestion pipeline.

    USAGE:
    ------
    pipeline = IngestionPipeline()
    await pipeline.connect()

    # Ingest a single file
    result = await pipeline.ingest_file("path/to/doc.md")

    # Ingest a directory
    results = await pipeline.ingest_directory("path/to/docs/", pattern="*.md")

    # Index code
    graph = await pipeline.index_code("path/to/src/")

    await pipeline.close()
    """

    def __init__(
        self,
        chunk_config: Optional[ChunkConfig] = None,
        embedding_batch_size: int = 32,
        skip_embedding: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            chunk_config: Configuration for the chunker
            embedding_batch_size: Batch size for embedding generation
            skip_embedding: If True, skip embedding (useful for testing)
        """
        self.chunk_config = chunk_config or ChunkConfig()
        self.embedding_batch_size = embedding_batch_size
        self.skip_embedding = skip_embedding

        self._parser = DocumentParser()
        self._chunker = Chunker(self.chunk_config)
        self._code_indexer = CodeIndexer()

        # Database clients (lazy initialized)
        self._postgres = None
        self._neo4j = None
        self._embedding_model = None

    async def connect(self):
        """Connect to databases and initialize models."""
        from ..storage.postgres_client import PostgresClient
        from ..storage.neo4j_client import Neo4jClient
        from ..utils.embedding import EmbeddingModel

        self._postgres = PostgresClient()
        await self._postgres.connect()

        self._neo4j = Neo4jClient()
        await self._neo4j.connect()

        if not self.skip_embedding:
            self._embedding_model = EmbeddingModel()
            # Warm up the model
            _ = await asyncio.to_thread(
                self._embedding_model.embed, "warmup"
            )

    async def close(self):
        """Close all connections."""
        if self._postgres:
            await self._postgres.close()
        if self._neo4j:
            await self._neo4j.close()

    # ========================================
    # DOCUMENT INGESTION
    # ========================================

    async def ingest_file(
        self,
        file_path: str | Path,
        knowledge_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> IngestionResult:
        """
        Ingest a single document file.

        Args:
            file_path: Path to the document
            knowledge_type: Optional type hint (api, tutorial, etc.)
            metadata: Optional additional metadata

        Returns:
            IngestionResult with status and details
        """
        start_time = datetime.now()
        path = Path(file_path)

        try:
            # Check if already ingested (by content hash)
            existing = await self._check_existing(str(path))
            if existing:
                return IngestionResult(
                    source_file=str(path),
                    status=IngestionStatus.COMPLETED,
                    chunk_count=existing.get("chunk_count", 0),
                    content_hash=existing.get("content_hash"),
                    processing_time_ms=0,
                    llm_enrichment_pending=existing.get("llm_enrichment_status") == "pending",
                )

            # Parse document
            parsed = await asyncio.to_thread(self._parser.parse, path)

            # Chunk document
            chunks = await asyncio.to_thread(self._chunker.chunk, parsed)

            if not chunks:
                return IngestionResult(
                    source_file=str(path),
                    status=IngestionStatus.COMPLETED,
                    document_type=parsed.document_type.value,
                    chunk_count=0,
                    content_hash=parsed.content_hash,
                    processing_time_ms=self._elapsed_ms(start_time),
                )

            # Generate embeddings
            if not self.skip_embedding and self._embedding_model:
                embeddings = await self._generate_embeddings(chunks)
            else:
                embeddings = [None] * len(chunks)

            # Store chunks
            await self._store_chunks(chunks, embeddings, parsed, knowledge_type, metadata)

            # Update manifest
            await self._update_manifest(
                source_file=str(path),
                file_type=parsed.document_type.value,
                knowledge_type=knowledge_type,
                chunk_count=len(chunks),
                content_hash=parsed.content_hash,
                status="completed",
            )

            return IngestionResult(
                source_file=str(path),
                status=IngestionStatus.COMPLETED,
                document_type=parsed.document_type.value,
                chunk_count=len(chunks),
                content_hash=parsed.content_hash,
                processing_time_ms=self._elapsed_ms(start_time),
                llm_enrichment_pending=True,
            )

        except Exception as e:
            await self._update_manifest(
                source_file=str(path),
                file_type="unknown",
                status="failed",
                error_message=str(e),
            )

            return IngestionResult(
                source_file=str(path),
                status=IngestionStatus.FAILED,
                error_message=str(e),
                processing_time_ms=self._elapsed_ms(start_time),
            )

    async def ingest_directory(
        self,
        directory: str | Path,
        pattern: str = "*",
        recursive: bool = True,
        knowledge_type: Optional[str] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> BatchIngestionResult:
        """
        Ingest all matching files in a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern (e.g., "*.md", "*.html")
            recursive: If True, search subdirectories
            knowledge_type: Optional type for all files
            exclude_patterns: Patterns to exclude

        Returns:
            BatchIngestionResult with all results
        """
        import fnmatch

        start_time = datetime.now()
        path = Path(directory)
        exclude_patterns = exclude_patterns or []

        # Find matching files
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        # Apply exclusions
        files = [
            f for f in files
            if not any(fnmatch.fnmatch(str(f), p) for p in exclude_patterns)
        ]

        results = []
        successful = 0
        failed = 0

        for file_path in files:
            result = await self.ingest_file(file_path, knowledge_type)
            results.append(result)

            if result.status == IngestionStatus.COMPLETED:
                successful += 1
            else:
                failed += 1

        return BatchIngestionResult(
            total_files=len(files),
            successful=successful,
            failed=failed,
            results=results,
            processing_time_ms=self._elapsed_ms(start_time),
        )

    # ========================================
    # CODE INDEXING
    # ========================================

    async def index_code(
        self,
        directory: str | Path,
        output_graph_path: Optional[str | Path] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> IndexedCodebase:
        """
        Index a codebase and store entities in Neo4j.

        Args:
            directory: Root directory of codebase
            output_graph_path: Optional path to save DKB graph JSON
            exclude_patterns: Patterns to exclude

        Returns:
            IndexedCodebase with all entities
        """
        # Index code (CPU-bound)
        indexed = await asyncio.to_thread(
            self._code_indexer.index_directory,
            directory,
            exclude_patterns,
        )

        # Convert to DKB graph
        graph = self._code_indexer.to_dkb_graph(indexed)

        # Save graph file if requested
        if output_graph_path:
            await asyncio.to_thread(
                self._code_indexer.save_dkb_graph,
                graph,
                output_graph_path,
            )

        # Store in Neo4j
        if self._neo4j:
            await self._neo4j.load_dkb_graph(graph)

        # Store code entities in Postgres (for embedding search)
        if self._postgres and not self.skip_embedding:
            await self._store_code_entities(indexed)

        return indexed

    # ========================================
    # HELPER METHODS
    # ========================================

    async def _check_existing(self, source_file: str) -> Optional[dict]:
        """Check if file has already been ingested."""
        if not self._postgres:
            return None

        try:
            result = await self._postgres.get_manifest_entry(source_file)
            if result and result.get("status") == "completed":
                return result
        except Exception:
            pass
        return None

    async def _generate_embeddings(self, chunks: list[Chunk]) -> list[list[float]]:
        """Generate embeddings for chunks in batches."""
        texts = [chunk.content for chunk in chunks]

        all_embeddings = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
            batch_embeddings = await asyncio.to_thread(
                self._embedding_model.embed_batch, batch
            )
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _store_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[Optional[list[float]]],
        parsed: ParsedDocument,
        knowledge_type: Optional[str],
        metadata: Optional[dict],
    ):
        """Store chunks in PostgreSQL."""
        if not self._postgres:
            return

        for chunk, embedding in zip(chunks, embeddings):
            chunk_metadata = {
                "title": parsed.title,
                "document_type": parsed.document_type.value,
                "knowledge_type": knowledge_type,
                **(parsed.frontmatter or {}),
                **(metadata or {}),
            }

            await self._postgres.upsert_chunk(
                source_file=chunk.source_file,
                chunk_index=chunk.index,
                content=chunk.content,
                header_path=chunk.header_path,
                embedding=embedding,
                metadata=chunk_metadata,
                content_hash=chunk.content_hash,
            )

    async def _store_code_entities(self, indexed: IndexedCodebase):
        """Store code entities in PostgreSQL with embeddings."""
        for entity in indexed.entities:
            # Create embeddable text
            embed_text = entity.name
            if entity.signature:
                embed_text = entity.signature
            if entity.docstring:
                embed_text += f"\n{entity.docstring}"

            embedding = None
            if self._embedding_model:
                embedding = await asyncio.to_thread(
                    self._embedding_model.embed, embed_text
                )

            await self._postgres.upsert_code_entity(
                id=entity.id,
                entity_type=entity.entity_type.value,
                name=entity.name,
                file_path=entity.file_path,
                start_line=entity.start_line,
                end_line=entity.end_line,
                signature=entity.signature,
                docstring=entity.docstring,
                docstring_embedding=embedding,
                content_hash=entity.content_hash,
            )

    async def _update_manifest(
        self,
        source_file: str,
        file_type: str,
        status: str = "pending",
        knowledge_type: Optional[str] = None,
        chunk_count: int = 0,
        content_hash: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Update the ingestion manifest."""
        if not self._postgres:
            return

        await self._postgres.upsert_manifest(
            source_file=source_file,
            file_type=file_type,
            status=status,
            knowledge_type=knowledge_type,
            chunk_count=chunk_count,
            content_hash=content_hash,
            error_message=error_message,
        )

    def _elapsed_ms(self, start: datetime) -> float:
        """Calculate elapsed time in milliseconds."""
        return (datetime.now() - start).total_seconds() * 1000


# ========================================
# CLI FUNCTIONS (for manual triggering)
# ========================================

async def ingest_files_cli(
    files: list[str],
    knowledge_type: Optional[str] = None,
    skip_embedding: bool = False,
) -> BatchIngestionResult:
    """
    CLI entry point for ingesting specific files.

    Example:
        python -m src.ingestion.pipeline --files doc1.md doc2.html
    """
    pipeline = IngestionPipeline(skip_embedding=skip_embedding)
    await pipeline.connect()

    results = []
    for file_path in files:
        result = await pipeline.ingest_file(file_path, knowledge_type)
        results.append(result)
        print(f"[{result.status.value}] {file_path}: {result.chunk_count} chunks")

    await pipeline.close()

    successful = sum(1 for r in results if r.status == IngestionStatus.COMPLETED)
    failed = len(results) - successful

    return BatchIngestionResult(
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results,
    )


async def ingest_directory_cli(
    directory: str,
    pattern: str = "*.md",
    knowledge_type: Optional[str] = None,
    skip_embedding: bool = False,
) -> BatchIngestionResult:
    """
    CLI entry point for ingesting a directory.

    Example:
        python -m src.ingestion.pipeline --directory ./docs --pattern "*.md"
    """
    pipeline = IngestionPipeline(skip_embedding=skip_embedding)
    await pipeline.connect()

    result = await pipeline.ingest_directory(directory, pattern, knowledge_type=knowledge_type)

    await pipeline.close()

    print(f"\nIngestion complete: {result.successful}/{result.total_files} successful")
    return result


async def index_codebase_cli(
    directory: str,
    output_path: Optional[str] = None,
    skip_embedding: bool = False,
) -> IndexedCodebase:
    """
    CLI entry point for indexing a codebase.

    Example:
        python -m src.ingestion.pipeline --code ./src --output dkb.json
    """
    pipeline = IngestionPipeline(skip_embedding=skip_embedding)
    await pipeline.connect()

    result = await pipeline.index_code(directory, output_path)

    await pipeline.close()

    print(f"\nIndexed {result.files_indexed} files, {len(result.entities)} entities")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apex RAG Ingestion Pipeline")
    parser.add_argument("--files", nargs="+", help="Specific files to ingest")
    parser.add_argument("--directory", help="Directory to ingest")
    parser.add_argument("--pattern", default="*.md", help="File pattern for directory")
    parser.add_argument("--code", help="Codebase directory to index")
    parser.add_argument("--output", help="Output path for DKB graph")
    parser.add_argument("--type", help="Knowledge type")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding generation")

    args = parser.parse_args()

    if args.files:
        asyncio.run(ingest_files_cli(args.files, args.type, args.skip_embedding))
    elif args.directory:
        asyncio.run(ingest_directory_cli(args.directory, args.pattern, args.type, args.skip_embedding))
    elif args.code:
        asyncio.run(index_codebase_cli(args.code, args.output, args.skip_embedding))
    else:
        parser.print_help()
