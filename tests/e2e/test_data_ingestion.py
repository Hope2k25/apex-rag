"""
End-to-end tests for data ingestion workflows.

Tests loading data into the system with real database connections.
"""

import pytest

from src.storage.schemas import (
    SemanticChunkCreate,
    IngestionStatus,
)
from src.utils.embedding import EmbeddingModel


@pytest.mark.e2e
class TestSemanticChunkIngestion:
    """Tests for ingesting semantic chunks."""

    async def test_create_single_chunk(self, postgres_client, embedding_model):
        """Should successfully create a single semantic chunk."""
        # Generate embedding
        embedding = embedding_model.embed("Test content for chunk")

        chunk = SemanticChunkCreate(
            source_file="test_file.md",
            chunk_index=0,
            header_path="Section 1",
            content="This is test content for the semantic chunk.",
            embedding=embedding,
            content_hash="test_hash_123",
            metadata={"type": "test", "author": "e2e"},
        )

        result = await postgres_client.create_chunk(chunk)

        # Verify chunk was created
        assert result.id is not None
        assert result.source_file == "test_file.md"
        assert result.chunk_index == 0
        assert result.content == "This is test content for the semantic chunk."
        assert result.header_path == "Section 1"
        assert result.content_hash == "test_hash_123"

    async def test_create_multiple_chunks(self, postgres_client, sample_chunks):
        """Should successfully create multiple chunks."""
        # Create all chunks
        created_chunks = []
        for chunk in sample_chunks:
            created = await postgres_client.create_chunk(chunk)
            created_chunks.append(created)

        # Verify all chunks were created
        assert len(created_chunks) == 3
        for chunk in created_chunks:
            assert chunk.id is not None
            assert chunk.source_file == "test_doc.md"

        # Verify chunk indices
        indices = [c.chunk_index for c in created_chunks]
        assert sorted(indices) == [0, 1, 2]

    async def test_get_chunk_by_id(self, postgres_client, sample_chunks):
        """Should retrieve a chunk by its ID."""
        # Create a chunk
        created = await postgres_client.create_chunk(sample_chunks[0])

        # Retrieve by ID
        retrieved = await postgres_client.get_chunk(created.id)

        # Verify retrieval
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.content == created.content
        assert retrieved.source_file == created.source_file

    async def test_get_chunks_by_file(self, postgres_client, sample_chunks):
        """Should retrieve all chunks for a file."""
        # Create chunks
        for chunk in sample_chunks:
            await postgres_client.create_chunk(chunk)

        # Retrieve all chunks for the file
        chunks = await postgres_client.get_chunks_by_file("test_doc.md")

        # Verify retrieval
        assert len(chunks) == 3
        assert all(c.source_file == "test_doc.md" for c in chunks)
        # Verify ordering by chunk_index
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2

    async def test_delete_chunks_by_file(self, postgres_client, sample_chunks):
        """Should delete all chunks for a file."""
        # Create chunks
        for chunk in sample_chunks:
            await postgres_client.create_chunk(chunk)

        # Verify chunks exist
        chunks_before = await postgres_client.get_chunks_by_file("test_doc.md")
        assert len(chunks_before) == 3

        # Delete chunks
        deleted_count = await postgres_client.delete_chunks_by_file("test_doc.md")
        assert deleted_count == 3

        # Verify chunks are deleted
        chunks_after = await postgres_client.get_chunks_by_file("test_doc.md")
        assert len(chunks_after) == 0

    async def test_update_existing_chunk(self, postgres_client, embedding_model):
        """Should update an existing chunk (upsert)."""
        # Create initial chunk
        embedding1 = embedding_model.embed("Original content")
        chunk1 = SemanticChunkCreate(
            source_file="update_test.md",
            chunk_index=0,
            header_path="Section",
            content="Original content",
            embedding=embedding1,
            content_hash="hash1",
        )
        created = await postgres_client.create_chunk(chunk1)

        # Update with new content (same source_file + chunk_index)
        embedding2 = embedding_model.embed("Updated content")
        chunk2 = SemanticChunkCreate(
            source_file="update_test.md",
            chunk_index=0,
            header_path="Section Updated",
            content="Updated content",
            embedding=embedding2,
            content_hash="hash2",
        )
        updated = await postgres_client.create_chunk(chunk2)

        # Verify update (same ID)
        assert updated.id == created.id
        assert updated.content == "Updated content"
        assert updated.header_path == "Section Updated"
        assert updated.content_hash == "hash2"

        # Verify only one chunk exists
        chunks = await postgres_client.get_chunks_by_file("update_test.md")
        assert len(chunks) == 1


@pytest.mark.e2e
class TestIngestionManifest:
    """Tests for ingestion manifest tracking."""

    async def test_create_manifest_entry(self, postgres_client):
        """Should create a manifest entry."""
        entry = await postgres_client.create_manifest_entry(
            source_file="test_doc.pdf",
            file_type="pdf",
            knowledge_type="documentation",
        )

        assert entry.id is not None
        assert entry.source_file == "test_doc.pdf"
        assert entry.file_type == "pdf"
        assert entry.knowledge_type == "documentation"
        assert entry.status == IngestionStatus.PENDING

    async def test_update_manifest_status(self, postgres_client):
        """Should update manifest status."""
        # Create entry
        entry = await postgres_client.create_manifest_entry(
            source_file="status_test.md",
            file_type="markdown",
        )

        # Update to completed
        updated = await postgres_client.update_manifest_status(
            source_file="status_test.md",
            status=IngestionStatus.COMPLETED,
            chunk_count=5,
            content_hash="content_hash_123",
        )

        assert updated.status == IngestionStatus.COMPLETED
        assert updated.chunk_count == 5
        assert updated.content_hash == "content_hash_123"
        assert updated.ingested_at is not None

    async def test_update_manifest_to_failed(self, postgres_client):
        """Should update manifest to failed status."""
        # Create entry
        entry = await postgres_client.create_manifest_entry(
            source_file="failed_test.txt",
            file_type="text",
        )

        # Update to failed
        updated = await postgres_client.update_manifest_status(
            source_file="failed_test.txt",
            status=IngestionStatus.FAILED,
            error_message="File parsing failed",
        )

        assert updated.status == IngestionStatus.FAILED
        assert updated.error_message == "File parsing failed"

    async def test_get_manifest_entry(self, postgres_client):
        """Should retrieve manifest entry by source file."""
        # Create entry
        created = await postgres_client.create_manifest_entry(
            source_file="get_test.docx",
            file_type="docx",
        )

        # Retrieve entry
        retrieved = await postgres_client.get_manifest_entry("get_test.docx")

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.source_file == "get_test.docx"

    async def test_needs_reingestion_new_file(self, postgres_client):
        """Should return True for files not in manifest."""
        # File not in manifest
        needs_update = await postgres_client.needs_reingestion(
            source_file="new_file.pdf",
            content_hash="hash123",
        )

        assert needs_update is True

    async def test_needs_reingestion_status_pending(self, postgres_client):
        """Should return True for files with pending status."""
        # Create pending entry
        await postgres_client.create_manifest_entry(
            source_file="pending_file.pdf",
            file_type="pdf",
        )

        needs_update = await postgres_client.needs_reingestion(
            source_file="pending_file.pdf",
            content_hash="hash456",
        )

        assert needs_update is True

    async def test_needs_reingestion_hash_changed(self, postgres_client):
        """Should return True when content hash changes."""
        # Create completed entry
        await postgres_client.create_manifest_entry(
            source_file="hash_changed.pdf",
            file_type="pdf",
        )
        await postgres_client.update_manifest_status(
            source_file="hash_changed.pdf",
            status=IngestionStatus.COMPLETED,
            content_hash="original_hash",
        )

        # Check with different hash
        needs_update = await postgres_client.needs_reingestion(
            source_file="hash_changed.pdf",
            content_hash="new_hash",
        )

        assert needs_update is True

    async def test_needs_reingestion_no_change(self, postgres_client):
        """Should return False when hash matches and status is completed."""
        # Create completed entry
        await postgres_client.create_manifest_entry(
            source_file="no_change.pdf",
            file_type="pdf",
        )
        await postgres_client.update_manifest_status(
            source_file="no_change.pdf",
            status=IngestionStatus.COMPLETED,
            content_hash="same_hash",
        )

        # Check with same hash
        needs_update = await postgres_client.needs_reingestion(
            source_file="no_change.pdf",
            content_hash="same_hash",
        )

        assert needs_update is False


@pytest.mark.e2e
class TestVectorOperations:
    """Tests for vector operations with real embeddings."""

    async def test_vector_search_with_embeddings(self, postgres_client, embedding_model):
        """Should perform vector similarity search."""
        # Create chunks with different content
        contents = [
            "Python is a programming language",
            "JavaScript is used for web development",
            "Python has excellent data science libraries",
        ]

        for i, content in enumerate(contents):
            embedding = embedding_model.embed(content)
            chunk = SemanticChunkCreate(
                source_file="vector_test.md",
                chunk_index=i,
                content=content,
                embedding=embedding,
                content_hash=f"hash_{i}",
            )
            await postgres_client.create_chunk(chunk)

        # Search for "Python"
        query_embedding = embedding_model.embed("Python")
        results = await postgres_client.vector_search(
            query_embedding=query_embedding,
            limit=2,
            min_similarity=0.3,
        )

        # Should return Python-related chunks
        assert len(results) >= 1
        # First result should mention Python
        assert "Python" in results[0].content

    async def test_hybrid_search(self, postgres_client, embedding_model):
        """Should perform hybrid search combining vector and BM25."""
        # Create chunks
        contents = [
            "The database connection uses asyncpg library",
            "PostgreSQL supports vector search with pgvector",
            "Connecting to databases requires proper credentials",
        ]

        for i, content in enumerate(contents):
            embedding = embedding_model.embed(content)
            chunk = SemanticChunkCreate(
                source_file="hybrid_test.md",
                chunk_index=i,
                content=content,
                embedding=embedding,
                content_hash=f"hybrid_hash_{i}",
            )
            await postgres_client.create_chunk(chunk)

        # Hybrid search for "database connection"
        query_embedding = embedding_model.embed("database connection")
        results = await postgres_client.hybrid_search(
            query_text="database connection",
            query_embedding=query_embedding,
            alpha=0.5,  # Equal weight
            limit=5,
        )

        # Should return results
        assert len(results) > 0
        # Check that results have both dense and sparse scores
        for result in results:
            assert result.dense_score >= 0
            assert result.sparse_score >= 0
            assert result.combined_score >= 0


@pytest.mark.e2e
class TestIngestionWorkflow:
    """Tests for complete ingestion workflows."""

    async def test_full_ingestion_workflow(self, postgres_client, embedding_model):
        """Should complete full ingestion workflow."""
        # Step 1: Create manifest entry
        manifest = await postgres_client.create_manifest_entry(
            source_file="workflow_test.pdf",
            file_type="pdf",
            knowledge_type="documentation",
        )
        assert manifest.status == IngestionStatus.PENDING

        # Step 2: Create chunks
        chunks_data = [
            "First chapter of the document",
            "Second chapter with important information",
            "Third chapter with technical details",
        ]

        created_chunks = []
        for i, content in enumerate(chunks_data):
            embedding = embedding_model.embed(content)
            chunk = SemanticChunkCreate(
                source_file="workflow_test.pdf",
                chunk_index=i,
                content=content,
                embedding=embedding,
                content_hash=f"workflow_hash_{i}",
            )
            created = await postgres_client.create_chunk(chunk)
            created_chunks.append(created)

        assert len(created_chunks) == 3

        # Step 3: Update manifest to completed
        updated_manifest = await postgres_client.update_manifest_status(
            source_file="workflow_test.pdf",
            status=IngestionStatus.COMPLETED,
            chunk_count=3,
            content_hash="workflow_final_hash",
        )

        assert updated_manifest.status == IngestionStatus.COMPLETED
        assert updated_manifest.chunk_count == 3

        # Step 4: Verify chunks can be retrieved
        retrieved_chunks = await postgres_client.get_chunks_by_file("workflow_test.pdf")
        assert len(retrieved_chunks) == 3

        # Step 5: Verify no reingestion needed
        needs_update = await postgres_client.needs_reingestion(
            source_file="workflow_test.pdf",
            content_hash="workflow_final_hash",
        )
        assert needs_update is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
