"""
Pydantic models and schemas for the Apex RAG system.

These models define the data structures for:
- Semantic chunks (document embeddings)
- Code entities (DKB nodes)
- Memory notes (agent LTM)
- Ingestion manifests
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================
# ENUMS
# ============================================

class MemoryType(str, Enum):
    """Types of agent memory."""
    EPISODIC = "episodic"      # What happened (experiences)
    SEMANTIC = "semantic"      # What is (facts, knowledge)
    PROCEDURAL = "procedural"  # How to (processes, workflows)


class KnowledgeType(str, Enum):
    """Types of knowledge in the system."""
    SNIPPETS = "snippets"           # Code snippets
    LIBRARY_DOCS = "library_docs"   # API documentation
    GUIDANCE = "guidance"           # Best practices, tutorials
    ERROR_FIXES = "error_fixes"     # Error messages and solutions
    MEMORIES = "memories"           # Agent-generated memories


class EntityType(str, Enum):
    """Types of code entities."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class IngestionStatus(str, Enum):
    """Status of document ingestion."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================
# SEMANTIC CHUNKS
# ============================================

class SemanticChunkBase(BaseModel):
    """Base model for semantic chunks."""
    source_file: str
    chunk_index: int
    header_path: Optional[str] = None
    content: str
    metadata: dict = Field(default_factory=dict)


class SemanticChunkCreate(SemanticChunkBase):
    """Model for creating a new semantic chunk."""
    embedding: Optional[list[float]] = None
    content_hash: Optional[str] = None


class SemanticChunk(SemanticChunkBase):
    """Full semantic chunk model with all fields."""
    id: UUID
    embedding: Optional[list[float]] = None
    content_hash: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# CODE ENTITIES
# ============================================

class CodeEntityBase(BaseModel):
    """Base model for code entities."""
    entity_type: EntityType
    name: str
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class CodeEntityCreate(CodeEntityBase):
    """Model for creating a new code entity."""
    id: str  # DKB entity ID (e.g., "src/user.py:UserService.get_user")
    docstring_embedding: Optional[list[float]] = None
    content_hash: Optional[str] = None


class CodeEntity(CodeEntityBase):
    """Full code entity model with all fields."""
    id: str
    docstring_embedding: Optional[list[float]] = None
    content_hash: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# MEMORY NOTES
# ============================================

class MemoryNoteBase(BaseModel):
    """Base model for memory notes."""
    content: str
    memory_type: MemoryType
    context: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    source_ref: Optional[str] = None


class MemoryNoteCreate(MemoryNoteBase):
    """Model for creating a new memory note."""
    agent_id: str = "default"
    embedding: Optional[list[float]] = None


class MemoryNote(MemoryNoteBase):
    """Full memory note model with all fields."""
    id: UUID
    agent_id: str
    embedding: Optional[list[float]] = None
    usage_count: int = 0
    last_accessed: datetime
    created_at: datetime
    is_active: bool = True
    deleted_reason: Optional[str] = None
    restored_from: Optional[UUID] = None

    class Config:
        from_attributes = True


# ============================================
# MEMORY CHECKPOINTS
# ============================================

class MemorySnapshot(BaseModel):
    """Snapshot of memory state for checkpoints."""
    memories: list[dict]
    memory_links: list[dict] = Field(default_factory=list)
    timestamp: datetime
    memory_count: int


class MemoryCheckpoint(BaseModel):
    """Memory checkpoint for rollback."""
    checkpoint_id: UUID
    checkpoint_name: str
    created_at: datetime
    created_by: str
    reason: Optional[str] = None
    memory_snapshot: MemorySnapshot
    is_current: bool = False

    class Config:
        from_attributes = True


# ============================================
# INGESTION MANIFEST
# ============================================

class IngestionManifestEntry(BaseModel):
    """Entry in the ingestion manifest."""
    id: UUID
    source_file: str
    output_file: Optional[str] = None
    file_type: str
    knowledge_type: Optional[str] = None
    chunk_count: int = 0
    status: IngestionStatus = IngestionStatus.PENDING
    error_message: Optional[str] = None
    content_hash: Optional[str] = None
    ingested_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# SEARCH RESULTS
# ============================================

class SearchResult(BaseModel):
    """A single search result."""
    id: UUID
    content: str
    source_file: str
    header_path: Optional[str] = None
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    relevance: Optional[str] = None  # "RELEVANT", "PARTIAL", "IRRELEVANT"
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response from a search query."""
    query: str
    results: list[SearchResult]
    total_found: int
    strategy_used: str = "SEMANTIC_SEARCH"
    confidence: float = 0.0


# ============================================
# DKB (DETERMINISTIC KNOWLEDGE BASE) GRAPH
# ============================================

class DKBNode(BaseModel):
    """A node in the DKB graph."""
    id: str
    type: EntityType
    name: str
    parent_class: Optional[str] = None
    file_path: str
    start_line: int
    end_line: int
    language: str
    content_hash: str
    docstring_summary: Optional[str] = None
    signature: Optional[str] = None
    modifiers: list[str] = Field(default_factory=list)


class DKBLink(BaseModel):
    """An edge in the DKB graph."""
    source: str
    target: str
    relation: str  # "CALLS", "IMPORTS", "INHERITS", etc.
    weight: float = 1.0
    metadata: dict = Field(default_factory=dict)


class DKBGraph(BaseModel):
    """The complete DKB graph structure."""
    directed: bool = True
    multigraph: bool = False
    graph_metadata: dict = Field(default_factory=dict)
    nodes: list[DKBNode]
    links: list[DKBLink]


# ============================================
# UNIVERSAL METADATA
# ============================================

class UniversalMetadata(BaseModel):
    """Universal metadata schema for all ingested content."""
    source_file: str
    title: Optional[str] = None
    knowledge_type: Optional[KnowledgeType] = None
    domain_tags: list[str] = Field(default_factory=list)
    language_tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    original_format: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: Optional[str] = None
