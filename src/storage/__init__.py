"""Storage module for database clients."""

from .postgres_client import PostgresClient
from .neo4j_client import Neo4jClient
from .schemas import (
    # Core models
    SemanticChunk,
    SemanticChunkCreate,
    CodeEntity,
    CodeEntityCreate,
    MemoryNote,
    MemoryNoteCreate,
    MemoryCheckpoint,
    MemorySnapshot,
    SearchResult,
    SearchResponse,
    DKBGraph,
    DKBNode,
    DKBLink,
    UniversalMetadata,
    IngestionManifestEntry,
    # Enums
    MemoryType,
    KnowledgeType,
    EntityType,
    IngestionStatus,
    # Library Documentation (Multi-Language)
    ProgrammingLanguage,
    PackageEcosystem,
    APIElementType,
    LibraryInfo,
    ParameterInfo,
    ReturnInfo,
    ExampleInfo,
    ErrorPatternBase,
    ErrorPattern,
    APIElementBase,
    APIElementCreate,
    APIElement,
    ErrorLookupResult,
    # Project Dependencies
    DependencyInfo,
    ProjectDependencies,
)

__all__ = [
    # Clients
    "PostgresClient",
    "Neo4jClient",
    # Core models
    "SemanticChunk",
    "SemanticChunkCreate",
    "CodeEntity",
    "CodeEntityCreate",
    "MemoryNote",
    "MemoryNoteCreate",
    "MemoryCheckpoint",
    "MemorySnapshot",
    "SearchResult",
    "SearchResponse",
    "DKBGraph",
    "DKBNode",
    "DKBLink",
    "UniversalMetadata",
    "IngestionManifestEntry",
    # Enums
    "MemoryType",
    "KnowledgeType",
    "EntityType",
    "IngestionStatus",
    # Library Documentation
    "ProgrammingLanguage",
    "PackageEcosystem",
    "APIElementType",
    "LibraryInfo",
    "ParameterInfo",
    "ReturnInfo",
    "ExampleInfo",
    "ErrorPatternBase",
    "ErrorPattern",
    "APIElementBase",
    "APIElementCreate",
    "APIElement",
    "ErrorLookupResult",
    # Project Dependencies
    "DependencyInfo",
    "ProjectDependencies",
]
