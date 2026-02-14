"""
Tests for Pydantic schemas.

Run with: uv run pytest tests/test_schemas.py -v
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.storage.schemas import (
    # Enums
    MemoryType,
    KnowledgeType,
    EntityType,
    IngestionStatus,
    ProgrammingLanguage,
    PackageEcosystem,
    APIElementType,
    # Semantic Chunks
    SemanticChunkBase,
    SemanticChunkCreate,
    SemanticChunk,
    # Code Entities
    CodeEntityBase,
    CodeEntityCreate,
    CodeEntity,
    # Memory Notes
    MemoryNoteBase,
    MemoryNoteCreate,
    MemoryNote,
    # Memory Checkpoints
    MemorySnapshot,
    MemoryCheckpoint,
    # Ingestion Manifest
    IngestionManifestEntry,
    # Search Results
    SearchResult,
    SearchResponse,
    # DKB Graph
    DKBNode,
    DKBLink,
    DKBGraph,
    # Universal Metadata
    UniversalMetadata,
    # Library Documentation
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


class TestEnums:
    """Tests for enum definitions."""
    
    def test_memory_type_values(self):
        """MemoryType should have correct values."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
    
    def test_knowledge_type_values(self):
        """KnowledgeType should have correct values."""
        assert KnowledgeType.SNIPPETS.value == "snippets"
        assert KnowledgeType.LIBRARY_DOCS.value == "library_docs"
        assert KnowledgeType.GUIDANCE.value == "guidance"
        assert KnowledgeType.ERROR_FIXES.value == "error_fixes"
        assert KnowledgeType.MEMORIES.value == "memories"
    
    def test_entity_type_values(self):
        """EntityType should have correct values."""
        assert EntityType.MODULE.value == "module"
        assert EntityType.CLASS.value == "class"
        assert EntityType.FUNCTION.value == "function"
        assert EntityType.METHOD.value == "method"
    
    def test_ingestion_status_values(self):
        """IngestionStatus should have correct values."""
        assert IngestionStatus.PENDING.value == "pending"
        assert IngestionStatus.PROCESSING.value == "processing"
        assert IngestionStatus.COMPLETED.value == "completed"
        assert IngestionStatus.FAILED.value == "failed"
    
    def test_programming_language_values(self):
        """ProgrammingLanguage should have correct values."""
        assert ProgrammingLanguage.PYTHON.value == "python"
        assert ProgrammingLanguage.JAVASCRIPT.value == "javascript"
        assert ProgrammingLanguage.TYPESCRIPT.value == "typescript"
        assert ProgrammingLanguage.GO.value == "go"
        assert ProgrammingLanguage.RUST.value == "rust"
        assert ProgrammingLanguage.CSHARP.value == "csharp"
        assert ProgrammingLanguage.JAVA.value == "java"
    
    def test_package_ecosystem_values(self):
        """PackageEcosystem should have correct values."""
        assert PackageEcosystem.PYPI.value == "pypi"
        assert PackageEcosystem.NPM.value == "npm"
        assert PackageEcosystem.CRATES.value == "crates"
        assert PackageEcosystem.GO_MODULES.value == "go-modules"
        assert PackageEcosystem.NUGET.value == "nuget"
        assert PackageEcosystem.MAVEN.value == "maven"
    
    def test_api_element_type_values(self):
        """APIElementType should have correct values."""
        assert APIElementType.CLASS.value == "class"
        assert APIElementType.FUNCTION.value == "function"
        assert APIElementType.METHOD.value == "method"
        assert APIElementType.INTERFACE.value == "interface"
        assert APIElementType.STRUCT.value == "struct"
        assert APIElementType.ENUM.value == "enum"
        assert APIElementType.CONSTANT.value == "constant"
        assert APIElementType.TYPE_ALIAS.value == "type_alias"
        assert APIElementType.TRAIT.value == "trait"
        assert APIElementType.IMPL.value == "impl"


class TestSemanticChunkModels:
    """Tests for semantic chunk models."""
    
    def test_semantic_chunk_base(self):
        """SemanticChunkBase should validate correctly."""
        chunk = SemanticChunkBase(
            source_file="test.py",
            chunk_index=0,
            header_path="module.function",
            content="def hello(): pass",
            metadata={"language": "python"},
        )
        assert chunk.source_file == "test.py"
        assert chunk.chunk_index == 0
        assert chunk.header_path == "module.function"
        assert chunk.content == "def hello(): pass"
        assert chunk.metadata == {"language": "python"}
    
    def test_semantic_chunk_base_defaults(self):
        """SemanticChunkBase should have correct defaults."""
        chunk = SemanticChunkBase(
            source_file="test.py",
            chunk_index=0,
            content="content",
        )
        assert chunk.header_path is None
        assert chunk.metadata == {}
    
    def test_semantic_chunk_create(self):
        """SemanticChunkCreate should validate correctly."""
        chunk = SemanticChunkCreate(
            source_file="test.py",
            chunk_index=0,
            content="content",
            embedding=[0.1, 0.2, 0.3],
            content_hash="abc123",
        )
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.content_hash == "abc123"
    
    def test_semantic_chunk(self):
        """SemanticChunk should validate correctly."""
        chunk_id = uuid4()
        now = datetime.utcnow()
        chunk = SemanticChunk(
            id=chunk_id,
            source_file="test.py",
            chunk_index=0,
            content="content",
            embedding=[0.1, 0.2],
            content_hash="hash",
            created_at=now,
            updated_at=now,
        )
        assert chunk.id == chunk_id
        assert chunk.created_at == now


class TestCodeEntityModels:
    """Tests for code entity models."""
    
    def test_code_entity_base(self):
        """CodeEntityBase should validate correctly."""
        entity = CodeEntityBase(
            entity_type=EntityType.FUNCTION,
            name="hello",
            file_path="test.py",
            start_line=1,
            end_line=5,
            signature="def hello():",
            docstring="Say hello",
            metadata={"public": True},
        )
        assert entity.entity_type == EntityType.FUNCTION
        assert entity.name == "hello"
        assert entity.start_line == 1
        assert entity.end_line == 5
    
    def test_code_entity_base_defaults(self):
        """CodeEntityBase should have correct defaults."""
        entity = CodeEntityBase(
            entity_type=EntityType.CLASS,
            name="MyClass",
            file_path="test.py",
        )
        assert entity.start_line is None
        assert entity.end_line is None
        assert entity.signature is None
        assert entity.docstring is None
        assert entity.metadata == {}
    
    def test_code_entity_create(self):
        """CodeEntityCreate should validate correctly."""
        entity = CodeEntityCreate(
            id="test.py:MyClass",
            entity_type=EntityType.CLASS,
            name="MyClass",
            file_path="test.py",
            docstring_embedding=[0.1, 0.2],
            content_hash="hash123",
        )
        assert entity.id == "test.py:MyClass"
        assert entity.docstring_embedding == [0.1, 0.2]
    
    def test_code_entity(self):
        """CodeEntity should validate correctly."""
        now = datetime.utcnow()
        entity = CodeEntity(
            id="test.py:MyClass",
            entity_type=EntityType.CLASS,
            name="MyClass",
            file_path="test.py",
            created_at=now,
            updated_at=now,
        )
        assert entity.id == "test.py:MyClass"


class TestMemoryNoteModels:
    """Tests for memory note models."""
    
    def test_memory_note_base(self):
        """MemoryNoteBase should validate correctly."""
        note = MemoryNoteBase(
            content="Remember this fact",
            memory_type=MemoryType.SEMANTIC,
            context="User asked about X",
            keywords=["fact", "remember"],
            source_ref="chat:123",
        )
        assert note.content == "Remember this fact"
        assert note.memory_type == MemoryType.SEMANTIC
        assert note.context == "User asked about X"
        assert note.keywords == ["fact", "remember"]
    
    def test_memory_note_base_defaults(self):
        """MemoryNoteBase should have correct defaults."""
        note = MemoryNoteBase(
            content="content",
            memory_type=MemoryType.EPISODIC,
        )
        assert note.context is None
        assert note.keywords == []
        assert note.source_ref is None
    
    def test_memory_note_create(self):
        """MemoryNoteCreate should validate correctly."""
        note = MemoryNoteCreate(
            content="content",
            memory_type=MemoryType.PROCEDURAL,
            agent_id="agent-1",
            embedding=[0.1, 0.2],
        )
        assert note.agent_id == "agent-1"
        assert note.embedding == [0.1, 0.2]
    
    def test_memory_note_create_default_agent(self):
        """MemoryNoteCreate should have default agent_id."""
        note = MemoryNoteCreate(
            content="content",
            memory_type=MemoryType.SEMANTIC,
        )
        assert note.agent_id == "default"
    
    def test_memory_note(self):
        """MemoryNote should validate correctly."""
        note_id = uuid4()
        now = datetime.utcnow()
        note = MemoryNote(
            id=note_id,
            content="content",
            memory_type=MemoryType.SEMANTIC,
            agent_id="agent-1",
            usage_count=5,
            last_accessed=now,
            created_at=now,
            is_active=True,
        )
        assert note.id == note_id
        assert note.usage_count == 5
        assert note.is_active is True


class TestMemoryCheckpointModels:
    """Tests for memory checkpoint models."""
    
    def test_memory_snapshot(self):
        """MemorySnapshot should validate correctly."""
        now = datetime.utcnow()
        snapshot = MemorySnapshot(
            memories=[{"id": "1", "content": "test"}],
            memory_links=[{"from": "1", "to": "2"}],
            timestamp=now,
            memory_count=1,
        )
        assert snapshot.memory_count == 1
        assert len(snapshot.memories) == 1
        assert len(snapshot.memory_links) == 1
    
    def test_memory_snapshot_defaults(self):
        """MemorySnapshot should have correct defaults."""
        snapshot = MemorySnapshot(
            memories=[],
            timestamp=datetime.utcnow(),
            memory_count=0,
        )
        assert snapshot.memory_links == []
    
    def test_memory_checkpoint(self):
        """MemoryCheckpoint should validate correctly."""
        checkpoint_id = uuid4()
        now = datetime.utcnow()
        snapshot = MemorySnapshot(
            memories=[],
            timestamp=now,
            memory_count=0,
        )
        checkpoint = MemoryCheckpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_name="checkpoint-1",
            created_at=now,
            created_by="agent-1",
            reason="Before risky operation",
            memory_snapshot=snapshot,
            is_current=True,
        )
        assert checkpoint.checkpoint_id == checkpoint_id
        assert checkpoint.checkpoint_name == "checkpoint-1"
        assert checkpoint.reason == "Before risky operation"
        assert checkpoint.is_current is True


class TestIngestionManifestModels:
    """Tests for ingestion manifest models."""
    
    def test_ingestion_manifest_entry(self):
        """IngestionManifestEntry should validate correctly."""
        entry_id = uuid4()
        now = datetime.utcnow()
        entry = IngestionManifestEntry(
            id=entry_id,
            source_file="test.pdf",
            output_file="test.json",
            file_type="pdf",
            knowledge_type="guidance",
            chunk_count=10,
            status=IngestionStatus.COMPLETED,
            error_message=None,
            content_hash="hash123",
            ingested_at=now,
            created_at=now,
            updated_at=now,
        )
        assert entry.id == entry_id
        assert entry.source_file == "test.pdf"
        assert entry.status == IngestionStatus.COMPLETED
        assert entry.chunk_count == 10
    
    def test_ingestion_manifest_entry_defaults(self):
        """IngestionManifestEntry should have correct defaults."""
        entry_id = uuid4()
        now = datetime.utcnow()
        entry = IngestionManifestEntry(
            id=entry_id,
            source_file="test.txt",
            file_type="text",
            created_at=now,
            updated_at=now,
        )
        assert entry.output_file is None
        assert entry.knowledge_type is None
        assert entry.chunk_count == 0
        assert entry.status == IngestionStatus.PENDING
        assert entry.error_message is None
        assert entry.content_hash is None


class TestSearchResultModels:
    """Tests for search result models."""
    
    def test_search_result(self):
        """SearchResult should validate correctly."""
        result_id = uuid4()
        result = SearchResult(
            id=result_id,
            content="matching content",
            source_file="test.py",
            header_path="module.function",
            dense_score=0.85,
            sparse_score=0.70,
            combined_score=0.78,
            relevance="RELEVANT",
            metadata={"language": "python"},
        )
        assert result.id == result_id
        assert result.combined_score == 0.78
        assert result.relevance == "RELEVANT"
    
    def test_search_result_defaults(self):
        """SearchResult should have correct defaults."""
        result_id = uuid4()
        result = SearchResult(
            id=result_id,
            content="content",
            source_file="test.py",
        )
        assert result.header_path is None
        assert result.dense_score == 0.0
        assert result.sparse_score == 0.0
        assert result.combined_score == 0.0
        assert result.relevance is None
        assert result.metadata == {}
    
    def test_search_response(self):
        """SearchResponse should validate correctly."""
        result_id = uuid4()
        results = [
            SearchResult(
                id=result_id,
                content="content",
                source_file="test.py",
            )
        ]
        response = SearchResponse(
            query="test query",
            results=results,
            total_found=1,
            strategy_used="SEMANTIC_SEARCH",
            confidence=0.85,
        )
        assert response.query == "test query"
        assert response.total_found == 1
        assert response.strategy_used == "SEMANTIC_SEARCH"
        assert response.confidence == 0.85
    
    def test_search_response_defaults(self):
        """SearchResponse should have correct defaults."""
        response = SearchResponse(
            query="test",
            results=[],
            total_found=0,
        )
        assert response.strategy_used == "SEMANTIC_SEARCH"
        assert response.confidence == 0.0


class TestDKBGraphModels:
    """Tests for DKB graph models."""
    
    def test_dkb_node(self):
        """DKBNode should validate correctly."""
        node = DKBNode(
            id="src/user.py:UserService",
            type=EntityType.CLASS,
            name="UserService",
            parent_class="BaseService",
            file_path="src/user.py",
            start_line=10,
            end_line=50,
            language="python",
            content_hash="hash123",
            docstring_summary="User service class",
            signature="class UserService(BaseService):",
            modifiers=["public"],
        )
        assert node.id == "src/user.py:UserService"
        assert node.type == EntityType.CLASS
        assert node.parent_class == "BaseService"
    
    def test_dkb_node_defaults(self):
        """DKBNode should have correct defaults."""
        node = DKBNode(
            id="test.py:func",
            type=EntityType.FUNCTION,
            name="func",
            file_path="test.py",
            start_line=1,
            end_line=5,
            language="python",
            content_hash="hash",
        )
        assert node.parent_class is None
        assert node.docstring_summary is None
        assert node.signature is None
        assert node.modifiers == []
    
    def test_dkb_link(self):
        """DKBLink should validate correctly."""
        link = DKBLink(
            source="src/user.py:UserService",
            target="src/db.py:Database",
            relation="CALLS",
            weight=1.5,
            metadata={"frequency": 10},
        )
        assert link.source == "src/user.py:UserService"
        assert link.relation == "CALLS"
        assert link.weight == 1.5
    
    def test_dkb_link_defaults(self):
        """DKBLink should have correct defaults."""
        link = DKBLink(
            source="a",
            target="b",
            relation="IMPORTS",
        )
        assert link.weight == 1.0
        assert link.metadata == {}
    
    def test_dkb_graph(self):
        """DKBGraph should validate correctly."""
        node = DKBNode(
            id="test",
            type=EntityType.CLASS,
            name="Test",
            file_path="test.py",
            start_line=1,
            end_line=5,
            language="python",
            content_hash="hash",
        )
        link = DKBLink(source="a", target="b", relation="CALLS")
        graph = DKBGraph(
            directed=True,
            multigraph=False,
            graph_metadata={"version": "1.0"},
            nodes=[node],
            links=[link],
        )
        assert graph.directed is True
        assert len(graph.nodes) == 1
        assert len(graph.links) == 1
    
    def test_dkb_graph_defaults(self):
        """DKBGraph should have correct defaults."""
        graph = DKBGraph(
            nodes=[],
            links=[],
        )
        assert graph.directed is True
        assert graph.multigraph is False
        assert graph.graph_metadata == {}


class TestUniversalMetadata:
    """Tests for UniversalMetadata model."""
    
    def test_universal_metadata(self):
        """UniversalMetadata should validate correctly."""
        metadata = UniversalMetadata(
            source_file="test.py",
            title="Test Module",
            knowledge_type=KnowledgeType.SNIPPETS,
            domain_tags=["backend", "api"],
            language_tags=["python"],
            keywords=["function", "test"],
            original_format="py",
            content_hash="hash123",
            extracted_at=datetime.utcnow(),
        )
        assert metadata.source_file == "test.py"
        assert metadata.title == "Test Module"
        assert metadata.knowledge_type == KnowledgeType.SNIPPETS
    
    def test_universal_metadata_defaults(self):
        """UniversalMetadata should have correct defaults."""
        metadata = UniversalMetadata(
            source_file="test.txt",
        )
        assert metadata.title is None
        assert metadata.knowledge_type is None
        assert metadata.domain_tags == []
        assert metadata.language_tags == []
        assert metadata.keywords == []
        assert metadata.original_format is None
        assert metadata.content_hash is None
        assert isinstance(metadata.extracted_at, datetime)


class TestLibraryDocumentationModels:
    """Tests for library documentation models."""
    
    def test_library_info(self):
        """LibraryInfo should validate correctly."""
        library = LibraryInfo(
            id="pypi:fastapi:0.115.6",
            name="fastapi",
            version="0.115.6",
            language=ProgrammingLanguage.PYTHON,
            ecosystem=PackageEcosystem.PYPI,
            repository_url="https://github.com/tiangolo/fastapi",
            license="MIT",
            homepage_url="https://fastapi.tiangolo.com",
            documentation_url="https://fastapi.tiangolo.com/docs",
        )
        assert library.id == "pypi:fastapi:0.115.6"
        assert library.language == ProgrammingLanguage.PYTHON
        assert library.ecosystem == PackageEcosystem.PYPI
    
    def test_library_info_defaults(self):
        """LibraryInfo should have correct defaults."""
        library = LibraryInfo(
            id="npm:react:18.0.0",
            name="react",
            version="18.0.0",
            language=ProgrammingLanguage.JAVASCRIPT,
            ecosystem=PackageEcosystem.NPM,
        )
        assert library.repository_url is None
        assert library.license is None
        assert isinstance(library.documented_at, datetime)
    
    def test_parameter_info(self):
        """ParameterInfo should validate correctly."""
        param = ParameterInfo(
            name="user_id",
            type="str",
            default=None,
            description="User identifier",
            required=True,
            is_variadic=False,
            is_keyword=False,
        )
        assert param.name == "user_id"
        assert param.required is True
        assert param.is_variadic is False
    
    def test_parameter_info_defaults(self):
        """ParameterInfo should have correct defaults."""
        param = ParameterInfo(
            name="data",
        )
        assert param.type is None
        assert param.default is None
        assert param.description is None
        assert param.required is True
        assert param.is_variadic is False
        assert param.is_keyword is False
    
    def test_return_info(self):
        """ReturnInfo should validate correctly."""
        ret = ReturnInfo(
            type="User",
            description="The user object",
            is_async=True,
            is_generator=False,
        )
        assert ret.type == "User"
        assert ret.is_async is True
        assert ret.is_generator is False
    
    def test_return_info_defaults(self):
        """ReturnInfo should have correct defaults."""
        ret = ReturnInfo()
        assert ret.type is None
        assert ret.description is None
        assert ret.is_async is False
        assert ret.is_generator is False
    
    def test_example_info(self):
        """ExampleInfo should validate correctly."""
        example = ExampleInfo(
            title="Basic Usage",
            language="python",
            code="user = get_user(123)",
            description="Get a user by ID",
        )
        assert example.title == "Basic Usage"
        assert example.language == "python"
        assert example.code == "user = get_user(123)"
    
    def test_example_info_defaults(self):
        """ExampleInfo should have correct defaults."""
        example = ExampleInfo(
            title="Example",
            language="js",
            code="console.log('test')",
        )
        assert example.description is None
    
    def test_error_pattern_base(self):
        """ErrorPatternBase should validate correctly."""
        pattern = ErrorPatternBase(
            exception_type="ValueError",
            message_pattern="User not found",
            message_regex=r"User \d+ not found",
            condition="When user_id doesn't exist",
            source_line=42,
        )
        assert pattern.exception_type == "ValueError"
        assert pattern.message_pattern == "User not found"
        assert pattern.message_regex == r"User \d+ not found"
    
    def test_error_pattern_base_defaults(self):
        """ErrorPatternBase should have correct defaults."""
        pattern = ErrorPatternBase(
            exception_type="RuntimeError",
            message_pattern="Something went wrong",
        )
        assert pattern.message_regex is None
        assert pattern.condition is None
        assert pattern.source_line is None
    
    def test_error_pattern(self):
        """ErrorPattern should validate correctly."""
        now = datetime.utcnow()
        pattern = ErrorPattern(
            id="err:fastapi:ValueError:user-not-found",
            exception_type="ValueError",
            message_pattern="User not found",
            library_id="pypi:fastapi:0.115.6",
            api_element_id="fastapi.app:FastAPI.get_user",
            language=ProgrammingLanguage.PYTHON,
            message_embedding=[0.1, 0.2],
            linked_doc_chunk_ids=["chunk1", "chunk2"],
            linked_fix_ids=["fix1"],
            times_encountered=5,
            last_seen=now,
            created_at=now,
        )
        assert pattern.id == "err:fastapi:ValueError:user-not-found"
        assert pattern.times_encountered == 5
    
    def test_error_pattern_defaults(self):
        """ErrorPattern should have correct defaults."""
        pattern = ErrorPattern(
            id="err:test",
            exception_type="Error",
            message_pattern="test",
            library_id="pypi:test:1.0",
            language=ProgrammingLanguage.PYTHON,
        )
        assert pattern.api_element_id is None
        assert pattern.message_embedding is None
        assert pattern.linked_doc_chunk_ids == []
        assert pattern.linked_fix_ids == []
        assert pattern.times_encountered == 0
        assert pattern.last_seen is None
        assert isinstance(pattern.created_at, datetime)
    
    def test_api_element_base(self):
        """APIElementBase should validate correctly."""
        element = APIElementBase(
            name="get_user",
            element_type=APIElementType.FUNCTION,
            module="fastapi.app",
            parent="UserService",
            visibility="public",
            signature="def get_user(user_id: str) -> User:",
            docstring="Get a user by ID",
            deprecated=False,
            since_version="0.1.0",
        )
        assert element.name == "get_user"
        assert element.element_type == APIElementType.FUNCTION
        assert element.visibility == "public"
    
    def test_api_element_base_defaults(self):
        """APIElementBase should have correct defaults."""
        element = APIElementBase(
            name="test",
            element_type=APIElementType.METHOD,
        )
        assert element.module is None
        assert element.parent is None
        assert element.visibility == "public"
        assert element.signature is None
        assert element.docstring is None
        assert element.deprecated is False
        assert element.deprecation_message is None
        assert element.since_version is None
    
    def test_api_element_create(self):
        """APIElementCreate should validate correctly."""
        element = APIElementCreate(
            id="fastapi.app:FastAPI.get_user",
            name="get_user",
            element_type=APIElementType.FUNCTION,
            library_id="pypi:fastapi:0.115.6",
            module="fastapi.app",
            parameters=[
                ParameterInfo(name="user_id", type="str", required=True),
            ],
            returns=ReturnInfo(type="User", is_async=False),
            docstring_embedding=[0.1, 0.2],
            source_file="fastapi/app.py",
            start_line=10,
            end_line=20,
            source_hash="hash123",
        )
        assert element.id == "fastapi.app:FastAPI.get_user"
        assert len(element.parameters) == 1
        assert element.returns is not None
    
    def test_api_element_create_defaults(self):
        """APIElementCreate should have correct defaults."""
        element = APIElementCreate(
            id="test",
            name="test",
            element_type=APIElementType.FUNCTION,
            library_id="pypi:test:1.0",
        )
        assert element.module is None
        assert element.parent is None
        assert element.parameters == []
        assert element.returns is None
        assert element.type_parameters == []
        assert element.examples == []
        assert element.see_also == []
        assert element.errors_raised == []
        assert element.docstring_embedding is None
        assert element.source_file is None
        assert element.start_line is None
        assert element.end_line is None
        assert element.source_hash is None
    
    def test_api_element(self):
        """APIElement should validate correctly."""
        now = datetime.utcnow()
        element = APIElement(
            id="test",
            name="test",
            element_type=APIElementType.FUNCTION,
            library_id="pypi:test:1.0",
            created_at=now,
            updated_at=now,
        )
        assert element.id == "test"
        assert isinstance(element.created_at, datetime)
        assert isinstance(element.updated_at, datetime)
    
    def test_error_lookup_result(self):
        """ErrorLookupResult should validate correctly."""
        pattern = ErrorPattern(
            id="err:test",
            exception_type="Error",
            message_pattern="test",
            library_id="pypi:test:1.0",
            language=ProgrammingLanguage.PYTHON,
        )
        library = LibraryInfo(
            id="pypi:test:1.0",
            name="test",
            version="1.0",
            language=ProgrammingLanguage.PYTHON,
            ecosystem=PackageEcosystem.PYPI,
        )
        result = ErrorLookupResult(
            error_pattern=pattern,
            api_element=None,
            library=library,
            documentation_chunks=[],
            known_fixes=[],
            confidence=0.95,
        )
        assert result.error_pattern == pattern
        assert result.library == library
        assert result.confidence == 0.95
    
    def test_error_lookup_result_defaults(self):
        """ErrorLookupResult should have correct defaults."""
        pattern = ErrorPattern(
            id="err:test",
            exception_type="Error",
            message_pattern="test",
            library_id="pypi:test:1.0",
            language=ProgrammingLanguage.PYTHON,
        )
        result = ErrorLookupResult(
            error_pattern=pattern,
        )
        assert result.api_element is None
        assert result.library is None
        assert result.documentation_chunks == []
        assert result.known_fixes == []
        assert result.confidence == 0.0


class TestProjectDependenciesModels:
    """Tests for project dependencies models."""
    
    def test_dependency_info(self):
        """DependencyInfo should validate correctly."""
        dep = DependencyInfo(
            name="fastapi",
            version="0.115.6",
            version_constraint=">=0.100.0",
            language=ProgrammingLanguage.PYTHON,
            ecosystem=PackageEcosystem.PYPI,
            is_direct=True,
            is_dev=False,
            parent_dependency=None,
        )
        assert dep.name == "fastapi"
        assert dep.version == "0.115.6"
        assert dep.version_constraint == ">=0.100.0"
        assert dep.is_direct is True
    
    def test_dependency_info_defaults(self):
        """DependencyInfo should have correct defaults."""
        dep = DependencyInfo(
            name="pytest",
            version="8.0.0",
            language=ProgrammingLanguage.PYTHON,
            ecosystem=PackageEcosystem.PYPI,
        )
        assert dep.version_constraint is None
        assert dep.is_direct is True
        assert dep.is_dev is False
        assert dep.parent_dependency is None
    
    def test_project_dependencies(self):
        """ProjectDependencies should validate correctly."""
        deps = [
            DependencyInfo(
                name="fastapi",
                version="0.115.6",
                language=ProgrammingLanguage.PYTHON,
                ecosystem=PackageEcosystem.PYPI,
            ),
            DependencyInfo(
                name="react",
                version="18.0.0",
                language=ProgrammingLanguage.JAVASCRIPT,
                ecosystem=PackageEcosystem.NPM,
            ),
        ]
        project = ProjectDependencies(
            project_path="/path/to/project",
            languages_detected=[ProgrammingLanguage.PYTHON, ProgrammingLanguage.JAVASCRIPT],
            dependencies=deps,
            lockfile_sources=["requirements.txt", "package-lock.json"],
        )
        assert project.project_path == "/path/to/project"
        assert len(project.dependencies) == 2
        assert len(project.lockfile_sources) == 2
    
    def test_project_dependencies_defaults(self):
        """ProjectDependencies should have correct defaults."""
        project = ProjectDependencies(
            project_path="/path/to/project",
            languages_detected=[ProgrammingLanguage.PYTHON],
            dependencies=[],
        )
        assert project.lockfile_sources == []
        assert isinstance(project.extracted_at, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
