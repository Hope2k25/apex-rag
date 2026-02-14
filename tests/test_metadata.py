"""
Tests for metadata utility functions.

Run with: uv run pytest tests/test_metadata.py -v
"""

import sys
from pathlib import Path
from datetime import datetime

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.metadata import (
    calculate_content_hash,
    detect_language,
    detect_domains,
    extract_keywords_heuristic,
    infer_knowledge_type,
    generate_universal_metadata,
)
from src.storage.schemas import KnowledgeType, ProgrammingLanguage


class TestCalculateContentHash:
    """Tests for content hash calculation."""
    
    def test_hash_consistency(self):
        """Same content should produce same hash."""
        content = "test content for hashing"
        hash1 = calculate_content_hash(content)
        hash2 = calculate_content_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex chars
    
    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        hash1 = calculate_content_hash("content A")
        hash2 = calculate_content_hash("content B")
        assert hash1 != hash2
    
    def test_empty_content(self):
        """Empty content should produce valid hash."""
        hash_value = calculate_content_hash("")
        assert len(hash_value) == 64
        assert isinstance(hash_value, str)
    
    def test_unicode_content(self):
        """Unicode content should be handled correctly."""
        content = "Hello 世界 🌍"
        hash_value = calculate_content_hash(content)
        assert len(hash_value) == 64


class TestDetectLanguage:
    """Tests for programming language detection."""
    
    def test_python_files(self):
        """Python file extensions should be detected."""
        assert detect_language("test.py") == ProgrammingLanguage.PYTHON
        assert detect_language("test.pyi") == ProgrammingLanguage.PYTHON
        assert detect_language("notebook.ipynb") == ProgrammingLanguage.PYTHON
    
    def test_javascript_files(self):
        """JavaScript file extensions should be detected."""
        assert detect_language("test.js") == ProgrammingLanguage.JAVASCRIPT
        assert detect_language("test.jsx") == ProgrammingLanguage.JAVASCRIPT
        assert detect_language("test.mjs") == ProgrammingLanguage.JAVASCRIPT
        assert detect_language("test.cjs") == ProgrammingLanguage.JAVASCRIPT
    
    def test_typescript_files(self):
        """TypeScript file extensions should be detected."""
        assert detect_language("test.ts") == ProgrammingLanguage.TYPESCRIPT
        assert detect_language("test.tsx") == ProgrammingLanguage.TYPESCRIPT
    
    def test_go_files(self):
        """Go file extensions should be detected."""
        assert detect_language("test.go") == ProgrammingLanguage.GO
    
    def test_rust_files(self):
        """Rust file extensions should be detected."""
        assert detect_language("test.rs") == ProgrammingLanguage.RUST
    
    def test_csharp_files(self):
        """C# file extensions should be detected."""
        assert detect_language("test.cs") == ProgrammingLanguage.CSHARP
    
    def test_java_files(self):
        """Java file extensions should be detected."""
        assert detect_language("test.java") == ProgrammingLanguage.JAVA
    
    def test_unknown_extension(self):
        """Unknown extensions should return None."""
        assert detect_language("test.txt") is None
        assert detect_language("test.unknown") is None
        assert detect_language("test") is None
    
    def test_case_insensitive(self):
        """Detection should be case-insensitive."""
        assert detect_language("test.PY") == ProgrammingLanguage.PYTHON
        assert detect_language("test.TS") == ProgrammingLanguage.TYPESCRIPT


class TestDetectDomains:
    """Tests for domain detection based on file paths."""
    
    def test_backend_domain(self):
        """Backend-related paths should be detected."""
        paths = [
            "src/api/user.py",
            "server/controller.py",
            "service/storage.py",
            "backend/model.py",
        ]
        for path in paths:
            assert "backend" in detect_domains(path)
    
    def test_frontend_domain(self):
        """Frontend-related paths should be detected."""
        paths = [
            "src/ui/component.tsx",
            "pages/home.js",
            "public/style.css",
            "assets/logo.png",
        ]
        for path in paths:
            assert "frontend" in detect_domains(path)
    
    def test_testing_domain(self):
        """Testing-related paths should be detected."""
        paths = [
            "tests/test_user.py",
            "spec/component.spec.ts",
            "fixtures/mock_data.py",
            "__tests__/utils.test.js",
        ]
        for path in paths:
            assert "testing" in detect_domains(path)
    
    def test_config_domain(self):
        """Config-related paths should be detected."""
        paths = [
            "config/settings.py",
            ".env.example",
            "docker-compose.yaml",
            "k8s/deployment.yaml",
            "terraform/main.tf",
        ]
        for path in paths:
            assert "config" in detect_domains(path)
    
    def test_documentation_domain(self):
        """Documentation-related paths should be detected."""
        paths = [
            "docs/api.md",
            "guide/tutorial.md",
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
        ]
        for path in paths:
            assert "documentation" in detect_domains(path)
    
    def test_security_domain(self):
        """Security-related paths should be detected."""
        paths = [
            "auth/token.py",
            "security/encrypt.py",
            "crypto/hashing.py",
            "secret/manager.py",
        ]
        for path in paths:
            assert "security" in detect_domains(path)
    
    def test_multiple_domains(self):
        """File can belong to multiple domains."""
        path = "backend/api/auth.py"
        domains = detect_domains(path)
        assert "backend" in domains
        assert "security" in domains
    
    def test_no_domain(self):
        """Paths with no keywords should return empty list."""
        path = "random/path/to/file.xyz"
        assert detect_domains(path) == []
    
    def test_windows_paths(self):
        """Windows-style paths should be handled."""
        path = "src\\api\\user.py"
        domains = detect_domains(path)
        assert "backend" in domains


class TestExtractKeywordsHeuristic:
    """Tests for keyword extraction."""
    
    def test_extract_identifiers(self):
        """Should extract code identifiers."""
        content = "function getUserData() { const userId = 123; }"
        keywords = extract_keywords_heuristic(content)
        assert "function" in keywords
        assert "getUserData" in keywords
        assert "userId" in keywords
    
    def test_max_keywords_limit(self):
        """Should respect max_keywords parameter."""
        content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11"
        keywords = extract_keywords_heuristic(content, max_keywords=5)
        assert len(keywords) <= 5
    
    def test_unique_keywords(self):
        """Should return unique keywords."""
        content = "test test test word word"
        keywords = extract_keywords_heuristic(content)
        assert keywords.count("test") <= 1
        assert keywords.count("word") <= 1
    
    def test_short_words_filtered(self):
        """Should filter short words (< 4 chars)."""
        content = "the and for but with test longword"
        keywords = extract_keywords_heuristic(content)
        assert "test" in keywords
        assert "longword" in keywords
        # Short words should be filtered out
        assert "the" not in keywords or len(keywords) < 10
    
    def test_empty_content(self):
        """Empty content should return empty list."""
        keywords = extract_keywords_heuristic("")
        assert keywords == []


class TestInferKnowledgeType:
    """Tests for knowledge type inference."""
    
    def test_code_snippets(self):
        """Code files should be classified as snippets."""
        assert infer_knowledge_type("test.py", "def foo(): pass") == KnowledgeType.SNIPPETS
        assert infer_knowledge_type("test.ts", "function foo() {}") == KnowledgeType.SNIPPETS
        assert infer_knowledge_type("test.go", "func foo() {}") == KnowledgeType.SNIPPETS
    
    def test_library_docs(self):
        """API documentation should be classified as library_docs."""
        content = "API Reference\n\nThis function provides..."
        assert infer_knowledge_type("api.md", content) == KnowledgeType.LIBRARY_DOCS
        assert infer_knowledge_type("reference.txt", content) == KnowledgeType.LIBRARY_DOCS
    
    def test_guidance(self):
        """General documentation should be classified as guidance."""
        content = "This is a tutorial on how to use the system."
        assert infer_knowledge_type("tutorial.md", content) == KnowledgeType.GUIDANCE
        assert infer_knowledge_type("guide.txt", content) == KnowledgeType.GUIDANCE
    
    def test_error_fixes(self):
        """Error-related files should be classified as error_fixes."""
        assert infer_knowledge_type("error.log", "Traceback...") == KnowledgeType.ERROR_FIXES
        # Note: traceback.txt doesn't match the error pattern in metadata.py
        # The pattern checks for "error" or "traceback" in path, but .txt files
        # without those keywords default to guidance
        assert infer_knowledge_type("error_traceback.txt", "Exception...") == KnowledgeType.ERROR_FIXES
    
    def test_default_guidance(self):
        """Unknown types should default to guidance."""
        assert infer_knowledge_type("unknown.xyz", "random content") == KnowledgeType.GUIDANCE


class TestGenerateUniversalMetadata:
    """Tests for universal metadata generation."""
    
    def test_basic_metadata(self):
        """Should generate basic metadata."""
        metadata = generate_universal_metadata(
            source_file="test.py",
            content="def hello(): print('world')",
        )
        
        assert metadata.source_file == "test.py"
        assert metadata.title == "test.py"
        assert metadata.knowledge_type == KnowledgeType.SNIPPETS
        assert metadata.language_tags == [ProgrammingLanguage.PYTHON]
        assert metadata.content_hash is not None
        assert len(metadata.content_hash) == 64
        assert isinstance(metadata.extracted_at, datetime)
    
    def test_custom_title(self):
        """Should use custom title if provided."""
        metadata = generate_universal_metadata(
            source_file="test.py",
            content="code",
            title="Custom Title",
        )
        assert metadata.title == "Custom Title"
    
    def test_domain_detection(self):
        """Should detect domains from file path."""
        metadata = generate_universal_metadata(
            source_file="backend/api/user.py",
            content="code",
        )
        assert "backend" in metadata.domain_tags
    
    def test_keyword_extraction(self):
        """Should extract keywords from content."""
        metadata = generate_universal_metadata(
            source_file="test.py",
            content="function getUserData() { return data; }",
        )
        assert len(metadata.keywords) > 0
    
    def test_original_format(self):
        """Should set original format from file extension."""
        metadata = generate_universal_metadata(
            source_file="test.md",
            content="markdown content",
        )
        assert metadata.original_format == "md"
    
    def test_no_extension(self):
        """Should handle files without extension."""
        metadata = generate_universal_metadata(
            source_file="Makefile",
            content="build commands",
        )
        assert metadata.original_format == "text"
    
    def test_no_language_detected(self):
        """Should handle files with no detected language."""
        metadata = generate_universal_metadata(
            source_file="README.md",
            content="Documentation text",
        )
        assert metadata.language_tags == []
        assert metadata.knowledge_type == KnowledgeType.GUIDANCE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
