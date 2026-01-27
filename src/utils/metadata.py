"""
Metadata utility functions for Apex RAG.

Handles:
- Universal metadata generation
- Heuristic domain/language tagging
- Content hash generation
"""

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..storage.schemas import (
    UniversalMetadata,
    KnowledgeType,
    ProgrammingLanguage,
    PackageEcosystem,
)

# Extension to Language mapping
EXTENSION_MAP = {
    # Python
    ".py": ProgrammingLanguage.PYTHON,
    ".pyi": ProgrammingLanguage.PYTHON,
    ".ipynb": ProgrammingLanguage.PYTHON,
    # JavaScript / TypeScript
    ".js": ProgrammingLanguage.JAVASCRIPT,
    ".jsx": ProgrammingLanguage.JAVASCRIPT,
    ".mjs": ProgrammingLanguage.JAVASCRIPT,
    ".cjs": ProgrammingLanguage.JAVASCRIPT,
    ".ts": ProgrammingLanguage.TYPESCRIPT,
    ".tsx": ProgrammingLanguage.TYPESCRIPT,
    # Go
    ".go": ProgrammingLanguage.GO,
    # Rust
    ".rs": ProgrammingLanguage.RUST,
    # C#
    ".cs": ProgrammingLanguage.CSHARP,
    # Java
    ".java": ProgrammingLanguage.JAVA,
}

# Domain keywords for simple tagging primarily based on file path
DOMAIN_KEYWORDS = {
    "backend": ["api", "server", "controller", "service", "storage", "db", "database", "model", "schema"],
    "frontend": ["ui", "view", "component", "page", "public", "assets", "style", "css", "html"],
    "testing": ["test", "spec", "fixture", "mock", "__tests__"],
    "config": ["config", "setting", "env", "docker", "k8s", "terraform"],
    "documentation": ["doc", "guide", "readme", "changelog", "license"],
    "security": ["auth", "security", "crypt", "secret", "token"],
}


def calculate_content_hash(content: str) -> str:
    """
    Calculate SHA-256 hash of content.
    
    Args:
        content: The string content to hash.
        
    Returns:
        Hex digest of the hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def detect_language(file_path: str) -> Optional[str]:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Language string (e.g., 'python') or None if unknown.
    """
    ext = Path(file_path).suffix.lower()
    return EXTENSION_MAP.get(ext)


def detect_domains(file_path: str) -> List[str]:
    """
    Detect domains based on file path keywords.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        List of detected domains (e.g., ['backend', 'security']).
    """
    path_str = file_path.lower().replace("\\", "/")
    detected = []
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in path_str for kw in keywords):
            detected.append(domain)
            
    return detected


def extract_keywords_heuristic(content: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords using basic heuristics (CamelCase, snake_case, frequent terms).
    NOTE: This is a placeholder for more advanced NLP or just simple statistical counting.
    
    Args:
        content: Text content.
        max_keywords: Maximum number of keywords to return.
        
    Returns:
        List of keywords.
    """
    # Simple extraction of capitalized words or words with underscores (likely code identifiers)
    # This is very basic.
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_]{3,}\b", content)
    
    # Filter common stop words if needed, or just return unique ones
    # For now, just return unique identifiers found in the code/text
    unique_words = list(set(words))
    
    # Sort by length or just take the first few? 
    # Let's just take a sample of potentially "interesting" words.
    return unique_words[:max_keywords]


def infer_knowledge_type(file_path: str, content: str) -> KnowledgeType:
    """
    Infer the type of knowledge based on file path and content.
    
    Args:
        file_path: Path to the file.
        content: File content.
        
    Returns:
        KnowledgeType enum value.
    """
    path_lower = file_path.lower()
    
    if any(ext in path_lower for ext in EXTENSION_MAP.keys()):
        return KnowledgeType.SNIPPETS
        
    if "doc" in path_lower or ".md" in path_lower or ".txt" in path_lower:
        # Check if it looks like library documentation
        if "api" in content.lower() or "reference" in content.lower():
            return KnowledgeType.LIBRARY_DOCS
        return KnowledgeType.GUIDANCE
        
    if "error" in path_lower or "traceback" in path_lower:
        return KnowledgeType.ERROR_FIXES
        
    return KnowledgeType.GUIDANCE


def generate_universal_metadata(
    source_file: str,
    content: str,
    title: Optional[str] = None
) -> UniversalMetadata:
    """
    Generate UniversalMetadata for a given file.
    
    Args:
        source_file: Path to the source file.
        content: The text content of the file.
        title: Optional title.
        
    Returns:
        UniversalMetadata object.
    """
    content_hash = calculate_content_hash(content)
    language = detect_language(source_file)
    domains = detect_domains(source_file)
    knowledge_type = infer_knowledge_type(source_file, content)
    keywords = extract_keywords_heuristic(content)
    
    # Determine original format
    ext = Path(source_file).suffix.lower().lstrip(".")
    original_format = ext if ext else "text"
    
    lang_tags = [language] if language else []

    return UniversalMetadata(
        source_file=source_file,
        title=title or Path(source_file).name,
        knowledge_type=knowledge_type,
        domain_tags=domains,
        language_tags=lang_tags,
        keywords=keywords,
        original_format=original_format,
        content_hash=content_hash,
        extracted_at=datetime.utcnow()
    )
