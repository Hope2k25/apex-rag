"""
Code Indexer - Deterministic code parsing using Tree-sitter.

Handles:
- Parsing code files into AST
- Extracting code entities (classes, functions, methods)
- Building DKB (Deterministic Knowledge Base) graph
- Extracting call relationships

Supports multiple languages via Tree-sitter grammars.
All indexing is deterministic - no LLM calls required.
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..storage.schemas import DKBGraph, DKBNode, DKBLink, EntityType


class SupportedLanguage(str, Enum):
    """Languages supported by the code indexer."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"


# Language detection based on file extension
EXTENSION_TO_LANGUAGE = {
    ".py": SupportedLanguage.PYTHON,
    ".pyw": SupportedLanguage.PYTHON,
    ".js": SupportedLanguage.JAVASCRIPT,
    ".mjs": SupportedLanguage.JAVASCRIPT,
    ".jsx": SupportedLanguage.JAVASCRIPT,
    ".ts": SupportedLanguage.TYPESCRIPT,
    ".tsx": SupportedLanguage.TYPESCRIPT,
    ".go": SupportedLanguage.GO,
    ".rs": SupportedLanguage.RUST,
}


@dataclass
class CodeEntity:
    """A code entity extracted from source."""
    id: str
    entity_type: EntityType
    name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    content_hash: str = ""
    modifiers: list[str] = field(default_factory=list)


@dataclass
class CodeRelationship:
    """A relationship between code entities."""
    source: str
    target: str
    relation: str  # "CALLS", "IMPORTS", "INHERITS", "USES"
    weight: float = 1.0


@dataclass
class IndexedCodebase:
    """Result of indexing a codebase."""
    entities: list[CodeEntity]
    relationships: list[CodeRelationship]
    files_indexed: int
    errors: list[str]


class CodeIndexer:
    """
    Deterministic code indexer using Tree-sitter.

    Parses source code and extracts:
    - Classes, functions, methods
    - Import statements
    - Call relationships
    - Docstrings and signatures
    """

    def __init__(self):
        """Initialize the code indexer."""
        self._parsers = {}
        self._tree_sitter_available = self._check_tree_sitter()

    def _check_tree_sitter(self) -> bool:
        """Check if tree-sitter is available."""
        try:
            import tree_sitter
            return True
        except ImportError:
            return False

    def _get_parser(self, language: SupportedLanguage):
        """Get or create a parser for the given language."""
        if language in self._parsers:
            return self._parsers[language]

        if not self._tree_sitter_available:
            return None

        try:
            import tree_sitter
            
            # Try to get language module
            if language == SupportedLanguage.PYTHON:
                import tree_sitter_python as ts_python
                parser = tree_sitter.Parser(ts_python.language())
            elif language == SupportedLanguage.JAVASCRIPT:
                import tree_sitter_javascript as ts_js
                parser = tree_sitter.Parser(ts_js.language())
            elif language == SupportedLanguage.TYPESCRIPT:
                import tree_sitter_typescript as ts_ts
                parser = tree_sitter.Parser(ts_ts.language_typescript())
            elif language == SupportedLanguage.GO:
                import tree_sitter_go as ts_go
                parser = tree_sitter.Parser(ts_go.language())
            elif language == SupportedLanguage.RUST:
                import tree_sitter_rust as ts_rust
                parser = tree_sitter.Parser(ts_rust.language())
            else:
                return None

            self._parsers[language] = parser
            return parser

        except ImportError:
            return None

    def index_file(self, file_path: str | Path) -> tuple[list[CodeEntity], list[CodeRelationship]]:
        """
        Index a single source file.

        Args:
            file_path: Path to the source file

        Returns:
            Tuple of (entities, relationships)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        # Detect language
        language = EXTENSION_TO_LANGUAGE.get(path.suffix.lower())
        if not language:
            return [], []

        # Read file content
        content = path.read_text(encoding="utf-8", errors="replace")
        content_bytes = content.encode("utf-8")

        # Try tree-sitter parsing
        parser = self._get_parser(language)
        if parser:
            return self._parse_with_tree_sitter(
                parser, content_bytes, str(path), language
            )
        else:
            # Fallback to regex-based extraction
            return self._parse_with_regex(content, str(path), language)

    def _parse_with_tree_sitter(
        self,
        parser,
        content: bytes,
        file_path: str,
        language: SupportedLanguage,
    ) -> tuple[list[CodeEntity], list[CodeRelationship]]:
        """Parse using tree-sitter (high fidelity)."""
        tree = parser.parse(content)
        root = tree.root_node

        entities = []
        relationships = []

        if language == SupportedLanguage.PYTHON:
            entities, relationships = self._extract_python(root, content, file_path)
        elif language in (SupportedLanguage.JAVASCRIPT, SupportedLanguage.TYPESCRIPT):
            entities, relationships = self._extract_javascript(root, content, file_path, language)
        elif language == SupportedLanguage.GO:
            entities, relationships = self._extract_go(root, content, file_path)
        elif language == SupportedLanguage.RUST:
            entities, relationships = self._extract_rust(root, content, file_path)

        return entities, relationships

    def _extract_python(
        self,
        root,
        content: bytes,
        file_path: str,
    ) -> tuple[list[CodeEntity], list[CodeRelationship]]:
        """Extract Python entities."""
        entities = []
        relationships = []
        current_class = None

        def visit(node, class_name=None):
            nonlocal current_class

            if node.type == "class_definition":
                # Extract class
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name = self._get_text(content, name_node)
                    current_class = class_name

                    entity = CodeEntity(
                        id=f"{file_path}:{class_name}",
                        entity_type=EntityType.CLASS,
                        name=class_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="python",
                        signature=f"class {class_name}",
                        docstring=self._extract_python_docstring(node, content),
                        content_hash=self._hash_node(content, node),
                    )

                    # Check for inheritance
                    arguments = node.child_by_field_name("superclasses")
                    if arguments:
                        for arg in arguments.children:
                            if arg.type == "identifier":
                                base = self._get_text(content, arg)
                                relationships.append(CodeRelationship(
                                    source=entity.id,
                                    target=base,
                                    relation="INHERITS",
                                ))
                        entity.signature = f"class {class_name}({self._get_text(content, arguments)})"

                    entities.append(entity)

                    # Process class body
                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            visit(child, class_name)

                    current_class = None
                    return

            elif node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = self._get_text(content, name_node)

                    # Determine if method or function
                    if class_name:
                        entity_type = EntityType.METHOD
                        entity_id = f"{file_path}:{class_name}.{func_name}"
                    else:
                        entity_type = EntityType.FUNCTION
                        entity_id = f"{file_path}:{func_name}"

                    # Get signature
                    params = node.child_by_field_name("parameters")
                    sig = f"def {func_name}"
                    if params:
                        sig += self._get_text(content, params)

                    # Get return type
                    return_type = node.child_by_field_name("return_type")
                    if return_type:
                        sig += f" -> {self._get_text(content, return_type)}"

                    entity = CodeEntity(
                        id=entity_id,
                        entity_type=entity_type,
                        name=func_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="python",
                        signature=sig,
                        docstring=self._extract_python_docstring(node, content),
                        parent_class=class_name,
                        content_hash=self._hash_node(content, node),
                        modifiers=self._extract_python_decorators(node, content),
                    )
                    entities.append(entity)

                    # Extract calls from function body
                    body = node.child_by_field_name("body")
                    if body:
                        calls = self._extract_calls(body, content)
                        for call in calls:
                            relationships.append(CodeRelationship(
                                source=entity.id,
                                target=call,
                                relation="CALLS",
                            ))

                    return

            elif node.type == "import_statement" or node.type == "import_from_statement":
                # Track imports
                module_node = node.child_by_field_name("module")
                if module_node:
                    module = self._get_text(content, module_node)
                    relationships.append(CodeRelationship(
                        source=file_path,
                        target=module,
                        relation="IMPORTS",
                    ))

            # Recurse for other nodes
            for child in node.children:
                visit(child, class_name)

        visit(root)
        return entities, relationships

    def _extract_javascript(
        self,
        root,
        content: bytes,
        file_path: str,
        language: SupportedLanguage,
    ) -> tuple[list[CodeEntity], list[CodeRelationship]]:
        """Extract JavaScript/TypeScript entities."""
        entities = []
        relationships = []

        def visit(node, class_name=None):
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    cls_name = self._get_text(content, name_node)
                    entities.append(CodeEntity(
                        id=f"{file_path}:{cls_name}",
                        entity_type=EntityType.CLASS,
                        name=cls_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=language.value,
                        content_hash=self._hash_node(content, node),
                    ))

                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            visit(child, cls_name)
                    return

            elif node.type in ("function_declaration", "method_definition", "arrow_function"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = self._get_text(content, name_node)

                    if class_name:
                        entity_type = EntityType.METHOD
                        entity_id = f"{file_path}:{class_name}.{func_name}"
                    else:
                        entity_type = EntityType.FUNCTION
                        entity_id = f"{file_path}:{func_name}"

                    entities.append(CodeEntity(
                        id=entity_id,
                        entity_type=entity_type,
                        name=func_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language=language.value,
                        parent_class=class_name,
                        content_hash=self._hash_node(content, node),
                    ))
                    return

            for child in node.children:
                visit(child, class_name)

        visit(root)
        return entities, relationships

    def _extract_go(
        self,
        root,
        content: bytes,
        file_path: str,
    ) -> tuple[list[CodeEntity], list[CodeRelationship]]:
        """Extract Go entities."""
        entities = []
        relationships = []

        def visit(node):
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = self._get_text(content, name_node)
                    entities.append(CodeEntity(
                        id=f"{file_path}:{func_name}",
                        entity_type=EntityType.FUNCTION,
                        name=func_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="go",
                        content_hash=self._hash_node(content, node),
                    ))

            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                receiver = node.child_by_field_name("receiver")
                if name_node:
                    func_name = self._get_text(content, name_node)
                    receiver_type = None
                    if receiver:
                        receiver_type = self._get_text(content, receiver)
                    
                    entities.append(CodeEntity(
                        id=f"{file_path}:{receiver_type}.{func_name}" if receiver_type else f"{file_path}:{func_name}",
                        entity_type=EntityType.METHOD,
                        name=func_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="go",
                        parent_class=receiver_type,
                        content_hash=self._hash_node(content, node),
                    ))

            elif node.type == "type_declaration":
                for spec in node.children:
                    if spec.type == "type_spec":
                        name_node = spec.child_by_field_name("name")
                        if name_node:
                            type_name = self._get_text(content, name_node)
                            entities.append(CodeEntity(
                                id=f"{file_path}:{type_name}",
                                entity_type=EntityType.CLASS,
                                name=type_name,
                                file_path=file_path,
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                language="go",
                                content_hash=self._hash_node(content, node),
                            ))

            for child in node.children:
                visit(child)

        visit(root)
        return entities, relationships

    def _extract_rust(
        self,
        root,
        content: bytes,
        file_path: str,
    ) -> tuple[list[CodeEntity], list[CodeRelationship]]:
        """Extract Rust entities."""
        entities = []
        relationships = []

        def visit(node, impl_name=None):
            if node.type == "struct_item":
                name_node = node.child_by_field_name("name")
                if name_node:
                    struct_name = self._get_text(content, name_node)
                    entities.append(CodeEntity(
                        id=f"{file_path}:{struct_name}",
                        entity_type=EntityType.CLASS,
                        name=struct_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="rust",
                        content_hash=self._hash_node(content, node),
                    ))

            elif node.type == "function_item":
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = self._get_text(content, name_node)
                    if impl_name:
                        entity_id = f"{file_path}:{impl_name}.{func_name}"
                        entity_type = EntityType.METHOD
                    else:
                        entity_id = f"{file_path}:{func_name}"
                        entity_type = EntityType.FUNCTION

                    entities.append(CodeEntity(
                        id=entity_id,
                        entity_type=entity_type,
                        name=func_name,
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        language="rust",
                        parent_class=impl_name,
                        content_hash=self._hash_node(content, node),
                    ))

            elif node.type == "impl_item":
                type_node = node.child_by_field_name("type")
                if type_node:
                    type_name = self._get_text(content, type_node)
                    body = node.child_by_field_name("body")
                    if body:
                        for child in body.children:
                            visit(child, type_name)
                    return

            for child in node.children:
                visit(child, impl_name)

        visit(root)
        return entities, relationships

    def _parse_with_regex(
        self,
        content: str,
        file_path: str,
        language: SupportedLanguage,
    ) -> tuple[list[CodeEntity], list[CodeRelationship]]:
        """Fallback regex-based extraction when tree-sitter unavailable."""
        import re
        entities = []
        relationships = []
        lines = content.split("\n")

        if language == SupportedLanguage.PYTHON:
            # Basic Python patterns
            class_pattern = re.compile(r'^class\s+(\w+)(?:\(([^)]*)\))?:', re.MULTILINE)
            func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE)

            for match in class_pattern.finditer(content):
                class_name = match.group(1)
                line_num = content[:match.start()].count("\n") + 1
                entities.append(CodeEntity(
                    id=f"{file_path}:{class_name}",
                    entity_type=EntityType.CLASS,
                    name=class_name,
                    file_path=file_path,
                    start_line=line_num,
                    end_line=line_num,  # Approximate
                    language="python",
                    signature=f"class {class_name}",
                    content_hash=hashlib.sha256(match.group(0).encode()).hexdigest()[:16],
                ))

            for match in func_pattern.finditer(content):
                indent = len(match.group(1))
                func_name = match.group(2)
                line_num = content[:match.start()].count("\n") + 1

                entity_type = EntityType.METHOD if indent > 0 else EntityType.FUNCTION
                entities.append(CodeEntity(
                    id=f"{file_path}:{func_name}",
                    entity_type=entity_type,
                    name=func_name,
                    file_path=file_path,
                    start_line=line_num,
                    end_line=line_num,
                    language="python",
                    signature=f"def {func_name}({match.group(3)})",
                    content_hash=hashlib.sha256(match.group(0).encode()).hexdigest()[:16],
                ))

        return entities, relationships

    # Helper methods

    def _get_text(self, content: bytes, node) -> str:
        """Get text content of a node."""
        return content[node.start_byte:node.end_byte].decode("utf-8")

    def _hash_node(self, content: bytes, node) -> str:
        """Hash the content of a node."""
        text = content[node.start_byte:node.end_byte]
        return hashlib.sha256(text).hexdigest()[:16]

    def _extract_python_docstring(self, node, content: bytes) -> Optional[str]:
        """Extract docstring from Python function/class."""
        body = node.child_by_field_name("body")
        if body and body.child_count > 0:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0] if first_stmt.children else None
                if expr and expr.type == "string":
                    docstring = self._get_text(content, expr)
                    # Remove quotes
                    return docstring.strip('"""').strip("'''").strip('"\'')
        return None

    def _extract_python_decorators(self, node, content: bytes) -> list[str]:
        """Extract decorator names from Python function/class."""
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                dec_text = self._get_text(content, child)
                decorators.append(dec_text.strip())
        return decorators

    def _extract_calls(self, node, content: bytes) -> list[str]:
        """Extract function call names from a node."""
        calls = []

        def find_calls(n):
            if n.type == "call":
                func = n.child_by_field_name("function")
                if func:
                    calls.append(self._get_text(content, func))
            for child in n.children:
                find_calls(child)

        find_calls(node)
        return calls

    def index_directory(
        self,
        directory: str | Path,
        exclude_patterns: Optional[list[str]] = None,
    ) -> IndexedCodebase:
        """
        Index all code files in a directory.

        Args:
            directory: Root directory to index
            exclude_patterns: Glob patterns to exclude (e.g., ["**/test_*", "**/__pycache__"])

        Returns:
            IndexedCodebase with all entities and relationships
        """
        import fnmatch

        path = Path(directory)
        exclude_patterns = exclude_patterns or ["**/__pycache__/**", "**/node_modules/**", "**/.git/**"]

        all_entities = []
        all_relationships = []
        files_indexed = 0
        errors = []

        # Find all code files
        for ext in EXTENSION_TO_LANGUAGE.keys():
            for file_path in path.rglob(f"*{ext}"):
                # Check exclusions
                rel_path = str(file_path.relative_to(path))
                if any(fnmatch.fnmatch(rel_path, p) for p in exclude_patterns):
                    continue

                try:
                    entities, relationships = self.index_file(file_path)
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    files_indexed += 1
                except Exception as e:
                    errors.append(f"{file_path}: {str(e)}")

        return IndexedCodebase(
            entities=all_entities,
            relationships=all_relationships,
            files_indexed=files_indexed,
            errors=errors,
        )

    def to_dkb_graph(self, indexed: IndexedCodebase) -> DKBGraph:
        """Convert indexed codebase to DKB graph format."""
        nodes = [
            DKBNode(
                id=e.id,
                type=e.entity_type,
                name=e.name,
                parent_class=e.parent_class,
                file_path=e.file_path,
                start_line=e.start_line,
                end_line=e.end_line,
                language=e.language,
                content_hash=e.content_hash,
                docstring_summary=e.docstring[:100] if e.docstring else None,
                signature=e.signature,
                modifiers=e.modifiers,
            )
            for e in indexed.entities
        ]

        links = [
            DKBLink(
                source=r.source,
                target=r.target,
                relation=r.relation,
                weight=r.weight,
            )
            for r in indexed.relationships
        ]

        return DKBGraph(
            nodes=nodes,
            links=links,
            graph_metadata={
                "files_indexed": indexed.files_indexed,
                "entity_count": len(nodes),
                "relationship_count": len(links),
            },
        )

    def save_dkb_graph(self, graph: DKBGraph, output_path: str | Path):
        """Save DKB graph to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph.model_dump(), f, indent=2, default=str)


# Convenience function
def index_codebase(
    directory: str | Path,
    output_path: Optional[str | Path] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> DKBGraph:
    """
    Index a codebase and optionally save the DKB graph.

    Args:
        directory: Root directory to index
        output_path: Optional path to save the DKB graph JSON
        exclude_patterns: Glob patterns to exclude

    Returns:
        DKBGraph structure
    """
    indexer = CodeIndexer()
    indexed = indexer.index_directory(directory, exclude_patterns)
    graph = indexer.to_dkb_graph(indexed)

    if output_path:
        indexer.save_dkb_graph(graph, output_path)

    return graph
