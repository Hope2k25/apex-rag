"""
Chunker - Splits documents into semantic chunks for embedding.

Handles:
- Header-based chunking (preserving document structure)
- Token-based size limits
- Overlap for context continuity
- Table and list preservation

All chunking is deterministic - no LLM calls required.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import hashlib

from .document_parser import ParsedDocument, ParsedSection


@dataclass
class ChunkConfig:
    """Configuration for chunking."""
    max_tokens: int = 512           # Maximum tokens per chunk
    min_tokens: int = 50            # Minimum tokens (avoid tiny chunks)
    overlap_tokens: int = 50        # Overlap between chunks
    chars_per_token: float = 4.0    # Approximate chars per token
    preserve_tables: bool = True    # Keep tables together
    preserve_lists: bool = True     # Keep list items together
    include_header_in_chunk: bool = True  # Prepend header to each chunk


@dataclass
class Chunk:
    """A single chunk of content."""
    index: int
    content: str
    header_path: str
    source_file: str
    start_line: int
    end_line: int
    token_count: int
    content_hash: str
    metadata: dict = field(default_factory=dict)


class Chunker:
    """
    Document chunker for the Apex RAG system.

    Uses header-based chunking to preserve document structure,
    with fallback to token-based splitting for long sections.
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize the chunker with configuration."""
        self.config = config or ChunkConfig()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """
        Chunk a parsed document.

        Args:
            document: A ParsedDocument from the document parser

        Returns:
            List of Chunks ready for embedding
        """
        chunks = []
        chunk_index = 0

        # If document has sections, chunk by section
        if document.sections:
            for section in document.sections:
                section_chunks = self._chunk_section(
                    section,
                    document.source_file,
                    chunk_index,
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        else:
            # Fallback: chunk the entire content
            content_chunks = self._split_by_tokens(
                document.content,
                header_path="",
                source_file=document.source_file,
                start_index=0,
                start_line=0,
            )
            chunks.extend(content_chunks)

        return chunks

    def _chunk_section(
        self,
        section: ParsedSection,
        source_file: str,
        start_index: int,
    ) -> list[Chunk]:
        """Chunk a single section."""
        chunks = []

        # Build content with optional header
        if self.config.include_header_in_chunk and section.header:
            header_prefix = f"{'#' * section.header_level} {section.header}\n\n"
            content = header_prefix + section.content
        else:
            content = section.content

        # Check if section fits in one chunk
        token_count = self._estimate_tokens(content)

        if token_count <= self.config.max_tokens:
            # Section fits in one chunk
            if token_count >= self.config.min_tokens:
                chunks.append(Chunk(
                    index=start_index,
                    content=content.strip(),
                    header_path=section.header_path,
                    source_file=source_file,
                    start_line=section.start_line,
                    end_line=section.end_line,
                    token_count=token_count,
                    content_hash=self._hash(content),
                ))
        else:
            # Section too large - split it
            sub_chunks = self._split_by_tokens(
                content,
                header_path=section.header_path,
                source_file=source_file,
                start_index=start_index,
                start_line=section.start_line,
            )
            chunks.extend(sub_chunks)

        return chunks

    def _split_by_tokens(
        self,
        content: str,
        header_path: str,
        source_file: str,
        start_index: int,
        start_line: int,
    ) -> list[Chunk]:
        """Split content by token count with overlap."""
        chunks = []

        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(content)

        current_content = []
        current_tokens = 0
        chunk_index = start_index
        chunk_start_line = start_line

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            # If single paragraph is too large, split it further
            if para_tokens > self.config.max_tokens:
                # Flush current content first
                if current_content:
                    chunk_text = "\n\n".join(current_content)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        chunk_index,
                        header_path,
                        source_file,
                        chunk_start_line,
                    ))
                    chunk_index += 1
                    current_content = []
                    current_tokens = 0

                # Split the large paragraph
                sub_chunks = self._split_large_paragraph(
                    para,
                    header_path,
                    source_file,
                    chunk_index,
                    chunk_start_line,
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                continue

            # Check if adding this paragraph exceeds limit
            if current_tokens + para_tokens > self.config.max_tokens:
                # Save current chunk
                if current_content:
                    chunk_text = "\n\n".join(current_content)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        chunk_index,
                        header_path,
                        source_file,
                        chunk_start_line,
                    ))
                    chunk_index += 1

                    # Add overlap from previous chunk
                    overlap_content = self._get_overlap(current_content)
                    current_content = overlap_content
                    current_tokens = self._estimate_tokens("\n\n".join(overlap_content))
                    chunk_start_line = start_line  # Approximate

            current_content.append(para)
            current_tokens += para_tokens

        # Don't forget the last chunk
        if current_content:
            chunk_text = "\n\n".join(current_content)
            if self._estimate_tokens(chunk_text) >= self.config.min_tokens:
                chunks.append(self._create_chunk(
                    chunk_text,
                    chunk_index,
                    header_path,
                    source_file,
                    chunk_start_line,
                ))

        return chunks

    def _split_into_paragraphs(self, content: str) -> list[str]:
        """Split content into paragraphs, preserving tables and lists."""
        paragraphs = []
        current = []
        in_code_block = False
        in_table = False

        lines = content.split("\n")

        for line in lines:
            # Track code blocks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current.append(line)
                continue

            # Track tables
            if self.config.preserve_tables and "|" in line:
                in_table = True

            if in_code_block or in_table:
                current.append(line)
                # End of table (blank line after)
                if in_table and not line.strip():
                    paragraphs.append("\n".join(current))
                    current = []
                    in_table = False
                continue

            # Blank line = paragraph break
            if not line.strip():
                if current:
                    paragraphs.append("\n".join(current))
                    current = []
            else:
                current.append(line)

        # Don't forget the last paragraph
        if current:
            paragraphs.append("\n".join(current))

        return [p.strip() for p in paragraphs if p.strip()]

    def _split_large_paragraph(
        self,
        paragraph: str,
        header_path: str,
        source_file: str,
        start_index: int,
        start_line: int,
    ) -> list[Chunk]:
        """Split a paragraph that's too large for a single chunk."""
        chunks = []
        max_chars = int(self.config.max_tokens * self.config.chars_per_token)
        overlap_chars = int(self.config.overlap_tokens * self.config.chars_per_token)

        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        current_text = ""
        chunk_index = start_index

        for sentence in sentences:
            if len(current_text) + len(sentence) > max_chars:
                if current_text:
                    chunks.append(self._create_chunk(
                        current_text.strip(),
                        chunk_index,
                        header_path,
                        source_file,
                        start_line,
                    ))
                    chunk_index += 1

                    # Add overlap
                    overlap_start = max(0, len(current_text) - overlap_chars)
                    current_text = current_text[overlap_start:] + " " + sentence
                else:
                    # Single sentence is too long, just include it
                    current_text = sentence
            else:
                current_text += " " + sentence if current_text else sentence

        # Last chunk
        if current_text.strip():
            chunks.append(self._create_chunk(
                current_text.strip(),
                chunk_index,
                header_path,
                source_file,
                start_line,
            ))

        return chunks

    def _get_overlap(self, paragraphs: list[str]) -> list[str]:
        """Get the overlap content from previous chunk."""
        if not paragraphs:
            return []

        overlap_tokens = 0
        overlap_paras = []

        # Take from the end
        for para in reversed(paragraphs):
            para_tokens = self._estimate_tokens(para)
            if overlap_tokens + para_tokens <= self.config.overlap_tokens:
                overlap_paras.insert(0, para)
                overlap_tokens += para_tokens
            else:
                break

        return overlap_paras

    def _create_chunk(
        self,
        content: str,
        index: int,
        header_path: str,
        source_file: str,
        start_line: int,
    ) -> Chunk:
        """Create a Chunk object."""
        return Chunk(
            index=index,
            content=content,
            header_path=header_path,
            source_file=source_file,
            start_line=start_line,
            end_line=start_line + content.count("\n"),
            token_count=self._estimate_tokens(content),
            content_hash=self._hash(content),
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate)."""
        return int(len(text) / self.config.chars_per_token)

    def _hash(self, content: str) -> str:
        """Generate content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# Convenience function
def chunk_document(
    document: ParsedDocument,
    config: Optional[ChunkConfig] = None,
) -> list[Chunk]:
    """Chunk a parsed document."""
    chunker = Chunker(config)
    return chunker.chunk(document)
