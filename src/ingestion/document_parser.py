"""
Document Parser - Deterministic document parsing for Apex RAG.

Handles:
- HTML (including SingleFile browser archives)
- Markdown (with YAML frontmatter)
- RST (reStructuredText)
- Plain text

PDF support is optional via MinerU (magic-pdf).

All parsing is deterministic - no LLM calls required.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import yaml

from bs4 import BeautifulSoup, Comment
import html2text


class DocumentType(str, Enum):
    """Supported document types."""
    HTML = "html"
    MARKDOWN = "markdown"
    RST = "rst"
    TEXT = "text"
    PDF = "pdf"


@dataclass
class ParsedSection:
    """A parsed section of a document."""
    header: str
    header_level: int
    content: str
    header_path: str  # Breadcrumb path like "Introduction > Getting Started > Installation"
    start_line: int
    end_line: int


@dataclass
class ParsedDocument:
    """Result of parsing a document."""
    source_file: str
    document_type: DocumentType
    title: Optional[str]
    content: str  # Full cleaned content
    sections: list[ParsedSection]
    frontmatter: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""
    parsed_at: datetime = field(default_factory=datetime.utcnow)


class DocumentParser:
    """
    Deterministic document parser.
    
    Converts various document formats to clean, structured Markdown
    suitable for chunking and embedding.
    """

    def __init__(self):
        """Initialize the parser."""
        self._html2text = html2text.HTML2Text()
        self._html2text.ignore_links = False
        self._html2text.ignore_images = False
        self._html2text.ignore_emphasis = False
        self._html2text.body_width = 0  # Don't wrap lines

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a document file.

        Args:
            file_path: Path to the document

        Returns:
            ParsedDocument with structured content
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        doc_type = self._detect_type(path, content)

        # Parse based on type
        if doc_type == DocumentType.HTML:
            return self._parse_html(str(path), content)
        elif doc_type == DocumentType.MARKDOWN:
            return self._parse_markdown(str(path), content)
        elif doc_type == DocumentType.RST:
            return self._parse_rst(str(path), content)
        elif doc_type == DocumentType.PDF:
            return self._parse_pdf(str(path))
        else:
            return self._parse_text(str(path), content)

    def _detect_type(self, path: Path, content: str) -> DocumentType:
        """Detect document type from extension and content."""
        ext = path.suffix.lower()

        if ext in (".html", ".htm"):
            return DocumentType.HTML
        elif ext in (".md", ".markdown"):
            return DocumentType.MARKDOWN
        elif ext in (".rst", ".rest"):
            return DocumentType.RST
        elif ext == ".pdf":
            return DocumentType.PDF
        elif ext == ".txt":
            return DocumentType.TEXT
        
        # Check content for HTML
        if content.strip().startswith("<!DOCTYPE") or content.strip().startswith("<html"):
            return DocumentType.HTML
        
        # Check for Markdown headers
        if re.match(r"^#+ ", content, re.MULTILINE):
            return DocumentType.MARKDOWN

        return DocumentType.TEXT

    # ========================================
    # HTML PARSING (including SingleFile)
    # ========================================

    def _parse_html(self, file_path: str, content: str) -> ParsedDocument:
        """
        Parse HTML document, including SingleFile browser archives.

        SingleFile archives are full HTML snapshots from browsers that include:
        - Embedded CSS/JS
        - Navigation elements
        - Ads and tracking scripts
        
        This parser cleans all of that out to extract just the main content.
        """
        soup = BeautifulSoup(content, "html.parser")

        # Check if this is a SingleFile archive
        is_singlefile = self._is_singlefile_archive(soup)

        # Remove unwanted elements
        self._clean_html(soup, aggressive=is_singlefile)

        # Extract title
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Find main content container
        main_content = self._find_main_content(soup)

        # Convert to Markdown
        markdown_content = self._html2text.handle(str(main_content))
        markdown_content = self._clean_markdown(markdown_content)

        # Extract sections
        sections = self._extract_sections_from_markdown(markdown_content)

        # Generate content hash
        content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]

        return ParsedDocument(
            source_file=file_path,
            document_type=DocumentType.HTML,
            title=title,
            content=markdown_content,
            sections=sections,
            metadata={
                "is_singlefile": is_singlefile,
                "original_title": title,
            },
            content_hash=content_hash,
        )

    def _is_singlefile_archive(self, soup: BeautifulSoup) -> bool:
        """Check if HTML is a SingleFile browser archive."""
        # SingleFile adds specific comments and attributes
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "SingleFile" in comment or "single-file" in comment.lower():
                return True

        # Check for SingleFile data attributes
        if soup.find(attrs={"data-single-file": True}):
            return True

        # Check for embedded resources (base64 data URIs are common in SingleFile)
        style_tags = soup.find_all("style")
        for style in style_tags:
            if style.string and "data:image" in style.string:
                return True

        return False

    def _clean_html(self, soup: BeautifulSoup, aggressive: bool = False):
        """
        Remove unwanted elements from HTML.

        Args:
            soup: BeautifulSoup object to clean in-place
            aggressive: If True, remove more elements (for SingleFile archives)
        """
        # Always remove these
        remove_tags = [
            "script", "style", "noscript", "iframe", "svg",
            "header", "footer", "nav", "aside",
            "form", "button", "input", "select", "textarea",
        ]

        # Additional tags to remove for SingleFile archives
        if aggressive:
            remove_tags.extend([
                "meta", "link", "base",
            ])

        for tag in remove_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove elements by class/id patterns
        unwanted_patterns = [
            r"nav", r"menu", r"sidebar", r"footer", r"header",
            r"comment", r"social", r"share", r"subscribe",
            r"ad", r"advertisement", r"sponsor",
            r"cookie", r"consent", r"popup", r"modal",
            r"related", r"recommended", r"trending",
        ]

        for pattern in unwanted_patterns:
            for element in soup.find_all(
                attrs={"class": re.compile(pattern, re.I)}
            ):
                element.decompose()
            for element in soup.find_all(
                attrs={"id": re.compile(pattern, re.I)}
            ):
                element.decompose()

        # Remove hidden elements
        for element in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            element.decompose()

        # Remove empty elements
        for element in soup.find_all():
            if not element.get_text(strip=True) and not element.find_all("img"):
                element.decompose()

    def _find_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Find the main content container in the HTML."""
        # Priority order for main content
        main_selectors = [
            ("main", {}),
            ("article", {}),
            ("div", {"class": re.compile(r"content|article|post|entry", re.I)}),
            ("div", {"id": re.compile(r"content|article|post|entry", re.I)}),
            ("div", {"role": "main"}),
        ]

        for tag, attrs in main_selectors:
            element = soup.find(tag, attrs)
            if element and len(element.get_text(strip=True)) > 200:
                return element

        # Fallback: find the div with the most text content
        divs = soup.find_all("div")
        if divs:
            divs_with_text = [
                (div, len(div.get_text(strip=True)))
                for div in divs
            ]
            divs_with_text.sort(key=lambda x: x[1], reverse=True)
            if divs_with_text:
                return divs_with_text[0][0]

        # Last resort: return body or entire soup
        return soup.find("body") or soup

    # ========================================
    # MARKDOWN PARSING
    # ========================================

    def _parse_markdown(self, file_path: str, content: str) -> ParsedDocument:
        """Parse Markdown document with YAML frontmatter."""
        frontmatter = {}
        markdown_content = content

        # Extract YAML frontmatter
        frontmatter_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n",
            content,
            re.DOTALL,
        )
        if frontmatter_match:
            try:
                frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
                markdown_content = content[frontmatter_match.end():]
            except yaml.YAMLError:
                pass

        # Clean the markdown
        markdown_content = self._clean_markdown(markdown_content)

        # Extract title
        title = frontmatter.get("title")
        if not title:
            title_match = re.match(r"^#\s+(.+)$", markdown_content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()

        # Extract sections
        sections = self._extract_sections_from_markdown(markdown_content)

        # Generate content hash
        content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]

        return ParsedDocument(
            source_file=file_path,
            document_type=DocumentType.MARKDOWN,
            title=title,
            content=markdown_content,
            sections=sections,
            frontmatter=frontmatter,
            content_hash=content_hash,
        )

    def _clean_markdown(self, content: str) -> str:
        """Clean up Markdown content."""
        # Remove excessive blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Remove trailing whitespace
        content = "\n".join(line.rstrip() for line in content.split("\n"))

        # Remove common artifacts from HTML conversion
        content = re.sub(r"\[Skip to .*?\]\(.*?\)", "", content)
        content = re.sub(r"\[Back to top\]\(.*?\)", "", content)

        return content.strip()

    def _extract_sections_from_markdown(self, content: str) -> list[ParsedSection]:
        """Extract sections from Markdown based on headers."""
        sections = []
        lines = content.split("\n")
        
        current_header = ""
        current_level = 0
        current_content_lines = []
        current_start = 0
        header_stack = []  # For building breadcrumb path

        for i, line in enumerate(lines):
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            
            if header_match:
                # Save previous section
                if current_header or current_content_lines:
                    sections.append(ParsedSection(
                        header=current_header,
                        header_level=current_level,
                        content="\n".join(current_content_lines).strip(),
                        header_path=" > ".join(h for h, _ in header_stack) if header_stack else current_header,
                        start_line=current_start,
                        end_line=i - 1,
                    ))

                # Parse new header
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                # Update header stack for breadcrumb
                while header_stack and header_stack[-1][1] >= level:
                    header_stack.pop()
                header_stack.append((header_text, level))

                current_header = header_text
                current_level = level
                current_content_lines = []
                current_start = i
            else:
                current_content_lines.append(line)

        # Don't forget the last section
        if current_header or current_content_lines:
            sections.append(ParsedSection(
                header=current_header,
                header_level=current_level,
                content="\n".join(current_content_lines).strip(),
                header_path=" > ".join(h for h, _ in header_stack) if header_stack else current_header,
                start_line=current_start,
                end_line=len(lines) - 1,
            ))

        return sections

    # ========================================
    # RST PARSING
    # ========================================

    def _parse_rst(self, file_path: str, content: str) -> ParsedDocument:
        """Parse reStructuredText document."""
        try:
            from docutils.core import publish_parts
            from docutils.parsers.rst import Parser

            # Convert RST to HTML
            parts = publish_parts(
                source=content,
                writer_name="html",
                settings_overrides={"report_level": 5},  # Suppress warnings
            )
            html_content = parts["html_body"]

            # Parse the HTML
            soup = BeautifulSoup(html_content, "html.parser")
            markdown_content = self._html2text.handle(str(soup))
            markdown_content = self._clean_markdown(markdown_content)

            # Extract title
            title = parts.get("title") or None

            # Extract sections
            sections = self._extract_sections_from_markdown(markdown_content)

            content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]

            return ParsedDocument(
                source_file=file_path,
                document_type=DocumentType.RST,
                title=title,
                content=markdown_content,
                sections=sections,
                content_hash=content_hash,
            )

        except ImportError:
            # Fallback: treat as plain text
            return self._parse_text(file_path, content)

    # ========================================
    # PDF PARSING (Optional - requires MinerU)
    # ========================================

    def _parse_pdf(self, file_path: str) -> ParsedDocument:
        """Parse PDF document using MinerU (if available)."""
        try:
            from magic_pdf.pipe.UNIPipe import UNIPipe
            from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

            # Use MinerU for layout-aware PDF parsing
            path = Path(file_path)
            reader = DiskReaderWriter(str(path.parent))
            
            pipe = UNIPipe(reader, path.name)
            pipe.pipe_parse()
            
            # Get Markdown output
            markdown_content = pipe.get_markdown()
            markdown_content = self._clean_markdown(markdown_content)

            # Extract sections
            sections = self._extract_sections_from_markdown(markdown_content)

            content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]

            return ParsedDocument(
                source_file=file_path,
                document_type=DocumentType.PDF,
                title=path.stem,
                content=markdown_content,
                sections=sections,
                metadata={"parser": "mineru"},
                content_hash=content_hash,
            )

        except ImportError:
            # MinerU not installed - return placeholder
            return ParsedDocument(
                source_file=file_path,
                document_type=DocumentType.PDF,
                title=Path(file_path).stem,
                content="[PDF parsing requires magic-pdf package. Install with: pip install 'apex-rag[pdf]']",
                sections=[],
                metadata={"error": "mineru_not_installed"},
                content_hash="",
            )

    # ========================================
    # PLAIN TEXT PARSING
    # ========================================

    def _parse_text(self, file_path: str, content: str) -> ParsedDocument:
        """Parse plain text document."""
        content = content.strip()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Try to detect title from first line
        lines = content.split("\n")
        title = lines[0].strip() if lines else None

        return ParsedDocument(
            source_file=file_path,
            document_type=DocumentType.TEXT,
            title=title,
            content=content,
            sections=[
                ParsedSection(
                    header="",
                    header_level=0,
                    content=content,
                    header_path="",
                    start_line=0,
                    end_line=len(lines) - 1,
                )
            ],
            content_hash=content_hash,
        )


# Convenience function
def parse_document(file_path: str | Path) -> ParsedDocument:
    """Parse a document file."""
    parser = DocumentParser()
    return parser.parse(file_path)
