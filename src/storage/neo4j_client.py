"""
Neo4j client for Apex RAG.

Handles all graph operations:
- DKB (Deterministic Knowledge Base) storage
- Code entity relationships
- Error → API documentation graph
- Personalized PageRank queries

Self-hosted - runs locally via Neo4j Community Edition (GPL-3.0).
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from .schemas import (
    DKBGraph,
    DKBNode,
    DKBLink,
    LibraryInfo,
    APIElement,
    ErrorPattern,
    ErrorLookupResult,
    SemanticChunk,
)


class Neo4jConfig:
    """Neo4j connection configuration."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Load configuration from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


class Neo4jClient:
    """
    Async Neo4j client for Apex RAG.

    Uses the official neo4j Python driver (Apache 2.0 license).
    Connects to Neo4j Community Edition running locally.
    """

    def __init__(self, config: Optional[Neo4jConfig] = None):
        """Initialize the client with configuration."""
        self.config = config or Neo4jConfig.from_env()
        self._driver: Optional[AsyncDriver] = None

    async def connect(self):
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
            )
            # Verify connectivity
            await self._driver.verify_connectivity()

    async def disconnect(self):
        """Close the connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()

    @asynccontextmanager
    async def session(self):
        """Get a session for running queries."""
        if not self._driver:
            await self.connect()
        async with self._driver.session(database=self.config.database) as session:
            yield session

    # ========================================
    # SCHEMA SETUP
    # ========================================

    async def setup_schema(self):
        """Create indexes and constraints for the graph schema."""
        async with self.session() as session:
            # Constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
                "CREATE CONSTRAINT module_id IF NOT EXISTS FOR (m:Module) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
                "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (m:Method) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT library_id IF NOT EXISTS FOR (l:Library) REQUIRE l.id IS UNIQUE",
                "CREATE CONSTRAINT api_element_id IF NOT EXISTS FOR (a:APIElement) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT error_id IF NOT EXISTS FOR (e:ErrorMessage) REQUIRE e.id IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception:
                    pass  # Constraint may already exist

            # Indexes for common queries
            indexes = [
                "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)",
                "CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)",
                "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
                "CREATE INDEX error_type IF NOT EXISTS FOR (e:ErrorMessage) ON (e.exception_type, e.language)",
                "CREATE INDEX library_name IF NOT EXISTS FOR (l:Library) ON (l.name, l.language)",
            ]

            for index in indexes:
                try:
                    await session.run(index)
                except Exception:
                    pass

    # ========================================
    # DKB GRAPH OPERATIONS
    # ========================================

    async def load_dkb_graph(self, graph: DKBGraph, project_id: str = "default"):
        """
        Load a DKB graph into Neo4j.

        Args:
            graph: The DKB graph structure (from JSON)
            project_id: Identifier for the project
        """
        async with self.session() as session:
            # Clear existing graph for this project
            await session.run(
                """
                MATCH (n {project_id: $project_id})
                DETACH DELETE n
                """,
                project_id=project_id,
            )

            # Create nodes
            for node in graph.nodes:
                label = node.type.value.capitalize()  # "function" -> "Function"
                await session.run(
                    f"""
                    CREATE (n:{label} {{
                        id: $id,
                        project_id: $project_id,
                        name: $name,
                        parent_class: $parent_class,
                        file_path: $file_path,
                        start_line: $start_line,
                        end_line: $end_line,
                        language: $language,
                        content_hash: $content_hash,
                        docstring_summary: $docstring_summary,
                        signature: $signature
                    }})
                    """,
                    id=node.id,
                    project_id=project_id,
                    name=node.name,
                    parent_class=node.parent_class,
                    file_path=node.file_path,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    language=node.language,
                    content_hash=node.content_hash,
                    docstring_summary=node.docstring_summary,
                    signature=node.signature,
                )

            # Create relationships
            for link in graph.links:
                await session.run(
                    f"""
                    MATCH (source {{id: $source, project_id: $project_id}})
                    MATCH (target {{id: $target, project_id: $project_id}})
                    CREATE (source)-[r:{link.relation} {{weight: $weight}}]->(target)
                    """,
                    source=link.source,
                    target=link.target,
                    project_id=project_id,
                    weight=link.weight,
                )

    async def get_entity_context(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> list[dict]:
        """
        Get the context (neighbors) of an entity.

        Args:
            entity_id: The entity to get context for
            depth: How many hops to traverse

        Returns:
            List of connected entities with relationships
        """
        async with self.session() as session:
            result = await session.run(
                f"""
                MATCH path = (start {{id: $entity_id}})-[*1..{depth}]-(connected)
                RETURN 
                    start.id as start_id,
                    start.name as start_name,
                    type(last(relationships(path))) as relationship,
                    connected.id as connected_id,
                    connected.name as connected_name,
                    labels(connected) as connected_labels,
                    length(path) as distance
                ORDER BY distance
                LIMIT 50
                """,
                entity_id=entity_id,
            )

            records = await result.data()
            return records

    async def find_callers(self, function_id: str) -> list[dict]:
        """Find all functions that call the given function."""
        async with self.session() as session:
            result = await session.run(
                """
                MATCH (caller)-[:CALLS]->(target {id: $function_id})
                RETURN 
                    caller.id as id,
                    caller.name as name,
                    caller.file_path as file_path,
                    labels(caller) as labels
                """,
                function_id=function_id,
            )
            return await result.data()

    async def find_callees(self, function_id: str) -> list[dict]:
        """Find all functions called by the given function."""
        async with self.session() as session:
            result = await session.run(
                """
                MATCH (source {id: $function_id})-[:CALLS]->(callee)
                RETURN 
                    callee.id as id,
                    callee.name as name,
                    callee.file_path as file_path,
                    labels(callee) as labels
                """,
                function_id=function_id,
            )
            return await result.data()

    # ========================================
    # ERROR → API DOCUMENTATION GRAPH
    # ========================================

    async def add_library(self, library: LibraryInfo):
        """Add or update a library node."""
        async with self.session() as session:
            await session.run(
                """
                MERGE (l:Library {id: $id})
                SET l.name = $name,
                    l.version = $version,
                    l.language = $language,
                    l.ecosystem = $ecosystem,
                    l.repository_url = $repository_url,
                    l.license = $license
                """,
                id=library.id,
                name=library.name,
                version=library.version,
                language=library.language.value,
                ecosystem=library.ecosystem.value,
                repository_url=library.repository_url,
                license=library.license,
            )

    async def add_api_element(self, element: APIElement):
        """Add or update an API element node."""
        async with self.session() as session:
            await session.run(
                """
                MERGE (a:APIElement {id: $id})
                SET a.name = $name,
                    a.element_type = $element_type,
                    a.module = $module,
                    a.signature = $signature,
                    a.docstring = $docstring
                WITH a
                MATCH (l:Library {id: $library_id})
                MERGE (l)-[:CONTAINS]->(a)
                """,
                id=element.id,
                name=element.name,
                element_type=element.element_type.value,
                module=element.module,
                signature=element.signature,
                docstring=element.docstring,
                library_id=element.library_id,
            )

    async def add_error_pattern(self, error: ErrorPattern):
        """Add or update an error pattern node."""
        async with self.session() as session:
            await session.run(
                """
                MERGE (e:ErrorMessage {id: $id})
                SET e.exception_type = $exception_type,
                    e.message_pattern = $message_pattern,
                    e.message_regex = $message_regex,
                    e.language = $language,
                    e.condition = $condition
                """,
                id=error.id,
                exception_type=error.exception_type,
                message_pattern=error.message_pattern,
                message_regex=error.message_regex,
                language=error.language.value,
                condition=error.condition,
            )

            # Link to API element if specified
            if error.api_element_id:
                await session.run(
                    """
                    MATCH (e:ErrorMessage {id: $error_id})
                    MATCH (a:APIElement {id: $api_id})
                    MERGE (a)-[:RAISES]->(e)
                    """,
                    error_id=error.id,
                    api_id=error.api_element_id,
                )

    async def lookup_error(
        self,
        error_message: str,
        error_type: str,
        language: str,
    ) -> list[dict]:
        """
        Look up error documentation by error message.

        Uses exact match first, then falls back to pattern matching.
        """
        async with self.session() as session:
            # Try exact match first
            result = await session.run(
                """
                MATCH (e:ErrorMessage)
                WHERE e.language = $language
                  AND e.exception_type = $error_type
                  AND (
                    e.message_pattern = $message
                    OR $message =~ e.message_regex
                  )
                OPTIONAL MATCH (a:APIElement)-[:RAISES]->(e)
                OPTIONAL MATCH (l:Library)-[:CONTAINS]->(a)
                RETURN 
                    e.id as error_id,
                    e.message_pattern as pattern,
                    e.condition as condition,
                    a.id as api_id,
                    a.name as api_name,
                    a.signature as api_signature,
                    a.docstring as api_docstring,
                    l.id as library_id,
                    l.name as library_name,
                    l.version as library_version
                ORDER BY 
                    CASE WHEN e.message_pattern = $message THEN 0 ELSE 1 END
                LIMIT 5
                """,
                message=error_message,
                error_type=error_type,
                language=language,
            )

            return await result.data()

    # ========================================
    # PERSONALIZED PAGERANK
    # ========================================

    async def personalized_pagerank(
        self,
        seed_nodes: list[str],
        damping: float = 0.85,
        iterations: int = 20,
        limit: int = 20,
    ) -> list[dict]:
        """
        Run Personalized PageRank from seed nodes.

        Note: Neo4j Community doesn't include GDS library.
        This is a simplified implementation using Cypher.
        For production, consider Neo4j Enterprise or implement PPR manually.

        Args:
            seed_nodes: List of node IDs to start from
            damping: PageRank damping factor
            iterations: Number of iterations
            limit: Maximum results to return

        Returns:
            List of nodes with their PageRank scores
        """
        async with self.session() as session:
            # Simplified: Just get neighbors with decreasing weight by distance
            # Real PPR would require GDS or custom implementation
            result = await session.run(
                """
                UNWIND $seeds as seed_id
                MATCH (seed {id: seed_id})
                MATCH path = (seed)-[*1..3]-(connected)
                WITH connected, min(length(path)) as distance
                RETURN 
                    connected.id as id,
                    connected.name as name,
                    labels(connected) as labels,
                    connected.file_path as file_path,
                    1.0 / (distance + 1) as score
                ORDER BY score DESC
                LIMIT $limit
                """,
                seeds=seed_nodes,
                limit=limit,
            )

            return await result.data()

    # ========================================
    # UTILITY METHODS
    # ========================================

    async def get_stats(self) -> dict:
        """Get graph statistics."""
        async with self.session() as session:
            result = await session.run(
                """
                MATCH (n)
                RETURN labels(n) as label, count(*) as count
                """
            )
            node_counts = {r["label"][0]: r["count"] for r in await result.data()}

            result = await session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                """
            )
            rel_counts = {r["type"]: r["count"] for r in await result.data()}

            return {
                "nodes": node_counts,
                "relationships": rel_counts,
            }

    async def clear_all(self, confirm: bool = False):
        """Clear all data from the graph. Requires confirmation."""
        if not confirm:
            raise ValueError("Must pass confirm=True to clear all data")

        async with self.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
