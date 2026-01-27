"""
Memory Tools for MCP Server.

Provides tools for agents to interact with Long-Term Memory (LTM).
"""

from typing import List, Optional
from uuid import UUID

from ..storage.postgres_client import PostgresClient
from ..storage.schemas import MemoryNote, MemoryNoteCreate, MemoryType, MemoryCheckpoint
from ..utils.embedding import EmbeddingModel


class MemoryTools:
    """
    Tool suite for memory operations.
    """

    def __init__(self, db_client: PostgresClient, embedding_model: Optional[EmbeddingModel] = None):
        self.db = db_client
        self.embed = embedding_model or EmbeddingModel()

    async def memory_add(
        self,
        content: str,
        memory_type: str = "episodic",
        context: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        agent_id: str = "default"
    ) -> str:
        """
        Store a new fact or experience in memory.
        
        Args:
            content: The content to remember.
            memory_type: One of 'episodic', 'semantic', 'procedural'.
            context: Context in which this memory was created.
            keywords: Tags for retrieval.
            agent_id: ID of the agent creating the memory.
            
        Returns:
            ID of the created memory.
        """
        # Validate memory_type
        try:
            m_type = MemoryType(memory_type.lower())
        except ValueError:
            return f"Error: Invalid memory_type '{memory_type}'. Must be: {', '.join([t.value for t in MemoryType])}"

        # Generate embedding
        embedding = self.embed.embed(content)

        note = MemoryNoteCreate(
            content=content,
            memory_type=m_type,
            context=context,
            keywords=keywords or [],
            agent_id=agent_id,
            embedding=embedding
        )
        
        result = await self.db.create_memory(note)
        return f"Memory created with ID: {result.id}"

    async def memory_retrieve(
        self,
        query: str,
        agent_id: str = "default",
        limit: int = 5
    ) -> str:
        """
        Retrieve relevant memories.
        
        Args:
            query: The search query.
            agent_id: The agent ID context.
            limit: Number of memories to return.
        """
        # Embed query
        query_vec = self.embed.embed(query)
        
        memories = await self.db.search_memories(
            query_embedding=query_vec,
            agent_id=agent_id,
            limit=limit,
            min_similarity=0.4 # Threshold
        )
        
        if not memories:
            return "No relevant memories found."
            
        result_text = f"Found {len(memories)} relevant memories:\n"
        for i, mem in enumerate(memories, 1):
            result_text += f"{i}. [{mem.memory_type}] {mem.content} (ID: {mem.id})\n"
            if mem.context:
                result_text += f"   Context: {mem.context}\n"
                
        return result_text

    async def memory_checkpoint(
        self,
        name: str,
        reason: str,
        agent_id: str = "default"
    ) -> str:
        """
        Create a snapshot of current memory state.
        
        Args:
            name: Name for the checkpoint.
            reason: Why this checkpoint is being created.
        """
        try:
            cp = await self.db.create_checkpoint(name, agent_id, reason)
            return f"Checkpoint '{cp.checkpoint_name}' created (ID: {cp.checkpoint_id})"
        except Exception as e:
            return f"Error creating checkpoint: {str(e)}"

    async def memory_rollback(
        self,
        checkpoint_id: str,
        agent_id: str = "default"
    ) -> str:
        """
        Restore memory state to a previous checkpoint.
        """
        try:
            uuid_id = UUID(checkpoint_id)
            success = await self.db.rollback_to_checkpoint(uuid_id, agent_id)
            if success:
                return f"Successfully rolled back to checkpoint {checkpoint_id}"
            else:
                return f"Checkpoint {checkpoint_id} not found."
        except ValueError:
            return "Invalid UUID format."
        except Exception as e:
            return f"Error rolling back: {str(e)}"

    async def memory_history(
        self,
        agent_id: str = "default",
        limit: int = 10
    ) -> str:
        """
        View recent checkpoints.
        """
        checkpoints = await self.db.list_checkpoints(agent_id, limit)
        if not checkpoints:
            return "No checkpoints found."
            
        report = "Recent Memory Checkpoints:\n"
        for cp in checkpoints:
            report += f"- {cp.created_at} | {cp.checkpoint_name} | {cp.reason} (ID: {cp.checkpoint_id})\n"
            
        return report
