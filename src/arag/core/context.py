"""Agent execution context for ARAG."""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

# Lazy import to avoid circular deps and keep baseline untouched
_EntityState = None

def _get_entity_state_class():
    global _EntityState
    if _EntityState is None:
        from arag.core.entity_state import EntityState
        _EntityState = EntityState
    return _EntityState


@dataclass
class RetrievalLog:
    """Log entry for a retrieval operation."""
    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Context manager for agent execution state."""
    
    def __init__(self, enable_entity_tracking: bool = False):
        # Token statistics
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: List[RetrievalLog] = []

        # State management
        self.read_chunk_ids: Set[str] = set()
        self.search_history: List[Dict[str, Any]] = []

        # Thesis extensions (only active when enable_entity_tracking=True)
        self._enable_entity_tracking = enable_entity_tracking
        self.entity_state = _get_entity_state_class()() if enable_entity_tracking else None
        self.verification_result: Optional[Dict[str, Any]] = None
        self.verification_passed: bool = False
    
    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Dict[str, Any] = None
    ):
        """Add a retrieval log entry."""
        log = RetrievalLog(
            tool_name=tool_name,
            tokens=tokens,
            metadata=metadata or {}
        )
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens
    
    def mark_chunk_as_read(self, chunk_id: str):
        """Mark chunk as read."""
        self.read_chunk_ids.add(str(chunk_id))
    
    def is_chunk_read(self, chunk_id: str) -> bool:
        """Check if chunk has been read."""
        return str(chunk_id) in self.read_chunk_ids
    
    # Aliases for backward compatibility
    def add_read_chunk(self, chunk_id: str, content: str = None):
        """Alias for mark_chunk_as_read."""
        self.mark_chunk_as_read(chunk_id)
    
    def has_read_chunk(self, chunk_id: str) -> bool:
        """Alias for is_chunk_read."""
        return self.is_chunk_read(chunk_id)
    
    def get_read_chunk(self, chunk_id: str):
        """Check if chunk was read (returns None, content not stored)."""
        return None if not self.is_chunk_read(chunk_id) else ""
    
    def reset(self):
        """Reset context for new query."""
        self.retrieval_logs = []
        self.read_chunk_ids = set()
        self.search_history = []
        self.total_retrieved_tokens = 0
        # Reset thesis extensions if active
        if self._enable_entity_tracking and self.entity_state is not None:
            self.entity_state = _get_entity_state_class()()
        self.verification_result = None
        self.verification_passed = False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        summary = {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "retrieval_logs": [
                {
                    "tool_name": log.tool_name,
                    "tokens": log.tokens,
                    "metadata": log.metadata
                }
                for log in self.retrieval_logs
            ],
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": list(self.read_chunk_ids)
        }
        # Thesis extensions: only added if entity tracking is enabled
        if self._enable_entity_tracking and self.entity_state is not None:
            summary["entity_count"] = len(self.entity_state)
            summary["entity_names"] = [n.name for n in list(self.entity_state.entities.values())[:50]]
            summary["entity_state"] = self.entity_state.to_dict()
        if self.verification_result is not None:
            summary["verification"] = self.verification_result
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary."""
        return self.get_summary()
