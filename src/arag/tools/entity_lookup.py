"""Entity lookup tool: query the in-memory EntityState without new retrieval."""

from typing import Any, Dict, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class EntityLookupTool(BaseTool):
    """Read-only tool to query entities already known from retrieved chunks.

    Use this BEFORE searching to avoid redundant retrieval.
    """

    @property
    def name(self) -> str:
        return "entity_lookup"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Look up what is already known about an entity from previously retrieved documents. "
                    "Use this BEFORE keyword_search or semantic_search to avoid redundant retrieval. "
                    "Returns known attributes and relationships for the entity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "The name of the entity to look up (e.g., 'Albert Einstein', 'Princeton IAS').",
                        }
                    },
                    "required": ["entity_name"],
                },
            },
        }

    def execute(self, context: "AgentContext", **kwargs) -> Tuple[str, Dict[str, Any]]:
        entity_name = kwargs.get("entity_name", "").strip()

        if not entity_name:
            return "Error: entity_name is required.", {"retrieved_tokens": 0}

        if context.entity_state is None:
            return (
                "Entity tracking is not enabled in this run.",
                {"retrieved_tokens": 0},
            )

        node = context.entity_state.get_node(entity_name)
        if node is None:
            result = (
                f"Entity '{entity_name}' not yet in knowledge base. "
                "Use keyword_search or semantic_search to find information about it."
            )
        else:
            lines = [f"Known about '{node.name}' [{node.entity_type}]:"]
            if node.attributes:
                for k, v in node.attributes.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append("  (no attributes recorded yet)")
            if node.relations:
                lines.append("  Relations:")
                for r in node.relations:
                    lines.append(f"    - {r.relation_type} -> {r.target_entity} (source: chunk {r.source_chunk_id})")
            if node.source_chunks:
                lines.append(f"  Source chunks: {', '.join(node.source_chunks)}")
            result = "\n".join(lines)

        return result, {"retrieved_tokens": 0, "entity_name": entity_name}
