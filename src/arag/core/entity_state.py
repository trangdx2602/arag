"""Entity state tracking for thesis-level entity-aware A-RAG."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Relation:
    relation_type: str       # e.g., "works_at", "born_in", "mother_of"
    target_entity: str
    source_chunk_id: str


@dataclass
class EntityNode:
    name: str
    entity_type: str                              # PERSON, ORG, LOC, DATE, EVENT, OTHER
    attributes: Dict[str, str] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def add_attribute(self, key: str, value: str, source_chunk_id: str):
        """Additive merge — latest write wins per key."""
        self.attributes[key] = value
        if source_chunk_id not in self.source_chunks:
            self.source_chunks.append(source_chunk_id)

    def add_relation(self, relation_type: str, target: str, source_chunk_id: str):
        for r in self.relations:
            if r.relation_type == relation_type and r.target_entity == target:
                return  # already exists
        self.relations.append(Relation(relation_type, target, source_chunk_id))
        if source_chunk_id not in self.source_chunks:
            self.source_chunks.append(source_chunk_id)


class EntityState:
    """Incremental entity knowledge base built during agent retrieval."""

    def __init__(self):
        self.entities: Dict[str, EntityNode] = {}

    def add_entity(self, name: str, entity_type: str = "OTHER") -> EntityNode:
        key = name.lower().strip()
        if key not in self.entities:
            self.entities[key] = EntityNode(name=name, entity_type=entity_type)
        return self.entities[key]

    def add_attribute(self, entity_name: str, attr_key: str, attr_value: str, source_chunk_id: str):
        node = self.add_entity(entity_name)
        node.add_attribute(attr_key, attr_value, source_chunk_id)

    def add_relation(self, from_entity: str, relation_type: str, to_entity: str, source_chunk_id: str):
        node = self.add_entity(from_entity)
        node.add_relation(relation_type, to_entity, source_chunk_id)

    def get_node(self, entity_name: str) -> Optional[EntityNode]:
        return self.entities.get(entity_name.lower().strip())

    def merge(self, extraction: Dict[str, Any]):
        """Merge an extraction result dict into this state.

        Expected format:
        {
            "entities": [{"name": str, "type": str, "attributes": {k: v}, "chunk_id": str}],
            "relations": [{"from": str, "rel": str, "to": str, "chunk_id": str}]
        }
        """
        for ent in extraction.get("entities", []):
            name = ent.get("name", "").strip()
            if not name:
                continue
            chunk_id = str(ent.get("chunk_id", ""))
            node = self.add_entity(name, ent.get("type", "OTHER"))
            for k, v in ent.get("attributes", {}).items():
                node.add_attribute(k, str(v), chunk_id)
            if chunk_id and chunk_id not in node.source_chunks:
                node.source_chunks.append(chunk_id)

        for rel in extraction.get("relations", []):
            from_e = rel.get("from", "").strip()
            to_e = rel.get("to", "").strip()
            rel_type = rel.get("rel", "related_to")
            chunk_id = str(rel.get("chunk_id", ""))
            if from_e and to_e:
                self.add_relation(from_e, rel_type, to_e, chunk_id)

    def get_entity_summary(self, max_entities: int = 20) -> str:
        """Return compact text summary for injection into LLM context."""
        if not self.entities:
            return ""
        lines = ["Known entities:"]
        for i, node in enumerate(list(self.entities.values())[:max_entities]):
            attrs = ", ".join(f"{k}={v}" for k, v in list(node.attributes.items())[:5])
            rels = "; ".join(f"{r.relation_type}->{r.target_entity} (chunk {r.source_chunk_id})"
                             for r in node.relations[:3])
            line = f"  - {node.name} [{node.entity_type}]"
            if attrs:
                line += f": {attrs}"
            if rels:
                line += f"; relations: [{rels}]"
            lines.append(line)
        if len(self.entities) > max_entities:
            lines.append(f"  ... and {len(self.entities) - max_entities} more entities")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, node in self.entities.items():
            result[key] = {
                "name": node.name,
                "type": node.entity_type,
                "attributes": node.attributes,
                "source_chunks": node.source_chunks,
                "relations": [
                    {"rel": r.relation_type, "to": r.target_entity, "chunk": r.source_chunk_id}
                    for r in node.relations
                ]
            }
        return result

    def __len__(self) -> int:
        return len(self.entities)
