"""LLM-based entity and relation extractor for thesis entity tracking."""

import json
from typing import Any, Dict

from arag.core.llm import LLMClient

_EXTRACTION_SYSTEM_PROMPT = """You are an entity extractor. Given a text passage, extract named entities and their relationships.

Output ONLY valid JSON with this exact structure:
{
  "entities": [
    {"name": "<entity name>", "type": "<PERSON|ORG|LOC|DATE|EVENT|OTHER>", "attributes": {"<key>": "<value>"}, "chunk_id": "<chunk_id>"}
  ],
  "relations": [
    {"from": "<entity name>", "rel": "<relation type>", "to": "<entity name>", "chunk_id": "<chunk_id>"}
  ]
}

Rules:
- Only extract entities explicitly mentioned in the text.
- Attributes must be specific facts (e.g., born_year, nationality, founded_year, location).
- Relations must be directional (e.g., born_in, works_at, created, mother_of, part_of).
- If nothing to extract, return {"entities": [], "relations": []}.
- Output ONLY the JSON object, no explanation."""

_EXTRACTION_USER_TEMPLATE = """Chunk ID: {chunk_id}
Text:
{chunk_text}

Extract entities and relations. Output only JSON."""


class EntityExtractor:
    """Extracts entities and relations from retrieved chunk text via LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self._cache: Dict[str, Dict[str, Any]] = {}

    def extract(self, chunk_text: str, chunk_id: str) -> Dict[str, Any]:
        """Extract entities/relations from a single chunk.

        Returns extraction dict with "entities" and "relations" keys.
        On any failure, returns empty extraction (never raises).
        """
        if not chunk_text or not chunk_text.strip():
            return {"entities": [], "relations": []}

        if chunk_id in self._cache:
            return self._cache[chunk_id]

        # Truncate to avoid very long chunks blowing up extraction cost
        text = chunk_text[:3000]

        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _EXTRACTION_USER_TEMPLATE.format(
                chunk_id=chunk_id, chunk_text=text
            )},
        ]

        try:
            response = self.llm.chat(
                messages=messages,
                tools=None,
                temperature=0.0,
                max_tokens=512,
            )
            content = response["message"].get("content", "")
            result = self._parse_json(content)
            # Stamp chunk_id onto every entity/relation that lacks one
            for ent in result.get("entities", []):
                if not ent.get("chunk_id"):
                    ent["chunk_id"] = chunk_id
            for rel in result.get("relations", []):
                if not rel.get("chunk_id"):
                    rel["chunk_id"] = chunk_id
            self._cache[chunk_id] = result
            return result
        except Exception:
            return {"entities": [], "relations": []}

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, stripping markdown fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            result = json.loads(text)
            if "entities" not in result:
                result["entities"] = []
            if "relations" not in result:
                result["relations"] = []
            return result
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}
