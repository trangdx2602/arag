"""Evidence Sufficiency Checker: verifies agent has enough evidence before answering."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.core.llm import LLMClient

if TYPE_CHECKING:
    from arag.core.context import AgentContext


@dataclass
class SufficiencyResult:
    is_sufficient: bool
    missing_gaps: List[str] = field(default_factory=list)
    coverage_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_sufficient": self.is_sufficient,
            "missing_gaps": self.missing_gaps,
            "coverage_score": self.coverage_score,
        }


_CHECKER_SYSTEM_PROMPT = """You are an evidence sufficiency checker for a multi-hop question answering system.

Given a question, a proposed answer, and a summary of retrieved evidence, determine whether the agent has gathered sufficient evidence to confidently answer the question.

Output ONLY valid JSON:
{
  "is_sufficient": true or false,
  "missing_gaps": ["description of missing evidence 1", "..."],
  "coverage_score": 0.0 to 1.0
}

Rules:
- is_sufficient = true if the retrieved evidence directly supports the proposed answer for all reasoning hops.
- is_sufficient = false if key facts are missing, unverified, or the reasoning chain has gaps.
- missing_gaps: list each specific piece of evidence still needed (empty list if sufficient).
- coverage_score: fraction of reasoning hops that are evidenced (0.0 = none, 1.0 = all).
- Output ONLY the JSON object, no explanation."""

_CHECKER_USER_TEMPLATE = """Question: {question}

Proposed Answer: {proposed_answer}

Retrieved Evidence Summary:
{evidence_summary}

Is the evidence sufficient to confidently answer the question? Output JSON."""


def _build_evidence_summary(context: "AgentContext", trajectory: List[Dict]) -> str:
    """Build a concise evidence summary from context and trajectory."""
    lines = []

    # Entity state summary if available
    if context.entity_state is not None and len(context.entity_state) > 0:
        lines.append("=== Known Entities ===")
        lines.append(context.entity_state.get_entity_summary(max_entities=10))

    # Retrieved chunks summary
    if context.read_chunk_ids:
        lines.append(f"\n=== Retrieved Chunks ({len(context.read_chunk_ids)} total) ===")
        lines.append(f"Chunk IDs: {', '.join(sorted(context.read_chunk_ids)[:20])}")

    # Tool call summary from trajectory
    if trajectory:
        lines.append(f"\n=== Tool Calls ({len(trajectory)} total) ===")
        for entry in trajectory[-8:]:  # last 8 tool calls
            tool = entry.get("tool_name", "?")
            args = entry.get("arguments", {})
            if tool == "keyword_search":
                lines.append(f"  keyword_search: {args.get('keywords', [])}")
            elif tool == "semantic_search":
                lines.append(f"  semantic_search: {args.get('query', '')[:80]}")
            elif tool == "read_chunk":
                lines.append(f"  read_chunk: {args.get('chunk_ids', [])}")

    return "\n".join(lines) if lines else "No evidence retrieved yet."


class EvidenceSufficiencyChecker:
    """Checks whether the agent has gathered sufficient evidence to answer."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def check(
        self,
        query: str,
        proposed_answer: str,
        context: "AgentContext",
        trajectory: List[Dict],
    ) -> Tuple[SufficiencyResult, float]:
        """Check if retrieved evidence is sufficient for the proposed answer.

        Returns (SufficiencyResult, api_cost_usd).
        Never raises — on any error returns sufficient=True to avoid blocking.
        """
        if not proposed_answer or not proposed_answer.strip():
            return SufficiencyResult(is_sufficient=True), 0.0

        if not context.read_chunk_ids and not trajectory:
            # No retrieval at all — flag as insufficient
            return SufficiencyResult(
                is_sufficient=False,
                missing_gaps=["No documents have been retrieved yet. Please search for relevant information."],
                coverage_score=0.0,
            ), 0.0

        evidence_summary = _build_evidence_summary(context, trajectory)

        messages = [
            {"role": "system", "content": _CHECKER_SYSTEM_PROMPT},
            {"role": "user", "content": _CHECKER_USER_TEMPLATE.format(
                question=query,
                proposed_answer=proposed_answer[:500],
                evidence_summary=evidence_summary[:2000],
            )},
        ]

        try:
            response = self.llm.chat(
                messages=messages,
                tools=None,
                temperature=0.0,
                max_tokens=256,
            )
            cost = response.get("cost", 0.0)
            content = response["message"].get("content", "")
            result = self._parse_result(content)
            return result, cost
        except Exception:
            # On any failure, allow the agent to proceed
            return SufficiencyResult(is_sufficient=True), 0.0

    def _parse_result(self, text: str) -> SufficiencyResult:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data = json.loads(text)
            return SufficiencyResult(
                is_sufficient=bool(data.get("is_sufficient", True)),
                missing_gaps=data.get("missing_gaps", []),
                coverage_score=float(data.get("coverage_score", 1.0)),
            )
        except (json.JSONDecodeError, ValueError):
            return SufficiencyResult(is_sufficient=True)
