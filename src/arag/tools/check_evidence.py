"""Check evidence sufficiency tool: agent calls this before giving a final answer."""

from typing import Any, Dict, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class CheckEvidenceTool(BaseTool):
    """Tool for the agent to explicitly verify evidence sufficiency.

    The agent calls this when it thinks it has enough information,
    passing its proposed reasoning chain for external verification.
    """

    def __init__(self, checker=None):
        """
        Args:
            checker: EvidenceSufficiencyChecker instance (injected at runtime).
                     If None, the tool returns a pass-through result.
        """
        self._checker = checker
        self._query: str = ""  # Set per-run by EntityAwareAgent

    @property
    def name(self) -> str:
        return "check_evidence"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Verify that you have sufficient evidence to answer the question. "
                    "Call this BEFORE providing your final answer to confirm all reasoning hops "
                    "are supported by retrieved chunks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "proposed_answer": {
                            "type": "string",
                            "description": "Your tentative answer to the question.",
                        },
                        "reasoning_summary": {
                            "type": "string",
                            "description": (
                                "Step-by-step reasoning: e.g. "
                                "'Hop 1: X is Y (chunk 5). Hop 2: Y was born in Z (chunk 12).'"
                            ),
                        },
                    },
                    "required": ["proposed_answer"],
                },
            },
        }

    def execute(self, context: "AgentContext", **kwargs) -> Tuple[str, Dict[str, Any]]:
        proposed_answer = kwargs.get("proposed_answer", "")
        reasoning_summary = kwargs.get("reasoning_summary", "")

        if self._checker is None or context.read_chunk_ids is None:
            return "Evidence check passed (checker not configured).", {"retrieved_tokens": 0}

        trajectory_summary = []
        if reasoning_summary:
            trajectory_summary.append({"tool_name": "reasoning", "arguments": {"text": reasoning_summary}})

        result, _ = self._checker.check(
            query=self._query,
            proposed_answer=proposed_answer,
            context=context,
            trajectory=trajectory_summary,
        )

        context.verification_result = result.to_dict()

        if result.is_sufficient:
            output = (
                f"[Evidence Check: SUFFICIENT] Coverage: {result.coverage_score:.0%}\n"
                "Your evidence is sufficient. You may now provide your final answer."
            )
        else:
            gaps = "\n".join(f"  - {g}" for g in result.missing_gaps)
            output = (
                f"[Evidence Check: INCOMPLETE] Coverage: {result.coverage_score:.0%}\n"
                f"Missing evidence:\n{gaps}\n"
                "Please retrieve the missing information before answering."
            )

        return output, {"retrieved_tokens": 0, "coverage_score": result.coverage_score}
