"""EntityAwareAgent: A-RAG with entity tracking and/or evidence verification.

Subclasses BaseAgent without modifying it. Adds two optional hooks:
  Hook 1 (entity tracking): After each read_chunk call, extract entities
          and merge into EntityState. Appends entity summary to tool result.
  Hook 2 (evidence verification): When agent wants to answer (no tool calls),
          run EvidenceSufficiencyChecker. If insufficient and loops remain,
          inject feedback and force one more loop.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from arag.agent.base import BaseAgent
from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class EntityAwareAgent(BaseAgent):
    """A-RAG variant with entity-aware context tracking and evidence verification.

    Args:
        llm_client: LLM client (same as BaseAgent).
        tools: ToolRegistry (same as BaseAgent).
        use_entity_tracking: If True, extract entities after each read_chunk.
        use_evidence_checking: If True, verify evidence before answering.
        system_prompt: Override system prompt. Defaults to entity_aware.txt.
        max_loops / max_token_budget / verbose: Same as BaseAgent.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        use_entity_tracking: bool = True,
        use_evidence_checking: bool = False,
        system_prompt: Optional[str] = None,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        verbose: bool = False,
    ):
        if system_prompt is None:
            if use_entity_tracking:
                system_prompt = _load_prompt("entity_aware.txt")
            else:
                system_prompt = _load_prompt("evidence_aware.txt")
        super().__init__(
            llm_client=llm_client,
            tools=tools,
            system_prompt=system_prompt,
            max_loops=max_loops,
            max_token_budget=max_token_budget,
            verbose=verbose,
        )
        self.use_entity_tracking = use_entity_tracking
        self.use_evidence_checking = use_evidence_checking

        # Lazy-init extractor and checker to avoid import overhead when not used
        self._extractor = None
        self._checker = None

    def _get_extractor(self):
        if self._extractor is None:
            from arag.verification.extractor import EntityExtractor
            self._extractor = EntityExtractor(self.llm)
        return self._extractor

    def _get_checker(self):
        if self._checker is None:
            from arag.verification.checker import EvidenceSufficiencyChecker
            self._checker = EvidenceSufficiencyChecker(self.llm)
        return self._checker

    def _apply_entity_hook(self, tool_result: str, chunk_ids: List[str],
                           context: AgentContext) -> str:
        """Hook 1: Extract entities from read_chunk result, update EntityState."""
        if not self.use_entity_tracking or context.entity_state is None:
            return tool_result

        extractor = self._get_extractor()
        # Extract per chunk_id; if multiple, use first as representative
        primary_chunk_id = chunk_ids[0] if chunk_ids else "unknown"
        extraction = extractor.extract(tool_result, primary_chunk_id)
        context.entity_state.merge(extraction)

        summary = context.entity_state.get_entity_summary(max_entities=15)
        if summary:
            return tool_result + f"\n\n[Entity Knowledge Base Updated]\n{summary}"
        return tool_result

    def _apply_verification_hook(
        self, query: str, proposed_answer: str, context: AgentContext,
        trajectory: List[Dict], loop_count: int, messages: List[Dict],
        total_cost: float,
    ):
        """Hook 2: Check evidence sufficiency before accepting the final answer.

        Returns (accept: bool, feedback_msg: str, total_cost: float).
        accept=True  → use proposed_answer as final answer.
        accept=False → inject feedback_msg and retry (one extra loop).
        """
        if not self.use_evidence_checking:
            return True, "", total_cost

        # Bypass gate near max_loops to avoid infinite loops
        if loop_count >= self.max_loops - 2:
            return True, "", total_cost

        # Already passed verification once — don't check again
        if context.verification_passed:
            return True, "", total_cost

        checker = self._get_checker()
        result, cost = checker.check(
            query=query,
            proposed_answer=proposed_answer,
            context=context,
            trajectory=trajectory,
        )
        total_cost += cost
        context.verification_result = result.to_dict()

        if result.is_sufficient:
            context.verification_passed = True
            return True, "", total_cost
        else:
            feedback = (
                "[Evidence Sufficiency Check]\n"
                f"Status: INCOMPLETE (coverage: {result.coverage_score:.0%})\n"
            )
            if result.missing_gaps:
                feedback += "Missing evidence:\n"
                for gap in result.missing_gaps:
                    feedback += f"  - {gap}\n"
            feedback += "Please retrieve the missing evidence before answering."
            return False, feedback, total_cost

    def run(self, query: str) -> Dict[str, Any]:
        """ReAct loop with optional entity tracking and evidence verification hooks."""
        context = AgentContext(enable_entity_tracking=self.use_entity_tracking, query=query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        trajectory = []
        total_cost = 0.0
        loop_count = 0
        tool_schemas = self.tools.get_all_schemas()
        retrieval_done = False  # True once any search/read tool has been called

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Question: {query}")
            mode = []
            if self.use_entity_tracking:
                mode.append("ET")
            if self.use_evidence_checking:
                mode.append("EV")
            print(f"Mode: {'+'.join(mode) or 'baseline-style'}")
            print(f"{'='*60}\n")

        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1

            current_tokens = self._calculate_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                if self.verbose:
                    print(f"Token budget exceeded ({current_tokens}), forcing answer...")
                final_answer, total_cost = self._force_final_answer(
                    messages, context, total_cost, "Token budget exceeded"
                )
                return {
                    "answer": final_answer,
                    "trajectory": trajectory,
                    "total_cost": total_cost,
                    "loops": loop_count,
                    "token_budget_exceeded": True,
                    **context.get_summary(),
                }

            if self.verbose:
                print(f"Loop {loop_count}/{self.max_loops} (Tokens: {current_tokens})")

            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas, tool_choice="auto")
            except Exception as e:
                if self.verbose:
                    print(f"LLM error: {e}")
                break

            total_cost += response["cost"]
            message = response["message"]
            messages.append(message)

            if self.verbose and message.get("content"):
                print(f"Assistant: {message['content'][:200]}...")

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                if not retrieval_done:
                    # Model answered without searching — force retrieval
                    messages.append({"role": "user", "content":
                        "You did not search the documents. You MUST call keyword_search or "
                        "semantic_search before answering. Search now."})
                    continue
                # Agent wants to answer — apply verification hook
                proposed_answer = message.get("content", "")
                accept, feedback, total_cost = self._apply_verification_hook(
                    query=query,
                    proposed_answer=proposed_answer,
                    context=context,
                    trajectory=trajectory,
                    loop_count=loop_count,
                    messages=messages,
                    total_cost=total_cost,
                )
                if accept:
                    return {
                        "answer": proposed_answer,
                        "trajectory": trajectory,
                        "total_cost": total_cost,
                        "loops": loop_count,
                        **context.get_summary(),
                    }
                else:
                    # Inject verification feedback and continue
                    if self.verbose:
                        print(f"[Verification] Insufficient — requesting more evidence")
                    messages.append({"role": "user", "content": feedback})
                    continue

            # Execute tool calls
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                if self.verbose:
                    print(f"Tool: {func_name}  Args: {func_args}")

                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                    if func_name in ("keyword_search", "semantic_search", "read_chunk"):
                        retrieval_done = True
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
                    tool_log = {"retrieved_tokens": 0, "error": str(e)}

                # Hook 1: entity extraction after read_chunk
                if func_name == "read_chunk" and self.use_entity_tracking:
                    chunk_ids = func_args.get("chunk_ids", [])
                    if isinstance(chunk_ids, str):
                        chunk_ids = [chunk_ids]
                    tool_result = self._apply_entity_hook(tool_result, chunk_ids, context)

                if self.verbose:
                    preview = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                    print(f"  Result: {preview}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

                traj_entry = {
                    "loop": loop_count,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log,
                }
                trajectory.append(traj_entry)

        # Max loops reached
        if self.verbose:
            print(f"Max loops reached ({self.max_loops}), forcing answer...")

        final_answer, total_cost = self._force_final_answer(
            messages, context, total_cost, "Maximum loops exceeded"
        )
        return {
            "answer": final_answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "loops": loop_count,
            "max_loops_exceeded": True,
            **context.get_summary(),
        }
