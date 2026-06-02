"""Naive RAG agent: single-shot retrieval then answer (no iterative loop)."""

from typing import Any, Dict, Optional

from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry

_NAIVE_SYSTEM_PROMPT = """You are a question-answering assistant. Answer the question based ONLY on the provided context documents.
If the context does not contain enough information, say so. Keep your answer concise and accurate."""


class NaiveRAGAgent:
    """Non-iterative RAG baseline: retrieve top-k chunks, answer in one LLM call.

    This is the simplest RAG approach — no ReAct loop, no multi-hop reasoning.
    Used as the weakest baseline to measure improvement headroom.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.top_k = top_k
        self.system_prompt = system_prompt or _NAIVE_SYSTEM_PROMPT
        self.verbose = verbose

    def run(self, query: str) -> Dict[str, Any]:
        """Single-shot: semantic_search → optionally read_chunk → answer."""
        context = AgentContext()
        trajectory = []
        total_cost = 0.0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Naive RAG] Question: {query}")
            print(f"{'='*60}\n")

        # Step 1: Retrieve via semantic_search
        try:
            tool_result, tool_log = self.tools.execute(
                "semantic_search", context, query=query, top_k=self.top_k
            )
        except Exception:
            try:
                # Fallback to keyword_search if semantic not available
                keywords = query.split()[:3]
                tool_result, tool_log = self.tools.execute(
                    "keyword_search", context, keywords=keywords, top_k=self.top_k
                )
                trajectory.append({
                    "loop": 1, "tool_name": "keyword_search",
                    "arguments": {"keywords": keywords, "top_k": self.top_k},
                    "tool_result": tool_result, **tool_log,
                })
            except Exception as e:
                tool_result = f"Retrieval error: {e}"
                tool_log = {"retrieved_tokens": 0}

        if "semantic_search" not in str(trajectory):
            trajectory.append({
                "loop": 1, "tool_name": "semantic_search",
                "arguments": {"query": query, "top_k": self.top_k},
                "tool_result": tool_result, **tool_log,
            })

        if self.verbose:
            print(f"Retrieved: {tool_result[:300]}...")

        # Step 2: Generate answer from retrieved context
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{tool_result}\n\n"
                    f"Question: {query}\n\n"
                    "Answer based only on the context above:"
                )
            },
        ]

        try:
            response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
            total_cost += response["cost"]
            answer = response["message"].get("content", "")
        except Exception as e:
            answer = f"Error generating answer: {e}"

        if self.verbose:
            print(f"Answer: {answer[:200]}")

        return {
            "answer": answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "loops": 1,
            **context.get_summary(),
        }
