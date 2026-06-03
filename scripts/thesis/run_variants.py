#!/usr/bin/env python3
"""Thesis experiment runner for all A-RAG variants.

Usage:
    python scripts/thesis/run_variants.py \\
        --config configs/thesis/musique_base.yaml \\
        --variant arag_baseline \\
        --questions data/musique/questions.json \\
        --output results/thesis/musique_base \\
        --limit 20 --workers 5

Variants:
    naive_rag               - Single-shot retrieval + answer
    arag_baseline           - Original A-RAG (BaseAgent, default.txt prompt)
    arag_entity_tracker     - A-RAG + EntityAwareAgent with entity tracking
    arag_evidence_checker   - A-RAG + EntityAwareAgent with evidence gate
    arag_entity_evidence_full - A-RAG + both modules
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Always run from repo root so relative paths in YAML configs resolve correctly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT / "src"))

from arag import LLMClient, ToolRegistry, Config, BaseAgent
from arag.agent.entity_agent import EntityAwareAgent
from arag.agent.naive_rag import NaiveRAGAgent
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.entity_lookup import EntityLookupTool
from arag.tools.check_evidence import CheckEvidenceTool
from arag.verification.checker import EvidenceSufficiencyChecker

logging.basicConfig(level=logging.ERROR)

VALID_VARIANTS = [
    "naive_rag",
    "arag_baseline",
    "arag_entity_tracker",
    "arag_evidence_checker",
    "arag_entity_evidence_full",
]


def _load_prompt(name: str) -> str:
    prompts_dir = Path(__file__).parent.parent.parent / "src/arag/agent/prompts"
    path = prompts_dir / name
    return path.read_text(encoding="utf-8") if path.exists() else "You are a helpful assistant."


def build_tools(config: Config, repo_root: Path,
                use_entity_lookup: bool = False,
                use_check_evidence: bool = False,
                checker: Optional[EvidenceSufficiencyChecker] = None) -> ToolRegistry:
    """Build shared ToolRegistry from config."""
    data_config = config.get("data", {})
    # Resolve relative paths from repo root (derived from --config absolute path)
    # so this works regardless of the process CWD
    chunks_file = str((repo_root / data_config.get("chunks_file", "data/chunks.json")).resolve())
    index_dir = str((repo_root / data_config.get("index_dir", "data/index")).resolve())

    tools = ToolRegistry()
    tools.register(KeywordSearchTool(chunks_file=chunks_file))
    tools.register(ReadChunkTool(chunks_file=chunks_file))

    index_file = Path(index_dir) / "sentence_index.pkl"
    if index_file.exists():
        emb_cfg = config.get("embedding", {})
        print(f"Loading embedding model: {emb_cfg.get('model', 'sentence-transformers/all-MiniLM-L6-v2')}")
        tools.register(SemanticSearchTool(
            chunks_file=chunks_file,
            index_dir=index_dir,
            model_name=emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=emb_cfg.get("device"),
        ))
        print("Embedding model loaded.")
    else:
        print(f"Warning: Index not found at {index_file}, semantic search disabled.")

    if use_entity_lookup:
        tools.register(EntityLookupTool())

    if use_check_evidence and checker is not None:
        ev_tool = CheckEvidenceTool(checker=checker)
        tools.register(ev_tool)

    return tools


def create_agent(variant: str, config: Config, tools: ToolRegistry,
                 checker: Optional[EvidenceSufficiencyChecker], verbose: bool = False):
    """Instantiate the correct agent for the given variant."""
    llm_cfg = config.get("llm", {})
    agent_cfg = config.get("agent", {})

    client = LLMClient(
        model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "gpt-4o-mini"),
        api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY"),
        base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1"),
        reasoning_effort=llm_cfg.get("reasoning_effort"),
    )
    max_loops = agent_cfg.get("max_loops", 15)
    max_token_budget = agent_cfg.get("max_token_budget", 128000)

    if variant == "naive_rag":
        return NaiveRAGAgent(llm_client=client, tools=tools, verbose=verbose)

    if variant == "arag_baseline":
        return BaseAgent(
            llm_client=client, tools=tools,
            system_prompt=_load_prompt("default.txt"),
            max_loops=max_loops, max_token_budget=max_token_budget, verbose=verbose,
        )

    if variant == "arag_entity_tracker":
        return EntityAwareAgent(
            llm_client=client, tools=tools,
            use_entity_tracking=True, use_evidence_checking=False,
            max_loops=max_loops, max_token_budget=max_token_budget, verbose=verbose,
        )

    if variant == "arag_evidence_checker":
        return EntityAwareAgent(
            llm_client=client, tools=tools,
            use_entity_tracking=False, use_evidence_checking=True,
            max_loops=max_loops, max_token_budget=max_token_budget, verbose=verbose,
        )

    if variant == "arag_entity_evidence_full":
        return EntityAwareAgent(
            llm_client=client, tools=tools,
            use_entity_tracking=True, use_evidence_checking=True,
            max_loops=max_loops, max_token_budget=max_token_budget, verbose=verbose,
        )

    raise ValueError(f"Unknown variant: {variant}. Valid: {VALID_VARIANTS}")


def process_one(item: Dict[str, Any], agent) -> Dict[str, Any]:
    qid = item.get("qid") or item.get("id")
    question = item.get("question", "")
    gold_answer = item.get("answer", item.get("gold_answer", ""))

    # Pass query to check_evidence tool if present
    for tool in getattr(getattr(agent, "tools", None), "_tools", {}).values():
        if hasattr(tool, "_query"):
            tool._query = question

    try:
        result = agent.run(question)
        return {
            "qid": qid,
            "question": question,
            "trajectory": result["trajectory"],
            "gold_answer": gold_answer,
            "pred_answer": result["answer"],
            "total_cost": result["total_cost"],
            "loops": result["loops"],
            "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
            "retrieval_logs": result.get("retrieval_logs", []),
            "chunks_read_count": result.get("chunks_read_count", 0),
            "chunks_read_ids": result.get("chunks_read_ids", []),
            "entity_count": result.get("entity_count", 0),
            "verification": result.get("verification"),
        }
    except Exception as e:
        return {
            "qid": qid, "question": question, "trajectory": [],
            "gold_answer": gold_answer, "pred_answer": f"Error: {e}",
            "total_cost": 0, "loops": 0,
            "total_retrieved_tokens": 0, "retrieval_logs": [],
            "chunks_read_count": 0, "chunks_read_ids": [],
            "entity_count": 0, "verification": None, "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Thesis Variant Runner")
    parser.add_argument("--config", "-c", required=True, help="YAML config file")
    parser.add_argument("--variant", "-V", required=True,
                        choices=VALID_VARIANTS, help="Agent variant")
    parser.add_argument("--questions", "-q", help="Override questions file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Max questions")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Concurrent workers")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    # Derive repo root from --config path (always absolute in notebook calls)
    # configs/thesis/musique_base.yaml -> configs/thesis -> configs -> repo_root
    repo_root = Path(args.config).resolve().parent.parent.parent

    data_cfg = config.get("data", {})
    raw_questions = data_cfg.get("questions_file", "data/questions.json")
    questions_file = args.questions or str((repo_root / raw_questions).resolve())

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / "predictions.jsonl"

    # Load completed checkpoints
    completed_qids: set = set()
    if predictions_file.exists():
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    qid = d.get("qid") or d.get("id")
                    if qid:
                        completed_qids.add(qid)
                except json.JSONDecodeError:
                    pass

    pending = [q for q in questions
               if (q.get("qid") or q.get("id")) not in completed_qids]

    print(f"Variant: {args.variant}")
    print(f"Total: {len(questions)} | Completed: {len(completed_qids)} | Pending: {len(pending)}")
    if not pending:
        print("All done!")
        return

    # Build shared LLM client for checker (separate from per-thread agent clients)
    llm_cfg = config.get("llm", {})
    shared_checker_llm = LLMClient(
        model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "gpt-4o-mini"),
        api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY"),
        base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1"),
    )
    checker = EvidenceSufficiencyChecker(shared_checker_llm)

    use_entity_lookup = args.variant in ("arag_entity_tracker", "arag_entity_evidence_full")
    use_check_evidence = args.variant in ("arag_evidence_checker", "arag_entity_evidence_full")

    shared_tools = build_tools(config, repo_root=repo_root,
                               use_entity_lookup=use_entity_lookup,
                               use_check_evidence=use_check_evidence, checker=checker)

    write_lock = Lock()

    def _run_one(item):
        agent = create_agent(args.variant, config, shared_tools, checker, verbose=args.verbose)
        result = process_one(item, agent)
        with write_lock:
            with open(predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    print(f"Running with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, q): q.get("qid") or q.get("id") for q in pending}
        with tqdm(total=len(pending), desc=args.variant) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error: {e}")
                pbar.update(1)

    print(f"Results saved to: {predictions_file}")


if __name__ == "__main__":
    main()
