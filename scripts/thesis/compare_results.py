#!/usr/bin/env python3
"""Compare results across thesis variants and print a summary table.

Usage:
    python scripts/thesis/compare_results.py \\
        --results results/thesis/ \\
        --variants arag_baseline arag_entity_tracker arag_evidence_checker arag_entity_evidence_full \\
        --dataset musique

Output: console table + results/thesis/comparison_<dataset>.json
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_answer(text: str) -> str:
    """Simple normalization for contain-match."""
    import re
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contain_match(pred: str, gold: str) -> bool:
    return normalize_answer(gold) in normalize_answer(pred)


def load_predictions(pred_file: Path) -> List[Dict[str, Any]]:
    records = []
    if not pred_file.exists():
        return records
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}

    n = len(records)
    answered = [r for r in records if r.get("pred_answer", "").strip()]
    contain_correct = sum(
        1 for r in answered
        if contain_match(r.get("pred_answer", ""), r.get("gold_answer", ""))
    )

    total_cost = sum(r.get("total_cost", 0) for r in records)
    total_loops = sum(r.get("loops", 0) for r in records)
    total_tokens = sum(r.get("total_retrieved_tokens", 0) for r in records)
    entity_counts = [r.get("entity_count", 0) for r in records]
    coverage_scores = [
        r["verification"]["coverage_score"]
        for r in records
        if r.get("verification") and r["verification"] is not None
    ]

    return {
        "n": n,
        "answer_rate": len(answered) / n,
        "contain_acc": contain_correct / n,
        "avg_loops": total_loops / n,
        "avg_retrieved_tokens": total_tokens / n,
        "avg_cost_usd": total_cost / n,
        "total_cost_usd": total_cost,
        "avg_entity_count": sum(entity_counts) / n if entity_counts else 0,
        "avg_coverage_score": (
            sum(coverage_scores) / len(coverage_scores) if coverage_scores else None
        ),
    }


def fmt(v, fmt_str=".3f") -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return format(v, fmt_str)
    return str(v)


def print_table(rows: List[Dict], variants: List[str]):
    headers = [
        "Variant", "N", "Contain-Acc", "Avg Loops",
        "Avg Tokens", "Avg Cost($)", "Avg Entities", "Avg Coverage",
    ]
    col_w = [28, 6, 12, 11, 12, 12, 14, 14]

    def row_str(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    sep = "-" * (sum(col_w) + 2 * (len(col_w) - 1))
    print(sep)
    print(row_str(headers))
    print(sep)
    for v, stats in zip(variants, rows):
        if not stats:
            print(row_str([v, "—", "—", "—", "—", "—", "—", "—"]))
            continue
        print(row_str([
            v,
            stats["n"],
            f"{stats['contain_acc']:.1%}",
            fmt(stats["avg_loops"]),
            fmt(stats["avg_retrieved_tokens"], ".0f"),
            fmt(stats["avg_cost_usd"], ".5f"),
            fmt(stats["avg_entity_count"], ".1f"),
            fmt(stats["avg_coverage_score"], ".1%") if stats.get("avg_coverage_score") else "N/A",
        ]))
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Compare thesis variant results")
    parser.add_argument("--results", "-r", default="results/thesis/", help="Results root dir")
    parser.add_argument("--variants", "-V", nargs="+",
                        default=["naive_rag", "arag_baseline", "arag_entity_tracker",
                                 "arag_evidence_checker", "arag_entity_evidence_full"])
    parser.add_argument("--dataset", "-d", default="musique",
                        choices=["musique", "hotpotqa"], help="Dataset name")
    parser.add_argument("--output-json", help="Save comparison JSON to this file")
    args = parser.parse_args()

    results_root = Path(args.results)
    all_stats = {}

    print(f"\n=== Thesis Results Comparison — {args.dataset.upper()} ===\n")

    rows = []
    for variant in args.variants:
        pred_file = results_root / f"{variant}_{args.dataset}" / "predictions.jsonl"
        if not pred_file.exists():
            # Try alternate naming
            pred_file = results_root / f"{args.dataset}_{variant}" / "predictions.jsonl"
        records = load_predictions(pred_file)
        if records:
            stats = compute_stats(records)
            all_stats[variant] = stats
            rows.append(stats)
            print(f"  {variant}: {len(records)} predictions loaded from {pred_file}")
        else:
            all_stats[variant] = None
            rows.append(None)
            print(f"  {variant}: no results found at {pred_file}")

    print()
    print_table(rows, args.variants)

    out_json = args.output_json or str(results_root / f"comparison_{args.dataset}.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to: {out_json}")


if __name__ == "__main__":
    main()
