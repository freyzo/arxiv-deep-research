"""
ArXiv MCP Server — Evaluation Benchmark
========================================

A lightweight benchmark harness measuring retrieval quality of the
search_papers tool against a ground-truth query dataset.

Metrics:
  - Precision@K  : fraction of top-K results that are relevant
  - Recall@K     : fraction of known relevant papers found in top-K
  - MRR          : Mean Reciprocal Rank of first relevant result
  - Latency      : per-query wall-clock time

Design mirrors the AgentInstruct / AutoGenBench evaluation philosophy:
benchmark datasets are structured, reproducible, and isolatable.

Usage:
    python eval/benchmark.py
    python eval/benchmark.py --k 5 --output results.json
"""

import asyncio
import json
import time
import argparse
import sys
import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Ground-truth query dataset
# Each entry: query params → known relevant paper IDs (arXiv short IDs)
# Seeded from real papers; can be extended with generate_eval_tasks.py
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES: list[dict] = [
    {
        "id": "q001",
        "description": "Multi-agent LLM orchestration frameworks",
        "query": "multi-agent LLM orchestration",
        "categories": ["cs.AI", "cs.MA"],
        "relevant_ids": ["2308.08155", "2403.16971"],  # AutoGen, Magentic-One papers
        "tags": ["agentic-ai", "orchestration"],
    },
    {
        "id": "q002",
        "description": "Human-in-the-loop agent systems",
        "query": "human-in-the-loop agentic AI co-planning",
        "categories": ["cs.AI", "cs.HC"],
        "relevant_ids": ["2507.22358"],  # Magentic-UI paper
        "tags": ["human-ai-collaboration", "agentic-ai"],
    },
    {
        "id": "q003",
        "description": "Synthetic data generation for LLM fine-tuning",
        "query": "synthetic data generation agentic flows fine-tuning",
        "categories": ["cs.AI", "cs.LG"],
        "relevant_ids": ["2407.10505"],  # AgentInstruct paper
        "tags": ["synthetic-data", "fine-tuning"],
    },
    {
        "id": "q004",
        "description": "Model Context Protocol (MCP) tool use",
        "query": "Model Context Protocol tool use AI agents",
        "categories": ["cs.AI", "cs.SE"],
        "relevant_ids": [],  # emerging area — tests search quality vs. known groundtruth
        "tags": ["mcp", "tool-use"],
    },
    {
        "id": "q005",
        "description": "Computer use agents — GUI interaction",
        "query": "computer use agent GUI web navigation",
        "categories": ["cs.AI", "cs.HC"],
        "relevant_ids": ["2412.13671"],  # Fara-7B paper
        "tags": ["computer-use", "web-agents"],
    },
    {
        "id": "q006",
        "description": "LLM reasoning and chain-of-thought",
        "query": '"chain of thought" reasoning large language models',
        "categories": ["cs.AI", "cs.CL"],
        "relevant_ids": ["2201.11903"],  # Wei et al. CoT paper
        "tags": ["reasoning", "cot"],
    },
    {
        "id": "q007",
        "description": "Retrieval-augmented generation (RAG)",
        "query": "retrieval augmented generation knowledge grounding",
        "categories": ["cs.CL", "cs.IR"],
        "relevant_ids": ["2005.11401"],  # RAG paper (Lewis et al.)
        "tags": ["rag", "retrieval"],
    },
    {
        "id": "q008",
        "description": "Agent reliability and robustness",
        "query": "agent reliability robustness failure recovery multi-agent",
        "categories": ["cs.AI", "cs.MA"],
        "relevant_ids": [],
        "tags": ["reliability", "robustness"],
    },
]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query_id: str
    description: str
    returned_ids: list[str]
    relevant_ids: list[str]
    latency_s: float
    error: str | None = None

    def precision_at_k(self, k: int) -> float:
        if not self.returned_ids:
            return 0.0
        top_k = self.returned_ids[:k]
        hits = sum(1 for pid in top_k if pid in self.relevant_ids)
        return hits / k

    def recall_at_k(self, k: int) -> float:
        if not self.relevant_ids:
            return None  # undefined — no ground truth
        top_k = self.returned_ids[:k]
        hits = sum(1 for pid in self.relevant_ids if pid in top_k)
        return hits / len(self.relevant_ids)

    def reciprocal_rank(self) -> float:
        for rank, pid in enumerate(self.returned_ids, start=1):
            if pid in self.relevant_ids:
                return 1.0 / rank
        return 0.0


@dataclass
class BenchmarkReport:
    k: int
    total_queries: int
    errored_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float          # over queries with ground truth
    mrr: float
    mean_latency_s: float
    per_query: list[dict] = field(default_factory=list)

    def print_summary(self):
        print("\n" + "="*60)
        print(f"  ArXiv MCP Benchmark Results  (K={self.k})")
        print("="*60)
        print(f"  Queries run       : {self.total_queries}")
        print(f"  Errored           : {self.errored_queries}")
        print(f"  Precision@{self.k}      : {self.mean_precision_at_k:.3f}")
        print(f"  Recall@{self.k}        : {self.mean_recall_at_k:.3f}")
        print(f"  MRR               : {self.mrr:.3f}")
        print(f"  Mean latency      : {self.mean_latency_s:.2f}s")
        print("="*60)

        print("\nPer-query breakdown:")
        for q in self.per_query:
            status = "✓" if not q["error"] else "✗"
            p = f"P@{self.k}={q['precision_at_k']:.2f}" if not q["error"] else "ERROR"
            print(f"  {status} [{q['query_id']}] {q['description'][:45]:<45} {p}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_query(
    query_spec: dict,
    k: int,
) -> QueryResult:
    """
    Execute a single benchmark query against the live MCP server tools.
    Imports handlers directly to avoid subprocess overhead during eval.
    """
    from arxiv_mcp_server.tools.search import handle_search

    start = time.monotonic()
    try:
        raw = await handle_search({
            "query": query_spec["query"],
            "max_results": k,
            "categories": query_spec.get("categories", []),
            "sort_by": "relevance",
        })
        elapsed = time.monotonic() - start

        # Parse returned paper IDs
        data = json.loads(raw[0].text)
        returned_ids = [p["id"].split("v")[0] for p in data.get("papers", [])]

        return QueryResult(
            query_id=query_spec["id"],
            description=query_spec["description"],
            returned_ids=returned_ids,
            relevant_ids=query_spec.get("relevant_ids", []),
            latency_s=elapsed,
        )

    except Exception as e:
        elapsed = time.monotonic() - start
        return QueryResult(
            query_id=query_spec["id"],
            description=query_spec["description"],
            returned_ids=[],
            relevant_ids=query_spec.get("relevant_ids", []),
            latency_s=elapsed,
            error=str(e),
        )


async def run_benchmark(k: int = 10, output_path: str | None = None) -> BenchmarkReport:
    """Run all benchmark queries and compute aggregate metrics."""
    print(f"\nRunning ArXiv MCP benchmark ({len(BENCHMARK_QUERIES)} queries, K={k})...\n")

    results: list[QueryResult] = []
    for i, query_spec in enumerate(BENCHMARK_QUERIES, 1):
        print(f"  [{i}/{len(BENCHMARK_QUERIES)}] {query_spec['description']}...", end="", flush=True)
        result = await run_query(query_spec, k)
        results.append(result)
        status = f"  {result.latency_s:.1f}s" if not result.error else f"  ERROR: {result.error[:40]}"
        print(status)

    # Aggregate metrics
    errored = [r for r in results if r.error]
    valid = [r for r in results if not r.error]

    precisions = [r.precision_at_k(k) for r in valid]
    # Only compute recall over queries that have ground truth
    recalls = [r.recall_at_k(k) for r in valid if r.recall_at_k(k) is not None]
    rrs = [r.reciprocal_rank() for r in valid if r.relevant_ids]
    latencies = [r.latency_s for r in valid]

    report = BenchmarkReport(
        k=k,
        total_queries=len(results),
        errored_queries=len(errored),
        mean_precision_at_k=sum(precisions) / len(precisions) if precisions else 0.0,
        mean_recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
        mrr=sum(rrs) / len(rrs) if rrs else 0.0,
        mean_latency_s=sum(latencies) / len(latencies) if latencies else 0.0,
        per_query=[
            {
                "query_id": r.query_id,
                "description": r.description,
                "precision_at_k": r.precision_at_k(k),
                "recall_at_k": r.recall_at_k(k),
                "mrr": r.reciprocal_rank(),
                "latency_s": r.latency_s,
                "returned_ids": r.returned_ids,
                "relevant_ids": r.relevant_ids,
                "error": r.error,
            }
            for r in results
        ],
    )

    report.print_summary()

    if output_path:
        Path(output_path).write_text(json.dumps(asdict(report), indent=2))
        print(f"\nFull results saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="ArXiv MCP Server Benchmark")
    parser.add_argument("--k", type=int, default=10, help="Top-K cutoff for metrics")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()
    asyncio.run(run_benchmark(k=args.k, output_path=args.output))


if __name__ == "__main__":
    main()
