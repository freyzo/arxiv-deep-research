"""
Synthetic Eval Task Generator — AgentInstruct-style Pipeline
=============================================================

Generates diverse benchmark queries from arXiv paper abstracts,
mirroring the AgentInstruct approach:
  Seed (abstract) → Content Transformation → Instruction Generation → Refinement

This creates a self-expanding eval dataset without manual curation.
The generated tasks are appended to eval/benchmark.py's BENCHMARK_QUERIES.

Usage:
    pip install openai arxiv
    export OPENAI_API_KEY=...
    python scripts/generate_eval_tasks.py --seed-category cs.AI --num-seeds 20
    python scripts/generate_eval_tasks.py --seed-category cs.MA --num-seeds 10 --output eval/generated_queries.json

Pipeline stages (mirrors AgentInstruct):
  1. Seed collection     : fetch paper abstracts from arXiv by category
  2. Content transform   : extract key concepts and problem statements
  3. Instruction gen     : generate realistic research queries from abstracts
  4. Instruction refine  : diversify + add difficulty variation
"""

import asyncio
import json
import argparse
import sys
import re
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class SeedPaper:
    paper_id: str
    title: str
    abstract: str
    categories: list[str]


@dataclass
class GeneratedQuery:
    id: str
    description: str
    query: str
    categories: list[str]
    relevant_ids: list[str]
    tags: list[str]
    source_paper_id: str
    difficulty: str  # "easy" | "medium" | "hard"


# ---------------------------------------------------------------------------
# Stage 1: Seed collection
# ---------------------------------------------------------------------------

async def collect_seeds(category: str, num_seeds: int) -> list[SeedPaper]:
    """
    Fetch recent paper abstracts from arXiv to use as generation seeds.
    These seeds ground the generated queries in real research.
    """
    from arxiv_mcp_server.tools.search import handle_search
    import json

    print(f"[Stage 1] Collecting {num_seeds} seed papers from {category}...")

    raw = await handle_search({
        "query": f"cat:{category}",
        "max_results": num_seeds,
        "categories": [category],
        "sort_by": "date",
    })

    data = json.loads(raw[0].text)
    seeds = []
    for paper in data.get("papers", []):
        seeds.append(SeedPaper(
            paper_id=paper["id"],
            title=paper["title"],
            abstract=paper["abstract"],
            categories=paper["categories"],
        ))

    print(f"  Collected {len(seeds)} seeds")
    return seeds


# ---------------------------------------------------------------------------
# Stage 2 + 3: Content transformation + instruction generation
# ---------------------------------------------------------------------------

TRANSFORM_SYSTEM_PROMPT = """You are a research query generator for an AI evaluation benchmark.
Given a paper title and abstract, generate a realistic search query that a researcher
would use to find this paper and related work.

Output ONLY valid JSON with this structure:
{
  "description": "one-line description of what the query finds",
  "query": "the search query string (2-8 words, no boolean operators)",
  "tags": ["tag1", "tag2"],
  "difficulty": "easy|medium|hard"
}

Rules:
- query must be a natural research topic, not the paper title verbatim
- difficulty: easy = broad topic, medium = specific technique, hard = niche intersection
- tags: 2-3 lowercase topic tags from the paper's domain
- Do not include author names in the query
"""

REFINE_SYSTEM_PROMPT = """You are refining a set of research queries for diversity.
Given a list of queries, add one harder variation of each that asks for a more specific
subtopic or intersection. Output ONLY a JSON array of the new harder queries with the
same structure as the input."""


async def transform_seed_to_query(
    seed: SeedPaper,
    client,
    idx: int,
) -> GeneratedQuery | None:
    """Stage 2+3: Transform a single paper abstract into an eval query."""
    try:
        prompt = f"Title: {seed.title}\n\nAbstract: {seed.abstract[:800]}"

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": TRANSFORM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        return GeneratedQuery(
            id=f"gen_{idx:03d}",
            description=data["description"],
            query=data["query"],
            categories=seed.categories[:2],
            relevant_ids=[seed.paper_id],
            tags=data.get("tags", []),
            source_paper_id=seed.paper_id,
            difficulty=data.get("difficulty", "medium"),
        )

    except Exception as e:
        print(f"  [warn] Failed to generate query for {seed.paper_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 4: Instruction refinement — generate harder variants
# ---------------------------------------------------------------------------

async def refine_queries(
    queries: list[GeneratedQuery],
    client,
) -> list[GeneratedQuery]:
    """
    Stage 4: Generate harder variations of existing queries.
    Mirrors AgentInstruct's 'Instruction Refinement Flow' that explores
    the neighborhood of existing instructions for diversity.
    """
    print(f"\n[Stage 4] Refining {len(queries)} queries to generate harder variants...")

    simple_queries = [
        {"description": q.description, "query": q.query, "tags": q.tags}
        for q in queries[:10]  # Refine a sample
    ]

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(simple_queries)},
            ],
            temperature=0.8,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        refined_data = json.loads(raw)

        refined = []
        for i, item in enumerate(refined_data):
            if i < len(queries):
                refined.append(GeneratedQuery(
                    id=f"ref_{i:03d}",
                    description=item.get("description", queries[i].description + " (hard variant)"),
                    query=item.get("query", queries[i].query),
                    categories=queries[i].categories,
                    relevant_ids=[],  # No ground truth for refined variants
                    tags=item.get("tags", queries[i].tags),
                    source_paper_id=queries[i].source_paper_id,
                    difficulty="hard",
                ))

        print(f"  Generated {len(refined)} refined variants")
        return refined

    except Exception as e:
        print(f"  [warn] Refinement failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def generate_eval_tasks(
    seed_category: str,
    num_seeds: int,
    output_path: str,
    openai_api_key: str,
) -> list[GeneratedQuery]:
    """
    Full AgentInstruct-style pipeline:
      Seed → Transform → Generate → Refine
    """
    import openai
    client = openai.AsyncOpenAI(api_key=openai_api_key)

    # Stage 1: Collect seeds
    seeds = await collect_seeds(seed_category, num_seeds)

    # Stage 2+3: Transform seeds into queries (concurrent)
    print(f"\n[Stage 2+3] Generating queries from {len(seeds)} seeds...")
    tasks = [
        transform_seed_to_query(seed, client, idx)
        for idx, seed in enumerate(seeds)
    ]
    raw_queries = await asyncio.gather(*tasks)
    queries = [q for q in raw_queries if q is not None]
    print(f"  Generated {len(queries)} queries")

    # Stage 4: Refinement
    refined = await refine_queries(queries, client)
    all_queries = queries + refined

    # Save output
    output = [
        {
            "id": q.id,
            "description": q.description,
            "query": q.query,
            "categories": q.categories,
            "relevant_ids": q.relevant_ids,
            "tags": q.tags,
            "source_paper_id": q.source_paper_id,
            "difficulty": q.difficulty,
        }
        for q in all_queries
    ]

    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"\n[Done] Saved {len(output)} queries to {output_path}")
    print("\nQuery difficulty distribution:")
    for difficulty in ["easy", "medium", "hard"]:
        count = sum(1 for q in all_queries if q.difficulty == difficulty)
        print(f"  {difficulty:8s}: {count}")

    return all_queries


def main():
    import os
    parser = argparse.ArgumentParser(
        description="Generate ArXiv MCP eval tasks using AgentInstruct-style pipeline"
    )
    parser.add_argument("--seed-category", default="cs.AI",
                        help="ArXiv category to seed from (default: cs.AI)")
    parser.add_argument("--num-seeds", type=int, default=15,
                        help="Number of seed papers to collect")
    parser.add_argument("--output", default="eval/generated_queries.json",
                        help="Output path for generated queries")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    asyncio.run(generate_eval_tasks(
        seed_category=args.seed_category,
        num_seeds=args.num_seeds,
        output_path=args.output,
        openai_api_key=api_key,
    ))


if __name__ == "__main__":
    main()
