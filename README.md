# arXiv MCP Server — Research Discovery Agent for Multi-Agent Systems

[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Server-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=arxiv&inputs=%5B%5D&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22arxiv-mcp-server%22%5D%7D) [![Install in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-Install_Server-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=arxiv&inputs=%5B%5D&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22arxiv-mcp-server%22%5D%7D&quality=insider)

A **Model Context Protocol (MCP) server** for searching, downloading, and reading arXiv papers — designed as a specialist agent for integration into multi-agent systems like [Microsoft Magentic-UI](https://github.com/microsoft/magentic-ui) and [AutoGen](https://github.com/microsoft/autogen).

> **The idea:** Rather than treating arXiv search as a simple lookup tool, this server is structured as a first-class research agent — one you can plug directly into a Magentic-One-style team as an `McpAgent`, giving an Orchestrator access to the full scientific literature as a delegatable resource.

---

## Integration with Magentic-UI

Magentic-UI supports custom `McpAgent` instances via `mcp_agent_configs` in its config file. This server plugs in directly:

```yaml
# examples/magentic_ui_config.yaml
client:
  mcp_agent_configs:
    - agent_name: ArxivResearcher
      description: >
        Specialist agent for searching and reading arXiv papers.
        Use when the task requires finding academic papers, understanding
        research literature, or retrieving technical details from published work.
      server_params:
        type: StdioServerParams
        command: python
        args: ["-m", "arxiv_mcp_server"]
        env:
          PYTHONPATH: /path/to/arxiv-mcp-server/src
```

Once registered, the Magentic-UI Orchestrator can delegate research subtasks to this agent through the standard Task Ledger / Progress Ledger pattern — exactly how WebSurfer handles web browsing, but for academic literature.

---

## Integration with AutoGen AgentChat

See [`examples/autogen_research_team.py`](examples/autogen_research_team.py) for a complete 3-agent team:

```
Orchestrator (MagenticOneGroupChat)
├── ArxivSurfer  ← this MCP server, wrapped via StdioServerParams + mcp_server_tools
└── Coder        ← synthesizes findings into structured markdown reports
```

```bash
pip install "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.2.0"
export OPENAI_API_KEY=...
python examples/autogen_research_team.py
```

---

## Tools

| Tool | Description |
|------|-------------|
| `search_papers` | Query arXiv with advanced filters: date range, category, sort by relevance or date |
| `download_paper` | Fetch a paper PDF and convert to clean markdown for LLM consumption |
| `read_paper` | Access previously downloaded paper content |
| `list_papers` | View all papers in local storage |

### search_papers

Supports rich query syntax — quoted phrases, boolean operators, field-specific search (`ti:`, `au:`, `abs:`), and category filtering:

```json
{
  "query": "\"multi-agent\" AND \"orchestration\" ANDNOT survey",
  "max_results": 10,
  "date_from": "2024-01-01",
  "categories": ["cs.AI", "cs.MA"],
  "sort_by": "relevance"
}
```

---

## Evaluation Benchmark

The repo includes a retrieval quality benchmark (`eval/benchmark.py`) measuring:

- **Precision@K** — fraction of top-K results that are relevant
- **Recall@K** — fraction of known relevant papers found in top-K
- **MRR** — Mean Reciprocal Rank of first relevant result

Ground-truth queries are seeded from landmark papers (AutoGen `2308.08155`, Magentic-One `2403.16971`, RAG `2005.11401`, CoT `2201.11903`) and can be extended automatically using the synthetic data pipeline below.

```bash
python eval/benchmark.py --k 10 --output results.json
```

---

## Synthetic Eval Data Generation (AgentInstruct-style)

`scripts/generate_eval_tasks.py` implements a 4-stage pipeline that generates diverse benchmark queries from arXiv abstracts — mirroring the [AgentInstruct](https://arxiv.org/abs/2407.10505) approach:

```
Stage 1: Seed collection     → fetch paper abstracts from arXiv by category
Stage 2: Content transform   → extract key concepts and problem statements
Stage 3: Instruction gen     → generate realistic research queries via GPT-4o-mini
Stage 4: Instruction refine  → create harder variants at subtopic intersections
```

```bash
export OPENAI_API_KEY=...
python scripts/generate_eval_tasks.py --seed-category cs.AI --num-seeds 20 --output eval/generated_queries.json
```

Output includes easy/medium/hard difficulty tiers for stratified evaluation.

---

## Observability: OpenTelemetry Tracing

Every tool call is instrumented with OpenTelemetry spans (mirrors [AutoGen v0.4's built-in OTel support](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/telemetry.html)):

```bash
# Console output (no infrastructure needed)
export ARXIV_MCP_TRACE_CONSOLE=true
python -m arxiv_mcp_server

# OTLP export to Jaeger / Azure Monitor
docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=arxiv-mcp-server
python -m arxiv_mcp_server
# View traces: http://localhost:16686
```

Spans recorded: `mcp.tool.search_papers`, `mcp.tool.download_paper`, `mcp.tool.read_paper` — each with query, categories, result count, latency, and error status as attributes.

Tracing is a zero-cost no-op when `opentelemetry-sdk` is not installed.

---

## Installation

**Requires Python 3.11+**

```bash
git clone https://github.com/freyzo/mcp-arxiv
cd mcp-arxiv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: OTel tracing
pip install -e ".[tracing]"
```

### Claude Desktop

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "arxiv_mcp_server", "--storage-path", "/path/to/papers"]
    }
  }
}
```

### Cursor

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "python",
      "args": ["-m", "arxiv_mcp_server"],
      "env": { "PYTHONPATH": "/path/to/mcp-arxiv/src" }
    }
  }
}
```

---

## Prompts

### deep-paper-analysis

Comprehensive analysis workflow covering executive summary, methodology, results, implications, and future directions:

```json
{ "paper_id": "2401.12345" }
```

---

## Repository Structure

```
mcp-arxiv/
├── src/arxiv_mcp_server/
│   ├── server.py          # MCP server + OTel init
│   ├── tracing.py         # @trace_tool decorator, OTLP + console exporters
│   ├── config.py
│   ├── tools/             # search, download, read, list
│   └── prompts/           # deep research analysis prompt
├── examples/
│   ├── autogen_research_team.py   # Magentic-One-style 3-agent team
│   └── magentic_ui_config.yaml    # McpAgent config for Magentic-UI
├── eval/
│   └── benchmark.py       # Precision@K / Recall@K / MRR harness
├── scripts/
│   └── generate_eval_tasks.py     # AgentInstruct-style query generator
└── pyproject.toml
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARXIV_STORAGE_PATH` | `~/.arxiv-mcp-server/papers` | Paper storage location |
| `ARXIV_MCP_TRACE_CONSOLE` | `false` | Enable console trace output |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OTLP endpoint (e.g. `http://localhost:4317`) |
| `OTEL_SERVICE_NAME` | `arxiv-mcp-server` | Service name in traces |

---

## License

MIT
