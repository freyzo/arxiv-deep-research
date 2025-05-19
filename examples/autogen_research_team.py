"""
AutoGen Research Team — ArXiv MCP Agent Integration
====================================================

Demonstrates integrating the arxiv-mcp-server as a specialist agent
inside a Magentic-One-style multi-agent team using AutoGen AgentChat.

Architecture mirrors Microsoft AI Frontiers' Magentic-One pattern:
  - Orchestrator  : high-level planning + task delegation
  - ArxivSurfer   : specialist agent backed by the ArXiv MCP server
  - Coder         : synthesizes findings into structured output

Usage:
    pip install "autogen-agentchat" "autogen-ext[openai]" "mcp>=1.2.0"
    export OPENAI_API_KEY=...
    python examples/autogen_research_team.py
"""

import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools


async def build_arxiv_tools() -> list:
    """
    Launch the arxiv-mcp-server as a subprocess and return its
    tools as AutoGen-compatible function tools.

    The MCP server is started via stdio transport — same mechanism
    used by Magentic-UI's McpAgent pattern.
    """
    server_params = StdioServerParams(
        command="python",
        args=["-m", "arxiv_mcp_server"],
        env={
            "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src"),
            **os.environ,
        },
    )
    tools = await mcp_server_tools(server_params)
    return tools


async def main():
    """
    Run a Magentic-One-style research team that:
      1. Takes a research topic as input
      2. Orchestrator breaks it into subtasks
      3. ArxivSurfer searches + downloads relevant papers
      4. Coder synthesizes findings into a structured report

    This pattern directly mirrors how Magentic-One's Orchestrator
    delegates to specialized agents (WebSurfer, FileSurfer, Coder).
    """
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # Load ArXiv MCP tools
    arxiv_tools = await build_arxiv_tools()
    print(f"[setup] Loaded {len(arxiv_tools)} tools from arxiv-mcp-server: "
          f"{[t.name for t in arxiv_tools]}")

    # ArxivSurfer — specialist agent with MCP tool access
    # Mirrors Magentic-One's WebSurfer pattern but for academic literature
    arxiv_surfer = AssistantAgent(
        name="ArxivSurfer",
        model_client=model_client,
        tools=arxiv_tools,
        system_message=(
            "You are ArxivSurfer, a specialist research agent with access to the "
            "arXiv academic paper repository. Your job is to:\n"
            "1. Search for papers relevant to the assigned topic\n"
            "2. Download and read the most relevant papers\n"
            "3. Extract key contributions, methods, and results\n"
            "4. Report findings back to the Orchestrator in structured form\n\n"
            "Always search before claiming papers don't exist. "
            "Prioritize recent papers (2023+) unless asked for foundational work."
        ),
    )

    # Coder — synthesizes raw findings into a structured markdown report
    coder = AssistantAgent(
        name="Coder",
        model_client=model_client,
        system_message=(
            "You are Coder, a technical writer and synthesis agent. "
            "When ArxivSurfer provides paper findings, your job is to:\n"
            "1. Synthesize findings into a clean markdown research report\n"
            "2. Create comparison tables where relevant\n"
            "3. Identify open problems and future directions\n"
            "4. Flag papers that should be read in full\n\n"
            "Be precise, cite paper IDs, and avoid hallucinating results."
        ),
    )

    # MagenticOneGroupChat — uses the same Orchestrator + inner/outer loop
    # pattern from the Magentic-One paper (Task Ledger + Progress Ledger)
    team = MagenticOneGroupChat(
        participants=[arxiv_surfer, coder],
        model_client=model_client,
        max_turns=20,
    )

    # Run the team on a research task
    research_task = (
        "Survey the latest research on multi-agent LLM systems with a focus on "
        "orchestration patterns, reliability, and human-in-the-loop design. "
        "Find at least 5 papers from 2024-2025. Produce a structured markdown report "
        "with: (1) paper summaries, (2) a comparison table of approaches, "
        "(3) open problems."
    )

    print("\n" + "="*60)
    print("ARXIV RESEARCH TEAM — Multi-Agent Run")
    print("="*60)
    print(f"Task: {research_task}\n")

    await Console(team.run_stream(task=research_task))


if __name__ == "__main__":
    asyncio.run(main())
