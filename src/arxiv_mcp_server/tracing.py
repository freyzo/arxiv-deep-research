"""
OpenTelemetry Tracing — ArXiv MCP Server
==========================================

Adds industry-standard observability to every tool call.
Mirrors AutoGen v0.4's built-in OTel support and Magentic-UI's
production-readiness philosophy.

Traces emitted:
  - mcp.tool.call          : every tool invocation (name, args, status, latency)
  - arxiv.search           : arXiv API query details + result count
  - arxiv.download         : paper download + conversion latency
  - arxiv.read             : paper read operations

Usage:
    # Automatic — just set env vars before running the server:
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
    export OTEL_SERVICE_NAME=arxiv-mcp-server
    python -m arxiv_mcp_server

    # Or use the Jaeger all-in-one for local dev:
    docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one
    open http://localhost:16686

    # Console exporter (no infrastructure needed):
    export ARXIV_MCP_TRACE_CONSOLE=true
    python -m arxiv_mcp_server
"""

import os
import time
import functools
import logging
from typing import Any, Callable

logger = logging.getLogger("arxiv-mcp-server.tracing")

# ---------------------------------------------------------------------------
# Optional OTel import — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        _OTLP_AVAILABLE = True
    except ImportError:
        _OTLP_AVAILABLE = False

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    logger.info(
        "opentelemetry-sdk not installed. Tracing disabled. "
        "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc"
    )


def setup_tracing() -> None:
    """
    Initialize OTel tracing. Called once at server startup.
    Supports OTLP (production) and Console (dev/debug) exporters.
    """
    if not _OTEL_AVAILABLE:
        return

    service_name = os.environ.get("OTEL_SERVICE_NAME", "arxiv-mcp-server")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)

    # Console exporter — useful for local dev + demos
    if os.environ.get("ARXIV_MCP_TRACE_CONSOLE", "").lower() in ("true", "1", "yes"):
        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
        logger.info("Tracing: console exporter enabled")

    # OTLP exporter — for Jaeger, Honeycomb, Azure Monitor, etc.
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint and _OTLP_AVAILABLE:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"Tracing: OTLP exporter enabled → {otlp_endpoint}")

    trace.set_tracer_provider(provider)
    logger.info(f"OTel tracing initialized (service={service_name})")


def get_tracer():
    """Return the module-level tracer."""
    if not _OTEL_AVAILABLE:
        return _NoOpTracer()
    return trace.get_tracer("arxiv-mcp-server", schema_url="https://opentelemetry.io/schemas/1.11.0")


# ---------------------------------------------------------------------------
# No-op tracer — used when OTel is not installed
# Keeps the rest of the code clean (no if-checks everywhere)
# ---------------------------------------------------------------------------

class _NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_attribute(self, *args): pass
    def set_status(self, *args): pass
    def record_exception(self, *args): pass


class _NoOpTracer:
    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Decorator: trace any async tool handler
# ---------------------------------------------------------------------------

def trace_tool(tool_name: str):
    """
    Decorator that wraps an async tool handler with an OTel span.

    Usage:
        @trace_tool("search_papers")
        async def handle_search(arguments):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(arguments: dict[str, Any], *args, **kwargs):
            tracer = get_tracer()
            start = time.monotonic()

            with tracer.start_as_current_span(f"mcp.tool.{tool_name}") as span:
                # Record tool invocation metadata
                span.set_attribute("mcp.tool.name", tool_name)
                span.set_attribute("mcp.tool.args_keys", str(list(arguments.keys())))

                # Tool-specific attributes
                if tool_name == "search_papers":
                    span.set_attribute("arxiv.query", arguments.get("query", ""))
                    span.set_attribute("arxiv.max_results", arguments.get("max_results", 10))
                    cats = arguments.get("categories", [])
                    span.set_attribute("arxiv.categories", ",".join(cats))

                elif tool_name in ("download_paper", "read_paper"):
                    span.set_attribute("arxiv.paper_id", arguments.get("paper_id", ""))

                try:
                    result = await func(arguments, *args, **kwargs)
                    elapsed = time.monotonic() - start
                    span.set_attribute("mcp.tool.latency_ms", int(elapsed * 1000))
                    span.set_attribute("mcp.tool.status", "ok")

                    # For search: record result count from response
                    if tool_name == "search_papers" and result:
                        import json
                        try:
                            data = json.loads(result[0].text)
                            span.set_attribute("arxiv.result_count", data.get("total_results", 0))
                        except Exception:
                            pass

                    return result

                except Exception as exc:
                    elapsed = time.monotonic() - start
                    span.set_attribute("mcp.tool.latency_ms", int(elapsed * 1000))
                    span.set_attribute("mcp.tool.status", "error")
                    span.record_exception(exc)
                    raise

        return wrapper
    return decorator
