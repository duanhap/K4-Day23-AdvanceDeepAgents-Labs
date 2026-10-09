"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic: str) -> str:
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic or not topic.strip():
        return "topic"
    # lowercase
    slug = topic.lower()
    # replace any non-alphanumeric character sequence with a single dash
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # strip leading/trailing dashes
    slug = slug.strip("-")
    # truncate to 60 chars, then strip trailing dash again
    slug = slug[:60].rstrip("-")
    # guard against empty result after stripping
    if not slug:
        return "topic"
    # prevent path traversal: strip any leading dots or slashes
    slug = slug.lstrip(".-/")
    if not slug:
        return "topic"
    return slug


def build_prompt(topic: str) -> str:
    """The user message sent to the lead agent."""
    return (
        f"Please produce a comprehensive survey report on the following topic:\n\n"
        f"{topic}\n\n"
        f"Follow all eight steps in your system prompt. "
        f"The report must be well-structured, cite real sources, and pass the citation validator."
    )


def summarize(messages, elapsed: float, model_name: str) -> dict:
    """Return a dict with model, elapsed_s, subagent_calls, tool_calls, tokens.

    Walks the lead's message list; counts tool calls (subagent_calls = count of 'task');
    sums input/output token counts from usage_metadata.
    Lead messages only — subagent tokens are not counted here.
    """
    tool_calls: Counter = Counter()
    tokens_in = 0
    tokens_out = 0

    for msg in messages:
        # Count tool calls — AI messages have a tool_calls attribute or content with tool_use blocks
        # LangChain AIMessage stores tool_calls as a list
        calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            calls = msg.tool_calls
        elif hasattr(msg, "additional_kwargs"):
            calls = msg.additional_kwargs.get("tool_calls", [])

        for call in calls:
            # call may be a dict {"name": ..., "id": ..., "args": ...} or an object
            if isinstance(call, dict):
                name = call.get("name", "unknown")
            else:
                name = getattr(call, "name", "unknown")
            tool_calls[name] += 1

        # Token counts from usage_metadata (LangChain sets this on AI messages)
        meta = getattr(msg, "usage_metadata", None)
        if meta:
            tokens_in  += meta.get("input_tokens",  0)
            tokens_out += meta.get("output_tokens", 0)

    subagent_calls = tool_calls.get("task", 0)

    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls),
        "tokens": {
            "input": tokens_in,
            "output": tokens_out,
        },
    }


def save_outputs(backend, topic: str, messages, elapsed: float, model_name: str,
                 reports_dir: Path = REPORTS) -> Path:
    """Download the report from the sandbox and write the three output files. Returns the report path.

    Raises RuntimeError (writing nothing) if the report is missing/empty or sources.json is invalid.
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])

    # --- validate report ---
    report_bytes: bytes | None = files.get(REPORT_PATH)
    if not report_bytes:
        raise RuntimeError(
            f"agent did not produce a report at {REPORT_PATH} (file missing or empty)"
        )
    report_text = report_bytes.decode("utf-8", errors="replace").strip()
    if not report_text:
        raise RuntimeError("agent produced an empty report")

    # --- validate sources.json ---
    sources_bytes: bytes | None = files.get(SOURCES_PATH)
    if not sources_bytes:
        raise RuntimeError(
            f"agent did not produce sources.json at {SOURCES_PATH} (file missing)"
        )
    try:
        sources = json.loads(sources_bytes.decode("utf-8", errors="replace"))
        if not isinstance(sources, list):
            raise ValueError("sources.json is not a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"sources.json is invalid: {exc}") from exc

    # --- build meta ---
    slug = slugify(topic)
    summary = summarize(messages, elapsed, model_name)
    source_families = sorted({
        s.get("source", "")
        for s in sources
        if isinstance(s, dict) and s.get("source")
    })

    meta = {
        "topic": topic,
        **summary,
        "n_sources": len(sources),
        "source_families": source_families,
    }

    # --- write files (all or nothing: build content first, then write) ---
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path      = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path    = reports_dir / f"{slug}.meta.json"

    md_path.write_text(report_text, encoding="utf-8")
    sources_path.write_text(
        json.dumps(sources, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return md_path


def main(topic: str) -> int:
    """Run the full deep-research pipeline. Returns 0 on success, 1 on failure, 2 on bad usage."""
    topic = topic.strip()
    if not topic:
        print(
            "Usage: python research.py \"<topic>\"\n"
            "Example: python research.py \"survey about world model\"",
            file=sys.stderr,
        )
        return 2

    try:
        model = make_model()
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or str(model)
    except Exception as exc:
        print(f"FAILED to initialise model: {exc}", file=sys.stderr)
        return 1

    start = time.monotonic()

    try:
        with open_sandbox() as backend:
            # create workspace directories
            backend.execute(
                f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report"
            )

            # upload the two Python scripts the agent will run inside the sandbox
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })

            agent = build_lead_agent(backend, model)

            try:
                print(f"[research] invoking agent on topic: {topic!r}", flush=True)
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                    config={"recursion_limit": 1000},
                )
                print(f"[research] agent finished, messages: {len(result.get('messages', []))}", flush=True)
            except Exception as exc:
                # GraphRecursionError or any agent-level failure
                name = type(exc).__name__
                print(f"FAILED during agent run ({name}): {exc}", file=sys.stderr)
                return 1

            elapsed = time.monotonic() - start
            messages = result.get("messages", [])

            # print last few messages for debugging
            for m in messages[-5:]:
                mtype = type(m).__name__
                content = getattr(m, "content", "")
                if isinstance(content, str):
                    print(f"[debug] {mtype}: {content[:200]}", flush=True)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            print(f"[debug] {mtype}: {part['text'][:200]}", flush=True)

            try:
                report_path = save_outputs(
                    backend, topic, messages, elapsed, model_name
                )
            except RuntimeError as exc:
                print(f"FAILED: {exc}", file=sys.stderr)
                return 1

    except Exception as exc:
        name = type(exc).__name__
        print(f"FAILED ({name}): {exc}", file=sys.stderr)
        return 1

    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
