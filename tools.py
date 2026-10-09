"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

# arXiv: at least 3 s between calls (API etiquette)
_arxiv_last_call: float = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Backoff: uses Retry-After header value when provided, otherwise exponential
    base * 2**attempt + random jitter, capped at `cap` seconds.
    Non-RetryableError exceptions bubble up immediately (no retry).
    On the final failed attempt, re-raises without sleeping.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise  # last attempt — re-raise, no sleep
            if exc.retry_after is not None:
                delay = min(float(exc.retry_after), cap)
            else:
                delay = min(base * (2 ** attempt), cap) + random.uniform(0, 1)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}.
    Use for academic papers on any AI/ML topic. Prefer specific multi-word queries (e.g. 'world model latent dynamics')."""
    global _arxiv_last_call

    # Sanitize query: keep only word chars and hyphens
    terms = re.findall(r"[\w\-]+", query)
    if not terms:
        return "NO RESULTS"

    # Respect arXiv 3-second etiquette
    wait = 3.0 - (time.time() - _arxiv_last_call)
    if wait > 0:
        time.sleep(wait)

    search_query = " AND ".join(f"all:{t}" for t in terms)
    max_r = max(1, min(int(max_results), 30))

    try:
        def _call():
            global _arxiv_last_call
            _arxiv_last_call = time.time()
            resp = httpx.get(
                ARXIV_URL,
                params={
                    "search_query": search_query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": max_r,
                },
                timeout=30,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                raise RetryableError(
                    f"HTTP {resp.status_code}",
                    retry_after=float(retry_after) if retry_after else None,
                )
            resp.raise_for_status()
            return resp.text

        xml_text = with_retry(_call, attempts=7, base=2.0, cap=60.0)

        ns = "http://www.w3.org/2005/Atom"
        root = ET.fromstring(xml_text)
        records = []
        for entry in root.findall(f"{{{ns}}}entry"):
            raw_id = (entry.findtext(f"{{{ns}}}id") or "").strip()
            # e.g. http://arxiv.org/abs/2501.00001v1 → 2501.00001
            abs_part = raw_id.split("/abs/")[-1]
            arxiv_id = re.sub(r"v\d+$", "", abs_part)
            url = f"https://arxiv.org/abs/{arxiv_id}"
            published = (entry.findtext(f"{{{ns}}}published") or "")[:10]
            title = " ".join((entry.findtext(f"{{{ns}}}title") or "").split())
            summary_raw = entry.findtext(f"{{{ns}}}summary") or ""
            summary = " ".join(summary_raw.split())[:600]
            if arxiv_id:
                records.append({
                    "id": arxiv_id,
                    "url": url,
                    "published": published,
                    "title": title,
                    "summary": summary,
                })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except RetryableError as exc:
        return f"ERROR: RetryableError: {exc}"
    except httpx.TransportError as exc:
        return f"ERROR: TransportError: {exc}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
def _hf_record(item: dict) -> dict | None:
    """Map a single HF API item to our record shape. Returns None if paper.id is missing."""
    paper = item.get("paper") or item  # search endpoint may embed differently
    pid = paper.get("id") or paper.get("arxivId")
    if not pid:
        return None
    # prefer ai_summary when present (search endpoint)
    summary = paper.get("ai_summary") or paper.get("summary") or ""
    summary = " ".join(str(summary).split())[:600]
    published = (
        (paper.get("publishedAt") or paper.get("submittedOnDate") or "")[:10]
    )
    github = paper.get("githubRepo") or ""
    stars = paper.get("githubStars") or 0
    upvotes = paper.get("upvotes") or 0
    return {
        "id": pid,
        "url": f"https://huggingface.co/papers/{pid}",
        "published": published,
        "title": paper.get("title") or "",
        "summary": summary,
        "upvotes": upvotes,
        "github": github,
        "stars": stars,
    }


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    max_limit = max(1, min(int(limit), 100))
    params: dict = {"limit": max_limit}
    if date:
        params["date"] = date

    try:
        def _call():
            resp = httpx.get(HF_DAILY_URL, params=params, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                raise RetryableError(
                    f"HTTP {resp.status_code}",
                    retry_after=float(retry_after) if retry_after else None,
                )
            resp.raise_for_status()
            return resp.json()

        items = with_retry(_call, attempts=5, base=1.0, cap=30.0)

        records = []
        for item in items:
            rec = _hf_record(item)
            if rec is None:
                continue
            if keyword:
                combined = (rec["title"] + " " + rec["summary"]).lower()
                if keyword.lower() not in combined:
                    continue
            records.append(rec)

        records.sort(key=lambda r: r["upvotes"], reverse=True)

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except RetryableError as exc:
        return f"ERROR: RetryableError: {exc}"
    except httpx.TransportError as exc:
        return f"ERROR: TransportError: {exc}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    max_limit = max(1, min(int(limit), 50))

    try:
        def _call():
            resp = httpx.get(
                HF_SEARCH_URL,
                params={"q": query, "limit": max_limit},
                timeout=30,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                raise RetryableError(
                    f"HTTP {resp.status_code}",
                    retry_after=float(retry_after) if retry_after else None,
                )
            resp.raise_for_status()
            return resp.json()

        items = with_retry(_call, attempts=5, base=1.0, cap=30.0)

        records = []
        for item in items:
            rec = _hf_record(item)
            if rec is None:
                continue
            records.append(rec)

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except RetryableError as exc:
        return f"ERROR: RetryableError: {exc}"
    except httpx.TransportError as exc:
        return f"ERROR: TransportError: {exc}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----

def _exa_endpoint() -> str:
    """Return the Exa MCP endpoint URL, appending the API key when available."""
    key = os.environ.get("EXA_API_KEY", "").strip()
    if key:
        return f"{EXA_URL}?exaApiKey={key}"
    return EXA_URL


def _exa_call(tool_name: str, arguments: dict) -> str:
    """Low-level: call one Exa MCP tool and return the text content as a string.
    Raises RetryableError on rate-limit, RuntimeError on JSON-RPC error."""
    exa_key = os.environ.get("EXA_API_KEY", "").strip()
    endpoint = _exa_endpoint()

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    resp = httpx.post(
        endpoint,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        timeout=60,
    )

    if resp.status_code in (429, 500, 502, 503, 504):
        retry_after = resp.headers.get("Retry-After")
        raise RetryableError(
            f"HTTP {resp.status_code}",
            retry_after=float(retry_after) if retry_after else None,
        )
    # Note: Exa free-tier rate-limit also comes as HTTP 200 — handled below
    resp.raise_for_status()

    # Parse server-sent events: find lines starting with "data:"
    raw_text = resp.text
    data_lines = [
        line[len("data:"):].strip()
        for line in raw_text.splitlines()
        if line.startswith("data:")
    ]

    # Combine all data lines (some responses split across multiple SSE events)
    parsed = None
    for line in data_lines:
        try:
            parsed = json.loads(line)
            break  # take first valid JSON object
        except json.JSONDecodeError:
            continue

    if parsed is None:
        # Fallback: try parsing the whole body as JSON
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Unparseable Exa response: {raw_text[:300]}")

    # JSON-RPC error key
    if "error" in parsed:
        err = parsed["error"]
        raise RuntimeError(f"Exa JSON-RPC error: {err}")

    result = parsed.get("result", {})

    # Detect free-tier rate-limit: Exa returns HTTP 200 with _meta flag
    meta = result.get("_meta") or {}
    if meta.get("rateLimited") or meta.get("rate_limited"):
        retry_after = meta.get("retryAfter") or meta.get("retry_after")
        raise RetryableError("Exa rate limited", retry_after=retry_after)

    # Also detect rate-limit signal inside content text
    content_parts = result.get("content") or []
    text_parts = [p["text"] for p in content_parts if p.get("type") == "text"]
    combined = "\n".join(text_parts)

    # Exa free tier sometimes embeds rate-limit text in content
    if "rate limit" in combined.lower() or "ratelimit" in combined.lower():
        raise RetryableError("Exa rate limited (in content)", retry_after=20)

    def _redact(s: str) -> str:
        if exa_key:
            return s.replace(exa_key, "***")
        return s

    return _redact(combined)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    exa_key = os.environ.get("EXA_API_KEY", "").strip()

    if not objective:
        objective = f"Find information about: {query}"

    arguments = {
        "query": query,
        "objective": objective,
        "numResults": max(1, min(int(num_results), 20)),
    }

    try:
        def _call():
            return _exa_call("web_search_exa", arguments)

        return with_retry(_call, attempts=6, base=2.0, cap=60.0)

    except RetryableError as exc:
        msg = str(exc)
        if exa_key:
            msg = msg.replace(exa_key, "***")
        return f"ERROR: RetryableError: {msg}"
    except httpx.TransportError as exc:
        msg = str(exc)
        if exa_key:
            msg = msg.replace(exa_key, "***")
        return f"ERROR: TransportError: {msg}"
    except Exception as exc:
        msg = str(exc)
        if exa_key:
            msg = msg.replace(exa_key, "***")
        return f"ERROR: {type(exc).__name__}: {msg}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    exa_key = os.environ.get("EXA_API_KEY", "").strip()

    try:
        def _call():
            return _exa_call("web_fetch_exa", {"urls": [url]})

        result = with_retry(_call, attempts=6, base=2.0, cap=60.0)
        return result[:12000]

    except RetryableError as exc:
        msg = str(exc)
        if exa_key:
            msg = msg.replace(exa_key, "***")
        return f"ERROR: RetryableError: {msg}"
    except httpx.TransportError as exc:
        msg = str(exc)
        if exa_key:
            msg = msg.replace(exa_key, "***")
        return f"ERROR: TransportError: {msg}"
    except Exception as exc:
        msg = str(exc)
        if exa_key:
            msg = msg.replace(exa_key, "***")
        return f"ERROR: {type(exc).__name__}: {msg}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    # Load .env so EXA_API_KEY and other keys are available during local testing
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
