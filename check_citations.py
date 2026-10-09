"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    Rules:
      1. sources must not be empty
      2. each source: n is int, url starts http(s)://, no duplicate urls
      3. report must have "## References"; body = text before it
      4. every [n] in body exists in sources; every source cited >= 1 time in body
      5. References section has exactly one line per source, line starts [n], no dup/missing numbers
      6. each reference line has exactly 1 URL matching sources[n].url; no bundling
      7. expand [1, 2] / [1-3] to individual numbers; ignore [n] in code blocks and [n](url) markdown links
    """
    import re

    problems = []

    # --- Rule 1: sources not empty ---
    if not sources:
        return ["no sources in sources.json"]

    # --- Rule 2: validate each source entry ---
    seen_urls = {}
    source_by_n = {}
    for entry in sources:
        n = entry.get("n")
        url = entry.get("url", "")
        if not isinstance(n, int):
            problems.append(f"source n={n!r} is not an integer")
        if not isinstance(url, str) or not re.match(r"https?://", url):
            problems.append(f"source n={n!r} has invalid url: {url!r}")
        else:
            if url in seen_urls:
                problems.append(f"duplicate url {url!r} in sources (n={seen_urls[url]} and n={n})")
            else:
                seen_urls[url] = n
        if isinstance(n, int):
            source_by_n[n] = entry

    # --- Rule 3: must have ## References heading ---
    ref_heading_re = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
    heading_matches = list(ref_heading_re.finditer(report_text))
    if not heading_matches:
        problems.append("report is missing ## References section")
        return problems  # can't continue without splitting body/refs

    # split at the LAST occurrence (finalize_citations.py does the same)
    split_pos = heading_matches[-1].start()
    body = report_text[:split_pos]
    refs_section = report_text[heading_matches[-1].end():]

    # --- Helpers for citation extraction ---

    # Remove code blocks from a text segment before scanning citations
    code_block_re = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
    # [n](url) markdown links — must be ignored
    md_link_re = re.compile(r"\[\d+(?:\s*[,–\-]\s*\d+)*\]\([^)]*\)")

    def extract_cited_numbers(text):
        """Extract all citation numbers from text, skipping code blocks and markdown links."""
        # strip code blocks first
        no_code = code_block_re.sub("", text)
        # strip markdown links [n](url)
        no_links = md_link_re.sub("", no_code)
        # match [n], [1,2], [1-3], [1–3]
        group_re = re.compile(r"\[(\d+(?:\s*[,–\-]\s*\d+)*)\](?!\()")
        cited = set()
        for m in group_re.finditer(no_links):
            group = m.group(1)
            # expand ranges and comma lists
            for part in re.split(r"\s*,\s*", group):
                span = re.fullmatch(r"(\d+)\s*[–\-]\s*(\d+)", part.strip())
                if span:
                    a, b = int(span.group(1)), int(span.group(2))
                    if 0 <= b - a <= 200:
                        cited.update(range(a, b + 1))
                    else:
                        cited.add(a)
                        cited.add(b)
                else:
                    try:
                        cited.add(int(part.strip()))
                    except ValueError:
                        pass
        return cited

    # --- Rule 4: citations in body vs sources ---
    body_cited = extract_cited_numbers(body)

    for n in sorted(body_cited):
        if n not in source_by_n:
            problems.append(f"[{n}] cited in body but missing from sources.json")

    for n in sorted(source_by_n):
        if n not in body_cited:
            problems.append(f"source [{n}] is never cited in the report body")

    # --- Rule 5 & 6: validate References section lines ---
    # A reference line starts with [n] at the beginning of the line
    ref_line_re = re.compile(r"(?m)^\[(\d+)\](.*)$")
    url_re = re.compile(r"https?://\S+")

    ref_line_numbers = []  # in order of appearance
    seen_ref_nums = {}     # n -> line content

    for m in ref_line_re.finditer(refs_section):
        n = int(m.group(1))
        line_content = m.group(0)

        # Rule 5a: no duplicate reference numbers
        if n in seen_ref_nums:
            problems.append(f"duplicate reference line [{n}] in ## References")
        else:
            seen_ref_nums[n] = line_content
            ref_line_numbers.append(n)

    # Rule 5b: every source must have exactly one reference line
    for n in sorted(source_by_n):
        if n not in seen_ref_nums:
            problems.append(f"source [{n}] has no line in ## References")

    # Rule 5c: no reference line for a number not in sources
    for n in ref_line_numbers:
        if n not in source_by_n:
            problems.append(f"## References has line [{n}] but it is not in sources.json")

    # Rule 6: each reference line has exactly 1 URL and it matches sources[n].url
    for n, line in seen_ref_nums.items():
        urls_in_line = url_re.findall(line)
        if len(urls_in_line) == 0:
            problems.append(f"reference line [{n}] has no URL")
        elif len(urls_in_line) > 1:
            problems.append(f"reference line [{n}] has multiple URLs (bundled sources not allowed)")
        else:
            if n in source_by_n:
                expected = source_by_n[n].get("url", "")
                # strip trailing punctuation that may have been caught by \S+
                actual = urls_in_line[0].rstrip(")")
                if actual != expected:
                    problems.append(
                        f"reference line [{n}] URL {actual!r} does not match sources.json url {expected!r}"
                    )

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8-sig") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8-sig") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
