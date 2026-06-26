#!/usr/bin/env python3
"""Extract C/C++ comments with source line numbers.

This is intentionally lexer-like rather than regex-based so comment markers
inside string and character literals are ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_comments(text: str) -> list[tuple[int, str]]:
    comments: list[tuple[int, str]] = []
    line = 1
    i = 0
    n = len(text)
    state = "code"

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if state == "code":
            if ch == "\n":
                line += 1
                i += 1
            elif ch == '"':
                state = "string"
                i += 1
            elif ch == "'":
                state = "char"
                i += 1
            elif ch == "/" and nxt == "/":
                start_line = line
                i += 2
                start = i
                while i < n and text[i] != "\n":
                    i += 1
                comments.append((start_line, "//" + text[start:i]))
            elif ch == "/" and nxt == "*":
                start_line = line
                i += 2
                lines = ["/*"]
                current = []
                while i < n:
                    ch = text[i]
                    nxt = text[i + 1] if i + 1 < n else ""
                    if ch == "*" and nxt == "/":
                        current.append("*/")
                        i += 2
                        break
                    if ch == "\n":
                        lines[-1] += "".join(current)
                        current = []
                        line += 1
                        i += 1
                        lines.append("")
                    else:
                        current.append(ch)
                        i += 1
                else:
                    lines[-1] += "".join(current)
                if current:
                    lines[-1] += "".join(current)
                comments.append((start_line, "\n".join(lines)))
            else:
                i += 1
        elif state == "string":
            if ch == "\\":
                i += 2
            elif ch == '"':
                state = "code"
                i += 1
            else:
                if ch == "\n":
                    line += 1
                i += 1
        elif state == "char":
            if ch == "\\":
                i += 2
            elif ch == "'":
                state = "code"
                i += 1
            else:
                if ch == "\n":
                    line += 1
                i += 1

    return comments


def write_comments(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    comments = extract_comments(text)

    with output.open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"# Comments extracted from {source}\n")
        out.write(f"# Total comments: {len(comments)}\n\n")
        for line_no, comment in comments:
            out.write(f"--- line {line_no} ---\n")
            out.write(comment.rstrip())
            out.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="C/C++ source file to scan")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output text file. Defaults to <source-stem>-comments.txt beside the source.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = (args.output or source.with_name(f"{source.stem}-comments.txt")).resolve()
    write_comments(source, output)
    print(output)


if __name__ == "__main__":
    main()
