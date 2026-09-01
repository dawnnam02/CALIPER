"""Make output survive a console that is not UTF-8.

Windows terminals still default to a legacy code page -- cp949 on a Korean
install, cp1252 elsewhere -- and printing an em dash there raises
UnicodeEncodeError and kills the program. That is a rude way for a tool to fail
on someone else's machine, and it is invisible when you only ever run it on
your own.

Two defences, because either alone is insufficient:
  * ``setup()`` reconfigures stdout to UTF-8 where the platform allows it, and
    falls back to replacing unencodable characters rather than raising.
  * ``ascii_safe()`` rewrites the handful of typographic characters this
    project actually uses, so output stays readable even when the fallback
    kicks in.

Korean note:
윈도우 콘솔은 아직 기본이 cp949 라서 em dash 하나에 프로그램이 죽는다.
자기 컴퓨터에서만 돌리면 절대 안 보이는 종류의 버그다.
"""

from __future__ import annotations

import sys

# characters this project types that legacy code pages cannot encode
REPLACEMENTS = {
    "\u2014": "--",   # em dash
    "\u2013": "-",    # en dash
    "\u2192": "->",   # right arrow
    "\u2190": "<-",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u00d7": "x",    # multiplication sign
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2605": "*",
}


def ascii_safe(text: str) -> str:
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def setup() -> None:
    """Best effort: UTF-8 stdout, and never raise on an unencodable character."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


def puts(*parts: object) -> None:
    """print() that cannot take the program down on a legacy console."""
    text = " ".join(str(p) for p in parts)
    try:
        print(text)
    except UnicodeEncodeError:
        print(ascii_safe(text))
