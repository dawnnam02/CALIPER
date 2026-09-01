"""Command line: point it at a CSV and it tells you what is wrong.

    caliper-audit designs.csv --score af3_ipSAE_min --outcome binder
    caliper-audit designs.csv --score iptm --outcome binding --group target_id
    caliper-audit designs.csv --score pae_interaction --outcome binding --lower-is-better

The project's earlier entry point ran a simulated campaign, which made sense
when the goal was to build a pipeline.  It is not the goal any more.  What a
campaign actually needs is to hand over the scores and outcomes it already has
and be told which known failure modes apply.

Korean note:
예전 진입점은 시뮬레이션 캠페인을 돌렸다.  파이프라인을 만드는 게 목표였을 때는
말이 됐지만 지금은 아니다.  실제로 필요한 건 "이미 가진 점수와 결과를 넘기면
알려진 실패 양상 중 어디에 해당하는지 말해주는 것"이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .console import ascii_safe, setup as _console_setup
from .audit import audit
from .metrics import roc_auc


def _read_csv(path: Path):
    try:
        import pandas as pd
    except ImportError:
        print("this command needs pandas:  pip install pandas", file=sys.stderr)
        raise SystemExit(2)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        raise SystemExit(2)
    return pd.read_csv(path, low_memory=False)


def _to_binary(col, positive: str | None):
    """Turn an outcome column into 0/1, accepting what real files contain.

    Real datasets use true/false strings, yes/no, 1/0, and booleans, often with
    an 'unknown' level that must be dropped rather than silently counted as a
    failure.
    """
    import pandas as pd

    s = pd.Series(col)
    if positive is not None:
        return (s.astype(str).str.lower() == positive.lower()).astype(float), s.notna()
    if s.dtype == bool:
        return s.astype(float), s.notna()
    lowered = s.astype(str).str.strip().str.lower()
    truthy = {"true", "yes", "1", "1.0", "binder", "binding", "positive"}
    falsy = {"false", "no", "0", "0.0", "non-binder", "none", "negative"}
    known = lowered.isin(truthy | falsy)
    if known.mean() > 0.8:
        return lowered.isin(truthy).astype(float), known
    numeric = pd.to_numeric(s, errors="coerce")
    return (numeric > 0).astype(float), numeric.notna()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="caliper-audit",
        description="Check a design campaign for known failure modes.")
    p.add_argument("csv", type=Path, help="file with one row per design")
    p.add_argument("--score", required=True,
                   help="column holding the confidence score you rank by")
    p.add_argument("--outcome", required=True,
                   help="column holding the experimental result")
    p.add_argument("--group", default=None,
                   help="column identifying the target, to audit each separately")
    p.add_argument("--lower-is-better", action="store_true",
                   help="set for scores like interface PAE where low is good")
    p.add_argument("--positive", default=None,
                   help="value of --outcome that means 'bound'; inferred if omitted")
    p.add_argument("--wells", type=int, default=None,
                   help="how many designs you have actually assayed so far. "
                        "Defaults to every labelled row.")
    args = p.parse_args(argv)
    _console_setup()

    df = _read_csv(args.csv)
    for col in (args.score, args.outcome, *( [args.group] if args.group else [])):
        if col not in df.columns:
            print(f"column {col!r} not in {args.csv.name}. Available:\n  "
                  + ", ".join(map(str, df.columns[:40])), file=sys.stderr)
            return 2

    import pandas as pd
    y_all, ok = _to_binary(df[args.outcome], args.positive)
    v = pd.to_numeric(df[args.score], errors="coerce")
    keep = ok & v.notna()
    dropped = int((~keep).sum())
    df = df[keep].copy()
    df["_y"] = y_all[keep].to_numpy()
    df["_s"] = (-v[keep] if args.lower_is_better else v[keep]).to_numpy()

    if df.empty:
        print("nothing left after dropping rows with a missing score or outcome",
              file=sys.stderr)
        return 1

    groups = ([(g, sub) for g, sub in df.groupby(args.group)]
              if args.group else [(None, df)])

    print(f"{args.csv.name}: {len(df):,} usable designs"
          + (f" ({dropped} dropped for missing score/outcome)" if dropped else "")
          + (f", {len(groups)} groups" if args.group else ""))

    exit_code = 0
    for name, sub in groups:
        s = sub["_s"].to_numpy(float)
        y = sub["_y"].to_numpy(float)
        if len(set(y.tolist())) < 2:
            print(f"\n--- {name or args.score} ---")
            print("  only one outcome class here; nothing to check.")
            continue

        n_wells = min(args.wells, len(s)) if args.wells else len(s)
        order = np.argsort(-s, kind="mergesort")[:n_wells]

        header = f"{name} | " if name is not None else ""
        print(f"\n--- {header}{len(sub):,} designs, "
              f"{int(y.sum())} bound ({100 * y.mean():.1f}%), "
              f"AUC {roc_auc(s, y):.3f} ---")

        rep = audit(scores=s[order], outcomes=y[order], all_scores=s,
                    n_target_wells=n_wells)
        print(rep)
        if rep.blockers:
            exit_code = 1

    if exit_code:
        print("\nexit 1: at least one blocking finding. See above.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
