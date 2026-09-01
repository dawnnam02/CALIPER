"""Did the ladder actually keep the good ones?

These are scheduler metrics, not chemistry metrics.  They answer: given that
we spent C units of compute instead of the brute-force cost, how much of the
true top-k did we retain?

Korean note:
"돈을 얼마나 아꼈나"만 보면 안 된다.  아끼면서 진짜 좋은 걸 놓쳤으면 손해다.
그래서 절감률과 회수율(recall)을 항상 같이 본다.
"""

from __future__ import annotations

import numpy as np


def topk_recall(kept_ids: list[str], truth: dict[str, float], k: int) -> float:
    """Fraction of the true top-k that survived.

    ``truth`` maps candidate id -> ground-truth quality.  Only meaningful in
    simulation, where ground truth exists; in a real campaign this is reported
    as ``None`` rather than guessed.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not truth:
        return float("nan")
    k = min(k, len(truth))
    best = {cid for cid, _ in sorted(truth.items(), key=lambda kv: -kv[1])[:k]}
    if not best:
        return float("nan")
    return len(best & set(kept_ids)) / len(best)


def enrichment(kept_scores: np.ndarray, all_scores: np.ndarray) -> float:
    """Mean quality of survivors divided by mean quality of the whole pool.

    1.0 means the filter did nothing.  Below 1.0 means it actively hurt.
    """
    a = np.asarray(all_scores, dtype=float)
    k = np.asarray(kept_scores, dtype=float)
    if a.size == 0 or k.size == 0 or a.mean() == 0:
        return float("nan")
    return float(k.mean() / a.mean())


def hit_rate(outcomes) -> float:
    o = np.asarray(outcomes, dtype=float)
    return float(o.mean()) if o.size else float("nan")


def spearman(a, b) -> float:
    """Rank correlation without scipy.stats, ties handled by average ranks."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or a.size != b.size:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def cost_summary(reports) -> dict:
    total = sum(r.cost_units for r in reports)
    wall = sum(r.wall_seconds for r in reports)
    return {
        "total_cost_units": float(total),
        "wall_seconds": float(wall),
        "by_stage": {r.stage: float(r.cost_units) for r in reports},
        "cache_hits": int(sum(r.cache_hits for r in reports)),
    }
