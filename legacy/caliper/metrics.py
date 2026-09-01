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


# ---------------------------------------------------------------------------
# Shortlist-quality metrics
#
# ``topk_recall`` is a knife-edge set-membership metric: a design ranked 25th
# out of 3,000 scores exactly the same as one ranked 3,000th.  Measured on this
# harness it returned 0.042 for BOTH a 3-stage cascade and a single-stage sweep,
# while the mean true quality of those same shortlists was 0.785 versus 0.710 --
# a large, consistent difference the recall metric threw away.
#
# Set-overlap on a 24-of-3,000 selection is also extremely high variance, which
# is why a real effect failed a significance test at n=12 seeds.  The metrics
# below measure the same thing with far less noise.
#
# Korean note:
# 상위24 "집합 일치"는 25등과 3000등을 똑같이 취급한다.  그래서 실제로 갈리는
# 두 정책이 똑같이 0.042 로 나왔다.  평균 품질과 상위분위 포함률로 재면 명확히 갈린다.
# ---------------------------------------------------------------------------
def mean_quality(kept_idx, truth) -> float:
    """Mean ground-truth quality of the selected shortlist."""
    t = np.asarray(truth, dtype=float)
    if len(kept_idx) == 0:
        return float("nan")
    return float(t[np.asarray(kept_idx, dtype=int)].mean())


def top_decile_rate(kept_idx, truth, q: float = 90.0) -> float:
    """Fraction of the shortlist that is in the true top (100-q)% of the pool.

    Less brittle than exact top-k membership and directly interpretable:
    "what share of the wells I spend are on genuinely good designs?"
    """
    t = np.asarray(truth, dtype=float)
    if len(kept_idx) == 0 or t.size == 0:
        return float("nan")
    cut = np.percentile(t, q)
    return float((t[np.asarray(kept_idx, dtype=int)] >= cut).mean())


def normalised_quality(kept_idx, truth, n_final: int) -> float:
    """Shortlist quality on a 0-1 scale where 0 = random pool, 1 = oracle.

    Absolute quality numbers are not comparable across seeds because the pools
    differ.  Normalising against the achievable range makes them so, and makes
    "how much of the available gain did this policy capture?" a direct read.
    """
    t = np.asarray(truth, dtype=float)
    if len(kept_idx) == 0 or t.size == 0:
        return float("nan")
    best = float(np.sort(t)[-n_final:].mean())
    base = float(t.mean())
    got = mean_quality(kept_idx, t)
    return float("nan") if best == base else (got - base) / (best - base)


def roc_auc(scores, labels) -> float:
    """ROC AUC via the Mann-Whitney statistic, ties counted as half.

    Lived in ``backends/simulator`` until five real-data experiments and
    ``multiround`` were all importing it from there.  A general ranking metric
    has no business being reachable only through the test harness, and having
    real-data code depend on the simulator module was the wrong shape.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1, dtype=float)
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
