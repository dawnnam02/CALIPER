"""Does a second round, informed by the first, actually find more binders?

Setup, on real data, leave-one-target-out
-----------------------------------------
Round 1 selects 24 designs for a held-out target using the metric that was best
across the OTHER 14 targets -- the only choice available before any local data
exists.  Those 24 outcomes are then revealed, and round 2 selects another 24
from what is left, under four policies:

  static        keep ranking by the same global metric (no learning)
  switch        rank by whichever metric had the best AUC on the revealed 24
  guarded       switch only if the lead clears one standard error of the AUC
                estimate (caliper/multiround.py)
  oracle        rank by the metric that is truly best on this target -- an
                upper bound, not achievable in practice

If `static` wins, round-1 labels are not worth acting on and the multi-round
machinery should not be built.  If `switch` beats `static`, naive adaptation
works.  If `guarded` beats `switch`, the winner's curse is real and the margin
is what makes adaptation safe.

Data: Overath et al. 2025, Zenodo 10.5281/zenodo.15722219, CC-BY-4.0
Run:  python experiments/multiround.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.metrics import roc_auc
from caliper.multiround import choose_metric, next_batch
from caliper.stats import bootstrap_ci, paired_bootstrap
from caliper.types import stable_hash

DATA = Path(__file__).resolve().parents[2] / "data" / "overath" / "final_dataset.csv"

# candidate ranking metrics; True means lower is better
CANDIDATES = {
    "af3_ipSAE_min": False, "af3_LIS": False, "af3_iptm_avg": False,
    "af3_ipae": True,
    "boltz1_ipSAE_min": False, "boltz1_LIS": False, "boltz1_iptm_avg": False,
    "colab_ipSAE_min": False, "colab_LIS": False, "colab_iptm_avg": False,
    "af2_pae_interaction": True, "af2_plddt_binder": False,
}
BATCH = 24
N_REPEATS = 12          # round-1 batches are deterministic, so repeats only
                        # vary the tie-breaking jitter; kept small on purpose
MIN_N = 90


def load() -> pd.DataFrame:
    if not DATA.exists():
        print(f"dataset not found: {DATA}", file=sys.stderr)
        raise SystemExit(2)
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df.binder.notna()].copy()
    df["y"] = df.binder.astype(bool).astype(int)
    for c in CANDIDATES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=list(CANDIDATES)).reset_index(drop=True)


def oriented(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for c, lower_better in CANDIDATES.items():
        v = frame[c].to_numpy(dtype=float)
        out[c] = -v if lower_better else v
    return out


def main() -> int:
    df = load()
    targets = sorted(df.target_id.unique())

    print("=" * 78)
    print("Round 2 informed by round 1 | Overath et al. 2025")
    print("=" * 78)
    print(f"  {len(df):,} designs | {len(targets)} targets | batch {BATCH} | "
          f"{len(CANDIDATES)} candidate metrics")

    res: dict = defaultdict(list)
    switch_log: list = []

    for held in targets:
        te = df[df.target_id == held]
        tr = df[df.target_id != held]
        if len(te) < MIN_N or te.y.sum() < 6:
            continue

        s_te, s_tr = oriented(te), oriented(tr)
        y_te, y_tr = te.y.to_numpy(), tr.y.to_numpy()
        n = len(te)

        # incumbent = best metric on the OTHER targets (no leakage)
        incumbent = max(CANDIDATES, key=lambda c: roc_auc(s_tr[c], y_tr))
        # oracle = truly best on this target (upper bound only)
        oracle = max(CANDIDATES, key=lambda c: roc_auc(s_te[c], y_te))

        for rep in range(N_REPEATS):
            rng = np.random.default_rng(
                int(stable_hash([held, rep]), 16) % 2**31)
            # tiny jitter breaks ties differently per repeat
            jitter = {c: v + rng.normal(0, 1e-9, n) for c, v in s_te.items()}

            r1 = next_batch(jitter[incumbent], set(), BATCH)
            done = set(int(i) for i in r1)
            revealed = {c: jitter[c][r1] for c in CANDIDATES}
            y1 = y_te[r1]

            picked = choose_metric(revealed, y1, incumbent)
            naive = max(CANDIDATES,
                        key=lambda c: (roc_auc(revealed[c], y1)
                                       if len(set(y1.tolist())) > 1 else -1))

            batches = {
                "static": next_batch(jitter[incumbent], done, BATCH),
                "switch": next_batch(jitter[naive], done, BATCH),
                "guarded": next_batch(jitter[picked.metric], done, BATCH),
                "oracle": next_batch(jitter[oracle], done, BATCH),
            }
            res["round1"].append(float(y1.mean()))
            for name, b in batches.items():
                res[name].append(float(y_te[b].mean()) if len(b) else 0.0)
            switch_log.append({
                "target": held, "rep": rep, "incumbent": incumbent,
                "naive_pick": naive, "guarded_pick": picked.metric,
                "guarded_switched": picked.switched, "oracle": oracle,
                "n_pos_revealed": int(y1.sum()),
            })

    print()
    print("Hit rate of the SECOND batch of 24 (mean [95% bootstrap CI])")
    print(f"  {'policy':<14}{'hit rate':>28}")
    print("  " + "-" * 42)
    print(f"  {'round 1':<14}{str(bootstrap_ci(res['round1'], seed=1)):>28}"
          "   <- for reference")
    for name in ("static", "switch", "guarded", "oracle"):
        print(f"  {name:<14}{str(bootstrap_ci(res[name], seed=1)):>28}")

    print()
    print("Paired tests against 'static' (no learning)")
    for name in ("switch", "guarded", "oracle"):
        t = paired_bootstrap(res[name], res["static"], name, "static", seed=6)
        print(f"  {t.verdict()}")
    print()
    t = paired_bootstrap(res["guarded"], res["switch"], "guarded", "switch", seed=6)
    print(f"  {t.verdict()}")

    L = pd.DataFrame(switch_log)
    print()
    print("How often does each rule change the metric?")
    print(f"  naive argmax switched : "
          f"{(L.naive_pick != L.incumbent).mean() * 100:5.1f}% of rounds")
    print(f"  guarded switched      : "
          f"{L.guarded_switched.mean() * 100:5.1f}% of rounds")
    print(f"  and was right to      : "
          f"{((L.guarded_switched) & (L.guarded_pick == L.oracle)).sum()} of "
          f"{int(L.guarded_switched.sum())} switches picked the true best")
    print(f"  naive picked the true best in "
          f"{(L.naive_pick == L.oracle).mean() * 100:5.1f}% of rounds")
    print(f"  positives revealed in round 1: median "
          f"{L.n_pos_revealed.median():.0f} of {BATCH}")

    p = Path(__file__).resolve().parent / "multiround_results.json"
    p.write_text(json.dumps(
        {"hit_rates": {k: [float(x) for x in v] for k, v in res.items()},
         "switches": switch_log}, indent=2, default=str), encoding="utf-8")
    print(f"\nraw results -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
