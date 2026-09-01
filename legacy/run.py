"""CALIPER 실행 진입점.

위에서 아래로 읽힌다. 바꿀 값은 전부 config.yaml 에 있고 이 파일은 손대지 않아도 된다.

    python run.py
    python run.py 내설정.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from caliper.harness.backends.simulator import (SimAssay, SimDesigner, SimScorer,
                                        noise_for_auc, true_affinity)
from caliper.metrics import topk_recall
from caliper.harness.pipeline import Campaign
from caliper.harness.report import write_report
from caliper.harness.store import RunDir, Store
from caliper.types import Target


def main(config_path: str = "config.yaml") -> int:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"설정 파일이 없다: {cfg_file}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

    # ---- 표적 ----------------------------------------------------------
    tcfg = cfg["target"]
    target = Target(
        name=tcfg["name"],
        sequence=tcfg["sequence"].strip().upper(),
        hotspots=tuple(tcfg.get("hotspots", [])),
    )

    # ---- 백엔드 --------------------------------------------------------
    backend = cfg.get("backend", "simulator")
    if backend != "simulator":
        from caliper.harness.backends.external import build_external
        designer, scorers, assay = build_external(cfg, target)
    else:
        scfg = cfg["simulator"]
        designer = SimDesigner(tuple(scfg.get("designer_length", (40, 80))))

        # 시뮬레이터를 문헌에 맞춘다:
        #  1) 실험 기저 성공률을 목표값에 맞도록 assay 임계값을 푼다
        #  2) 각 단계의 잡음을 "발표된 ROC AUC를 재현하는 값"으로 역산한다
        probe_c = designer.design(target, scfg.get("probe_n", 3000), seed=99)
        probe = [c.sequence for c in probe_c]
        assay = SimAssay.for_base_rate(
            target, probe, scfg.get("base_rate", 0.116),
            unit_cost=scfg.get("assay_unit_cost", 500.0))
        probe_y = assay.run(target, probe_c, seed=99)

        scorers = []
        for st in scfg["stages"]:
            gain, bias = st.get("gain", 1.0), st.get("bias", 0.0)
            if "auc" in st:
                noise = noise_for_auc(target, probe, probe_y, st["auc"],
                                      gain=gain, bias=bias)
            else:
                noise = st["noise"]
            scorers.append(SimScorer(st["stage"], noise=noise,
                                     unit_cost=st["unit_cost"],
                                     bias=bias, gain=gain))

    # ---- 실행 ----------------------------------------------------------
    rcfg = cfg["run"]
    ocfg = cfg.get("output", {})
    run_id = time.strftime("%Y%m%d-%H%M%S")
    store = Store(ocfg.get("cache_dir", ".caliper_cache"),
                  enabled=ocfg.get("use_cache", True))
    rundir = RunDir(ocfg.get("runs_dir", "runs"), run_id)

    n_start = rcfg["n_start"]
    if rcfg.get("budget"):
        from caliper.harness.allocate import budget_to_start
        n_start = budget_to_start(
            float(rcfg["budget"]),
            [s.stage for s in scorers], [s.unit_cost for s in scorers],
            reduction=rcfg.get("reduction", 3.0),
            n_final=rcfg.get("n_final", 8))
        print(f"예산 {rcfg['budget']:,} → 시작 후보 수 {n_start:,} 로 역산")
        if n_start == 0:
            print("예산이 너무 작아 후보 하나도 못 돌린다.", file=sys.stderr)
            return 1

    camp = Campaign(target, designer, scorers, assay,
                    store=store, rundir=rundir,
                    explore_fraction=rcfg.get("explore_fraction", 0.25),
                    seed=rcfg.get("seed", 0))
    result = camp.run(n_start,
                      reduction=rcfg.get("reduction", 3.0),
                      n_final=rcfg.get("n_final", 8),
                      assay_capacity=rcfg.get("assay_capacity"))

    # ---- 시뮬레이터일 때만 정답 대조 -----------------------------------
    if backend == "simulator":
        truth = {c.cid: true_affinity(target, c.sequence) for c in result.candidates}
        result.diagnostics["topk_recall"] = topk_recall(
            [c.cid for c in result.shortlist], truth, rcfg.get("n_final", 8))
        result.diagnostics["ground_truth_available"] = True
    else:
        result.diagnostics["ground_truth_available"] = False

    paths = write_report(result, rundir)
    print()
    print(result.diagnostics["ladder"])
    print()
    d = result.diagnostics
    print(f"설계 {d['n_designed']:,} → 최종후보 {d['n_shortlist']} → 실험 {d['n_assayed']}"
          f" (탐색 {d['n_explore']})")
    if d.get("hit_rate_shortlist") is not None:
        print(f"적중률: 상위후보 {100*d['hit_rate_shortlist']:.1f}%", end="")
        if d.get("hit_rate_explore") is not None:
            print(f" | 탐색표본 {100*d['hit_rate_explore']:.1f}%", end="")
        print()
    if "ece_calibrated" in d:
        print(f"교정 ECE: {d['ece_raw']:.4f} → {d['ece_calibrated']:.4f}")
    if d.get("topk_recall") is not None:
        print(f"진짜 상위 {rcfg.get('n_final', 8)}개 회수율: {100*d['topk_recall']:.1f}%")
    print(f"총 비용: {d['cost']['total_cost_units']:,.0f} 단위")
    print()
    for p in paths:
        print("결과 →", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
