"""Run report: markdown, CSV, and a dependency-free SVG reliability diagram.

Every reported number is placed against published values from
``caliper.benchmarks`` so a run says whether its result is normal, good, or
suspicious, rather than leaving a bare figure on the screen.

Korean note:
숫자만 찍으면 그게 좋은 건지 나쁜 건지 모른다.  그래서 발표된 수치와 나란히 놓는다.
matplotlib이 없어도 되도록 SVG를 직접 그린다.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import benchmarks as bm
from .store import RunDir


def reliability_svg(rows: list[dict], width: int = 460, height: int = 460) -> str:
    """Reliability diagram as raw SVG.  No plotting library required."""
    pad = 55
    w = width - 2 * pad
    h = height - 2 * pad

    def X(v: float) -> float:
        return pad + v * w

    def Y(v: float) -> float:
        return height - pad - v * h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<style>'
        'text{font:11px system-ui,sans-serif;fill:#333}'
        '.t{font-size:13px;font-weight:600}'
        '.g{stroke:#e5e5e5;stroke-width:1}'
        '.ax{stroke:#666;stroke-width:1.2}'
        '</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="t" x="{pad}" y="26">Reliability: predicted vs observed</text>',
    ]
    for i in range(11):
        v = i / 10
        parts.append(f'<line class="g" x1="{X(v):.1f}" y1="{Y(0):.1f}" '
                     f'x2="{X(v):.1f}" y2="{Y(1):.1f}"/>')
        parts.append(f'<line class="g" x1="{X(0):.1f}" y1="{Y(v):.1f}" '
                     f'x2="{X(1):.1f}" y2="{Y(v):.1f}"/>')
    # perfect-calibration diagonal
    parts.append(f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" '
                 f'y2="{Y(1):.1f}" stroke="#bbb" stroke-dasharray="5 4"/>')
    parts.append(f'<line class="ax" x1="{X(0):.1f}" y1="{Y(0):.1f}" '
                 f'x2="{X(1):.1f}" y2="{Y(0):.1f}"/>')
    parts.append(f'<line class="ax" x1="{X(0):.1f}" y1="{Y(0):.1f}" '
                 f'x2="{X(0):.1f}" y2="{Y(1):.1f}"/>')

    pts = [(r["predicted"], r["observed"], r["n"]) for r in rows
           if r.get("n") and r.get("predicted") is not None]
    if pts:
        d = " ".join(f'{"M" if i == 0 else "L"}{X(p):.1f},{Y(o):.1f}'
                     for i, (p, o, _) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="#d33" stroke-width="2"/>')
        nmax = max(n for _, _, n in pts)
        for p, o, n in pts:
            r = 3 + 6 * (n / nmax) ** 0.5
            parts.append(f'<circle cx="{X(p):.1f}" cy="{Y(o):.1f}" r="{r:.1f}" '
                         f'fill="#d33" fill-opacity="0.65"/>')
    else:
        parts.append(f'<text x="{X(0.5):.1f}" y="{Y(0.5):.1f}" '
                     f'text-anchor="middle" fill="#999">no labelled data</text>')

    for i in range(0, 11, 2):
        v = i / 10
        parts.append(f'<text x="{X(v):.1f}" y="{height - pad + 18:.1f}" '
                     f'text-anchor="middle">{v:.1f}</text>')
        parts.append(f'<text x="{pad - 10:.1f}" y="{Y(v) + 4:.1f}" '
                     f'text-anchor="end">{v:.1f}</text>')
    parts.append(f'<text x="{width/2:.0f}" y="{height - 10}" '
                 f'text-anchor="middle">predicted probability</text>')
    parts.append(f'<text transform="translate(16,{height/2:.0f}) rotate(-90)" '
                 f'text-anchor="middle">observed frequency</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_report(result, rundir: RunDir) -> list[Path]:
    d = result.diagnostics
    out: list[Path] = []

    # --- candidates.csv ---------------------------------------------------
    rows = [c.as_row() for c in result.candidates]
    if rows:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        assayed = {c.cid for c in result.assayed}
        for r in rows:
            r["assayed"] = r["cid"] in assayed
            r["outcome"] = result.outcomes.get(r["cid"])
        keys += ["assayed", "outcome"]
        p = rundir.path / "candidates.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        out.append(p)

    # --- reliability.svg --------------------------------------------------
    if d.get("reliability"):
        out.append(rundir.write_text("reliability.svg",
                                     reliability_svg(d["reliability"])))

    # --- diagnostics.json -------------------------------------------------
    out.append(rundir.write_json("diagnostics.json",
                                 {k: v for k, v in d.items() if k != "ladder"}))

    # --- report.md --------------------------------------------------------
    L: list[str] = ["# CALIPER run report", "", f"Target: **{result.target}**", ""]
    L += ["## Ladder", "", "```", d["ladder"], "```", ""]

    L += ["## Outcome", "",
          "| quantity | value | vs published |",
          "|---|---|---|"]
    hr = d.get("hit_rate_shortlist")
    if hr is not None:
        L.append(f"| hit rate, shortlist | {hr:.3f} | {bm.compare(hr, bm.HIT_RATES)} |")
    ex = d.get("hit_rate_explore")
    if ex is not None:
        L.append(f"| hit rate, exploration sample | {ex:.3f} | baseline for the "
                 "population the filter rejected |")
    rc = d.get("topk_recall")
    if rc is not None:
        loss = 1.0 - rc
        L.append(f"| true top-k retained | {rc:.3f} | "
                 f"{bm.compare(loss, bm.RECALL_LOSS)} (as loss={loss:.3f}) |")
    L.append(f"| designs made | {d['n_designed']:,} | |")
    L.append(f"| assayed | {d['n_assayed']} "
             f"({d['n_explore']} exploration) | |")
    L.append(f"| compute cost | {d['cost']['total_cost_units']:,.0f} units | |")
    L.append("")

    if "ece_calibrated" in d:
        L += ["## Calibration", "",
              f"- ECE, raw score as probability: **{d['ece_raw']:.4f}**",
              f"- ECE, after calibration: **{d['ece_calibrated']:.4f}**",
              f"- Brier, raw / calibrated: {d['brier_raw']:.4f} / "
              f"{d['brier_calibrated']:.4f}",
              f"- labels used: {d['calibrator_labels']}",
              "",
              "![reliability](reliability.svg)",
              ""]

    L += ["## Published comparisons", "",
          "Experimental hit rates:", "", "```", bm.table(bm.HIT_RATES), "```", "",
          "Discrimination (ROC AUC) of confidence metrics:", "",
          "```", bm.table(bm.DISCRIMINATION), "```", "",
          "True positives lost to filtering:", "",
          "```", bm.table(bm.RECALL_LOSS), "```", "",
          "Exploration quota needed to detect selection bias:", "",
          "```", bm.table(bm.EXPLORATION), "```", ""]

    if not d.get("ground_truth_available", False):
        L += ["> **Note.** This run used a real backend, so no ground truth "
              "exists. Recall figures are unavailable and any claim about "
              "'true top-k' is not measurable here.", ""]
    else:
        L += ["> **Note.** This run used the simulator. Ground-truth recall is "
              "exact, but hit rates reflect the simulator's configured base "
              "rate, not a real assay.", ""]

    out.append(rundir.write_text("report.md", "\n".join(L)))
    return out
