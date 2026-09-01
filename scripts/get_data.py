"""Fetch the two public datasets the real-data experiments need.

    python scripts/get_data.py            # all three
    python scripts/get_data.py adaptyv    # just the small one (0.2 MB)
    python scripts/get_data.py adaptyv overath

None of the files are committed: they are large and they belong to their
authors. This
downloads them to the paths the experiments expect and checks that what arrived
looks like what was expected, so a truncated download fails here rather than
three experiments later.

Korean note:
데이터는 커밋돼 있지 않다.  하나는 82MB이고, 둘 다 원저자 것이다.
이 스크립트가 실험이 기대하는 위치로 받아오고, 받은 게 예상과 맞는지 확인한다.
중간에 끊긴 다운로드는 여기서 걸려야지 실험 세 개 뒤에서 터지면 곤란하다.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))        # so fetch_bennett imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caliper.console import setup as _console_setup

_console_setup()

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "overath": {
        "url": ("https://zenodo.org/api/records/15722219/files/"
                "final_dataset.csv/content"),
        "dest": ROOT / "data" / "overath" / "final_dataset.csv",
        "about": "Overath et al. 2025 -- 3,650 designs, 15 targets (CC-BY-4.0)",
        "size_mb": 82,
        "expect_columns": ["binder_id", "target_id", "binder", "af3_ipSAE_min"],
        "min_rows": 3000,
    },
    "adaptyv": {
        "url": ("https://raw.githubusercontent.com/adaptyvbio/"
                "egfr_competition_2/main/results/result_summary.csv"),
        "dest": ROOT / "data" / "adaptyv" / "round2.csv",
        "about": "Adaptyv EGFR competition round 2 -- 402 designs (ODbL)",
        "size_mb": 0.24,
        "expect_columns": ["binding", "iptm", "plddt", "pae_interaction"],
        "min_rows": 300,
    },
    "bindcraft": {
        "url": ("https://static-content.springer.com/esm/"
                "art%3A10.1038%2Fs41586-025-09429-6/MediaObjects/"
                "41586_2025_9429_MOESM4_ESM.csv"),
        "dest": ROOT / "data" / "bindcraft" / "screening.csv",
        "about": ("Pacesa et al. 2025 BindCraft -- 212 designs, 13 targets "
                  "(CC BY-NC-ND-4.0; the most restrictive licence here)"),
        "size_mb": 0.28,
        "expect_columns": ["Target", "Binding", "Average_i_pTM", "Average_i_pAE"],
        "min_rows": 200,
    },
}


def download(name: str, spec: dict) -> bool:
    dest: Path = spec["dest"]
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  {name}: already here ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {name}: downloading ~{spec['size_mb']} MB ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(spec["url"], timeout=120) as r, \
                tmp.open("wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        tmp.unlink(missing_ok=True)
        print(f"  {name}: FAILED -- {e}", file=sys.stderr)
        print(f"         download it by hand from:\n         {spec['url']}",
              file=sys.stderr)
        return False

    tmp.replace(dest)
    print(f"  {name}: {dest.stat().st_size / 1e6:.1f} MB -> {dest}")
    return True


def verify(name: str, spec: dict) -> bool:
    """Check the file is the one we meant, not an error page or half a file."""
    dest: Path = spec["dest"]
    if not dest.exists():
        return False
    try:
        header = dest.open(encoding="utf-8", errors="replace").readline()
    except OSError as e:
        print(f"  {name}: cannot read -- {e}", file=sys.stderr)
        return False
    missing = [c for c in spec["expect_columns"] if c not in header]
    if missing:
        print(f"  {name}: MISSING COLUMNS {missing}. The file is probably an "
              "error page or the source layout changed.", file=sys.stderr)
        return False
    with dest.open(encoding="utf-8", errors="replace") as fh:
        rows = sum(1 for _ in fh) - 1
    if rows < spec["min_rows"]:
        print(f"  {name}: only {rows} rows, expected at least "
              f"{spec['min_rows']}. Download was likely truncated.",
              file=sys.stderr)
        return False
    print(f"  {name}: verified -- {rows:,} rows, expected columns present")
    return True


def fetch_proteinbase(which: str) -> bool:
    """Delegate: these two need their JSON evaluations column flattened."""
    import fetch_proteinbase as fp
    return fp.main(["", which]) == 0


def fetch_bennett() -> bool:
    """Delegate to the sibling script: this one lives inside a 139 MB zip."""
    import fetch_bennett as fb
    return fb.main() == 0


def main(argv: list[str]) -> int:
    known = list(DATASETS) + ["bennett", "nipah", "rbx1"]
    wanted = argv[1:] or known
    unknown = [w for w in wanted if w not in known]
    if unknown:
        print(f"unknown dataset(s): {unknown}. Choose from {known}",
              file=sys.stderr)
        return 2

    print("CALIPER -- fetching public datasets")
    for name in wanted:
        if name in ("nipah", "rbx1"):
            if not fetch_proteinbase(name):
                return 1
            continue
        if name == "bennett":
            print()
            if not fetch_bennett():
                return 1
            continue
        print(f"\n{DATASETS[name]['about']}")
        if not download(name, DATASETS[name]):
            return 1
        if not verify(name, DATASETS[name]):
            return 1

    print("\nready. Now try:")
    print("  python experiments/detector.py      # the headline result")
    print("  pytest                              # 53 tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
