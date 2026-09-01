"""Pull one table out of a 139 MB supplementary zip without downloading it all.

Bennett et al. 2023 ship their retrospective analysis inside Supplementary Data
4, a zip that is mostly PDB models. The one file worth having is a 100 MB score
table: 603,178 designs across 10 targets, each with AF2 and RF2 confidence
scores and an experimental binding readout.

Downloading 139 MB to read 24 MB of it is rude to whoever is on a slow link, so
this reads the zip the way a zip is meant to be read: fetch the central
directory from the end of the file with an HTTP range request, look up where
the member starts, fetch only those bytes, and inflate them locally. Then throw
away every column the experiments do not use, which turns 100 MB of text into
about 40 MB on disk.

If the server refuses range requests this falls back to streaming the whole zip
through Python's zipfile, which works but costs the full 139 MB.

Korean note:
139MB 짜리 보충자료 zip 안에서 필요한 표 하나만 꺼낸다.
zip은 파일 끝에 "목차"(central directory)가 있어서, 그 목차만 먼저 받아보면
원하는 파일이 몇 번째 바이트에 있는지 알 수 있다. 그 구간만 HTTP 범위 요청으로
받아서 압축을 푼다. 전체를 받지 않아도 된다.
쓰지 않는 열은 버려서 100MB 텍스트를 40MB 정도로 줄인다.
"""

from __future__ import annotations

import io
import struct
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "bennett" / "retrospective.csv"

ZIP_URL = ("https://static-content.springer.com/esm/"
           "art%3A10.1038%2Fs41467-023-38328-5/MediaObjects/"
           "41467_2023_38328_MOESM4_ESM.zip")
MEMBER = "all_data/retrospective_analysis/retrospective_analysis_more_scores.sc"

# what the experiments actually read. "Rosetta ddG" has a space in it, which is
# why this list is spelled out rather than inferred.
KEEP = ["description", "target", "avid_ub", "avid_lb", "kd_ub", "kd_lb",
        "pAE_interaction", "RF2_pAE_interaction", "pAE_interaction_no_guess",
        "AF2_plddt_monomer", "RF2_plddt_monomer", "AF2_complex_RMSD",
        "DAN_interface_lddt", "Rosetta ddG"]

TAIL_BYTES = 1_500_000     # enough to hold the central directory of this zip


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    elif end is not None:
        headers["Range"] = f"bytes=-{end}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _find_member(tail: bytes, member: str) -> tuple[int, int, int] | None:
    """Return (local header offset, compressed size, method) for `member`.

    Walks the central-directory records in the tail of the zip. Each starts
    with the signature PK\\x01\\x02; the fields we need sit at fixed offsets.
    """
    want = member.encode()
    pos = 0
    while True:
        pos = tail.find(b"PK\x01\x02", pos)
        if pos < 0:
            return None
        try:
            method = struct.unpack_from("<H", tail, pos + 10)[0]
            csize = struct.unpack_from("<I", tail, pos + 20)[0]
            nlen = struct.unpack_from("<H", tail, pos + 28)[0]
            offset = struct.unpack_from("<I", tail, pos + 42)[0]
            name = tail[pos + 46:pos + 46 + nlen]
        except struct.error:
            pos += 4
            continue
        if name == want:
            return offset, csize, method
        pos += 4


def _by_range() -> bytes | None:
    """Fetch and inflate just the member. Returns None if ranges are refused."""
    try:
        tail = _get(ZIP_URL, end=TAIL_BYTES)
    except (urllib.error.URLError, OSError) as e:
        print(f"  could not read the end of the zip: {e}", file=sys.stderr)
        return None
    if len(tail) >= 100_000_000:
        return None                      # server ignored the range, sent it all

    found = _find_member(tail, MEMBER)
    if not found:
        print("  the zip's layout changed: expected member not in its index.",
              file=sys.stderr)
        return None
    offset, csize, method = found

    # the local header repeats the name and extra-field lengths; read them so we
    # know where the compressed bytes actually start.
    head = _get(ZIP_URL, offset, offset + 29)
    nlen, elen = struct.unpack_from("<HH", head, 26)
    start = offset + 30 + nlen + elen

    print(f"  fetching {csize / 1e6:.0f} MB of a {139} MB archive ...", flush=True)
    raw = _get(ZIP_URL, start, start + csize - 1)
    if len(raw) != csize:
        print(f"  short read: {len(raw)} of {csize} bytes", file=sys.stderr)
        return None
    return zlib.decompress(raw, -15) if method == 8 else raw


def _whole_zip() -> bytes:
    import zipfile
    print("  range requests refused; downloading the whole 139 MB archive ...",
          flush=True)
    blob = io.BytesIO(_get(ZIP_URL))
    with zipfile.ZipFile(blob) as z:
        return z.read(MEMBER)


def main() -> int:
    try:
        import pandas as pd
    except ImportError:
        print("this needs pandas:  pip install pandas", file=sys.stderr)
        return 2

    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"  bennett: already here ({DEST.stat().st_size / 1e6:.0f} MB)")
        return 0

    print("Bennett et al. 2023 -- 603,178 designs, 10 targets (CC-BY-4.0)")
    text = _by_range()
    if text is None:
        try:
            text = _whole_zip()
        except Exception as e:                       # noqa: BLE001 - report and stop
            print(f"  FAILED -- {e}", file=sys.stderr)
            print(f"  download it by hand from:\n  {ZIP_URL}\n"
                  f"  then extract {MEMBER}", file=sys.stderr)
            return 1

    df = pd.read_csv(io.BytesIO(text), low_memory=False)
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        print(f"  MISSING COLUMNS {missing}. The file's layout changed.",
              file=sys.stderr)
        return 1
    if len(df) < 500_000:
        print(f"  only {len(df):,} rows, expected about 603,000. Truncated?",
              file=sys.stderr)
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    df[KEEP].to_csv(tmp, index=False)
    tmp.replace(DEST)
    print(f"  bennett: {len(df):,} rows, {df.target.nunique()} targets, "
          f"{DEST.stat().st_size / 1e6:.0f} MB -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
