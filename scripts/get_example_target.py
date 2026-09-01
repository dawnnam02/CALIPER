"""예제 표적 구조를 받는다 (MDM2, PDB 1YCR 사슬 A).

    python scripts/get_example_target.py

왜 MDM2 인가:
  - p53 이 꽂히는 소수성 주머니가 잘 알려져 있어 핫스팟 예시로 좋다
  - 구조가 작아서(85 잔기) 빠르다
  - 이 저장소의 검증 데이터(Overath)에도 Mdm2 표적이 들어 있어서
    "정석 필터가 이 표적에서 어떻게 작동했는지"를 바로 볼 수 있다
"""
import sys, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DEST = Path(__file__).resolve().parents[1] / "data" / "targets" / "1ycr.pdb"
URL = "https://files.rcsb.org/download/1YCR.pdb"

if DEST.exists() and DEST.stat().st_size > 1000:
    print(f"  이미 있다: {DEST}")
    raise SystemExit(0)

DEST.parent.mkdir(parents=True, exist_ok=True)
print(f"  받는 중: {URL}")
try:
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
except Exception as e:
    print(f"  실패: {e}", file=sys.stderr)
    print(f"  직접 받아서 {DEST} 에 두면 된다.", file=sys.stderr)
    raise SystemExit(1)

if b"ATOM" not in data:
    print("  받은 파일에 ATOM 줄이 없다. 주소가 바뀐 것 같다.", file=sys.stderr)
    raise SystemExit(1)

DEST.write_bytes(data)
print(f"  저장: {DEST}  ({len(data)/1000:.0f} KB)")
print("  다음:  python pipeline/step1_target.py")
