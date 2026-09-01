"""Does the shortlist actually contain near-duplicates? Measure before building.

CRITIQUE F2 says the pipeline has no diversity control, so a 24-well plate
could hold one design tested 24 times.  That is a real failure mode.  Whether it
is a real problem HERE is a separate question, and the discipline this project
learned the hard way is to measure before building.

Result: it is not a problem in this data.  All ten targets yield 24 distinct
sequences, mean pairwise identity is 0.114, and exactly one pair across ~2,760
comparisons exceeds 90% identity.

The caveat matters more than the result
---------------------------------------
Overath's designs were pooled from many separate published campaigns, so they
are diverse by construction.  A single real campaign -- one RFdiffusion run
emitting thousands of backbones from one target -- would produce far more
near-duplicates than this.

So this measurement does NOT show the diversity problem is imaginary.  It shows
**this dataset cannot test for it**, and that building a clustering step now
would be building against a problem we cannot yet observe.  The right trigger is
a single-campaign design pool, and until one exists the honest position is to
record the gap rather than fill it blind.

Korean note:
"웰 24개가 같은 서열로 찰 수 있다"는 지적은 타당하지만, 이 데이터에서는 안 일어난다.
Overath 설계가 여러 캠페인에서 모은 것이라 원래 다양하기 때문이다.
즉 이 데이터로는 그 문제를 검증할 수 없다는 뜻이지, 문제가 없다는 뜻이 아니다.
단일 캠페인 데이터가 생기기 전에 클러스터링을 붙이는 건 보이지 않는 적과 싸우는 것이다.

Data: Overath et al. 2025, Zenodo 10.5281/zenodo.15722219, CC-BY-4.0
Run:  python experiments/diversity_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from itertools import combinations

DATA = Path(__file__).resolve().parents[2] / "data" / "overath" / "final_dataset.csv"

def _require(path):
    """Fail with instructions, not a traceback, when the dataset is absent."""
    path = Path(path)
    if not path.exists():
        print("dataset not found: " + str(path), file=sys.stderr)
        print("  python scripts/get_data.py overath", file=sys.stderr)
        print("  (82 MB, CC-BY-4.0, Zenodo 10.5281/zenodo.15722219)",
              file=sys.stderr)
        raise SystemExit(2)
    return path


df=pd.read_csv(_require(DATA),low_memory=False)
df=df[df.binder.notna()].copy(); df["y"]=df.binder.astype(bool).astype(int)
ST=[("af2_pae_interaction",True),("colab_ipSAE_min",False),("af3_ipSAE_min",False)]
for c,_ in ST: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=[c for c,_ in ST]).reset_index(drop=True)
NF,RED=24,3.0
SEQ="A_seq"   # 바인더 서열
print("서열 컬럼 존재:", SEQ in df.columns, "| 결측", int(df[SEQ].isna().sum()))
df=df.dropna(subset=[SEQ]).reset_index(drop=True)

def ident(a,b):
    """길이 다르면 짧은 쪽 기준 최대 정렬 없이 단순 동일비율 (빠른 근사)"""
    if a==b: return 1.0
    n=min(len(a),len(b))
    if n==0: return 0.0
    return sum(1 for i in range(n) if a[i]==b[i])/max(len(a),len(b))

def orient(g,c,low):
    v=g[c].to_numpy(float); return -v if low else v

print()
print("="*78); print("최종 후보 24개가 서로 얼마나 다른가 (실데이터)"); print("="*78)
print("  %-16s %7s %10s %10s %10s %8s"%("표적","n","고유서열","평균동일도","최대동일도",">90% 쌍"))
print("  "+"-"*66)
rows=[]
for t,g in df.groupby("target_id"):
    n=len(g)
    if n<60 or g.y.sum()<3: continue
    S={c:orient(g,c,low) for c,low in ST}
    keeps=[max(NF,int(n/RED)),max(NF,int(n/RED/RED)),NF]; alive=np.arange(n)
    for (c,_),k in zip(ST,keeps): alive=alive[np.argsort(-S[c][alive],kind="mergesort")[:k]]
    seqs=g[SEQ].to_numpy()[alive]
    uniq=len(set(seqs))
    pairs=[ident(a,b) for a,b in combinations(seqs,2)]
    hi=sum(1 for p in pairs if p>0.9)
    rows.append(dict(t=t,n=n,uniq=uniq,mean=np.mean(pairs),mx=max(pairs),hi=hi))
    print("  %-16s %7d %10d %10.3f %10.3f %8d"%(str(t)[:16],n,uniq,np.mean(pairs),max(pairs),hi))
R=pd.DataFrame(rows)
print("  "+"-"*66)
print("  고유서열 24/24 인 표적: %d/%d"%((R.uniq==24).sum(),len(R)))
print("  평균 동일도 전체 평균: %.3f"%R["mean"].mean())
print("  >90%% 동일 쌍이 있는 표적: %d/%d"%((R.hi>0).sum(),len(R)))
print()
# 판정 기준: 중복 서열이 있거나, >90% 쌍이 전체 쌍의 1% 를 넘으면 문제로 본다.
# 쌍 하나로 경보를 울리면 실제 문제와 잡음을 구분하지 못한다.
total_pairs = len(R) * (NF * (NF - 1) // 2)
dup_rate = R.hi.sum() / max(total_pairs, 1)
print("  >90%% 쌍 비율: %d / %d = %.3f%%" % (R.hi.sum(), total_pairs, 100 * dup_rate))
print()
if (R.uniq == NF).all() and dup_rate < 0.01:
    print("  판정: 이 데이터에서는 다양성 문제가 관측되지 않는다.")
    print("        Overath 설계는 여러 캠페인에서 모은 것이라 원래 다양하다.")
    print("        → 이 데이터로는 검증이 불가능하다는 뜻이지, 문제가 없다는 뜻이 아니다.")
    print("        단일 캠페인 설계 풀이 생기면 그때 다시 잰다.")
else:
    print("  판정: 다양성 문제가 실재한다. 클러스터링 제어가 필요하다.")
