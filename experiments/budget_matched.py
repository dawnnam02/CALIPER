"""The decisive experiment: is a cascade worth it at EQUAL COST?

experiments/real_data.py compares policies on a fixed candidate pool and the
cascade LOSES -- 0.316 hit rate against 0.368 for simply running the best
single metric on everything (paired d=-0.64, significant).  That is the honest
result when compute is free.

But a cascade is not a method for when compute is free.  It exists to let you
screen MORE candidates for the same spend.  This file gives every policy the
same budget and asks which one finds more binders with it.

Result: the cascade screens 3.2x more designs for the same cost and wins
(0.438 vs 0.323, paired d=1.33).  Only 4 targets are large enough for the
comparison to be meaningful, so the interval is wide and this is suggestive,
not settled.

Korean note:
고정된 후보 풀에서는 캐스케이드가 진다.  연산이 공짜면 제일 좋은 지표를 전부에
돌리는 게 낫다.  캐스케이드는 그런 상황을 위한 게 아니다.  같은 예산으로 더 많이
훑기 위한 것이고, 예산을 맞추면 이긴다.  다만 표적 4개뿐이라 아직 단정할 수 없다.

Data: Overath et al. 2025, Zenodo 10.5281/zenodo.15722219, CC-BY-4.0
Run:  python experiments/budget_matched.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from caliper.stats import bootstrap_ci, paired_bootstrap
from caliper.types import stable_hash

DATA = Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv"
df=pd.read_csv(DATA,low_memory=False)
df=df[df.binder.notna()].copy(); df["y"]=df.binder.astype(bool).astype(int)
ST=[("af2_pae_interaction",1.0,True),("colab_ipSAE_min",8.0,False),("af3_ipSAE_min",20.0,False)]
for c,_,_ in ST: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=[c for c,_,_ in ST]).reset_index(drop=True)
N_FINAL=24; RED=3.0

def orient(g,c,low): 
    v=g[c].to_numpy(float); return -v if low else v

def casc_cost(n):
    k=[max(N_FINAL,int(n/RED)), max(N_FINAL,int(n/RED/RED)), N_FINAL]
    ins=[n,k[0],k[1]]
    return sum(i*c for i,(_,c,_) in zip(ins,ST)), k

def run_casc(s,n):
    _,k=casc_cost(n); alive=np.arange(n)
    for (c,_,_),kk in zip(ST,k):
        alive=alive[np.argsort(-s[c][alive],kind="mergesort")[:kk]]
    return alive

print("="*76)
print("예산 일치 비교 — 같은 연산비로 캐스케이드는 몇 배 더 훑을 수 있나")
print("="*76)
rows=[]
for held,g in df.groupby("target_id"):
    n=len(g)
    if n<120 or g.y.sum()<5: continue
    s={c:orient(g,c,low) for c,_,low in ST}
    y=g.y.to_numpy()
    # 캐스케이드: 전체 n 훑기
    cc,_=casc_cost(n); ci=run_casc(s,n)
    # best_single: 같은 예산으로 살 수 있는 설계 수
    afford=int(cc/ST[-1][1])
    rng=np.random.default_rng(int(stable_hash(["bm",held]),16)%2**31)
    if afford<N_FINAL: continue
    sub=rng.choice(n,size=min(afford,n),replace=False)
    bi=sub[np.argsort(-s[ST[-1][0]][sub],kind="mergesort")[:N_FINAL]]
    rows.append(dict(target=held,n=n,cost=cc,afford=afford,ratio=n/afford,
                     casc=float(y[ci].mean()),best=float(y[bi].mean())))
R=pd.DataFrame(rows)
print("  %-16s %6s %8s %8s %7s %8s %8s" % ("표적","풀","예산","훑기가능","배율","캐스케이드","단일최고"))
print("  "+"-"*68)
for _,r in R.iterrows():
    print("  %-16s %6d %8.0f %8d %6.1fx %8.3f %8.3f" %
          (str(r.target)[:16],r.n,r.cost,r.afford,r.ratio,r.casc,r.best))
print()
print("  캐스케이드 %s" % bootstrap_ci(R.casc,seed=1))
print("  단일최고   %s" % bootstrap_ci(R.best,seed=1))
t=paired_bootstrap(R.casc,R.best,"cascade","best_single(예산일치)",seed=3)
print("  %s" % t.verdict())
print("  평균 배율: %.1f배 더 많은 설계를 훑는다" % R.ratio.mean())
