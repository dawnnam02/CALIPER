"""Why the cascade lost on real data, and what the answer is worth.

CALIPER's headline claim did not survive contact with the Overath dataset.
This file is the post-mortem, and it produced something more useful than the
claim it killed.

Sections
--------
1. STAGE CORRELATION.  The simulator modelled stage errors as independent.
   Measured Spearman on real data: AF2-AF3 0.550, ColabFold-AF3 0.657,
   AF2-ColabFold 0.574.  Independence is the most favourable possible
   assumption for a cascade, and it was wrong.

2. MECHANISM.  The cheap AF2 rung discards 29% of the real binders that the
   expensive AF3 rung's top-24 would have found.

3. DECOMPOSITION.  Across 10 evaluable targets, what predicts whether the
   cascade wins:
       AUC gap between cheap and final stage   rho = -0.652   <- dominant
       pool size                               rho = +0.518
       absolute AUC of the cheap stage         rho = +0.146   <- irrelevant
   Injecting the measured correlation into the simulator moved the cascade
   from "significantly better" to "no significant difference" -- it explains
   part of the gap.  The AUC gap explains the rest.

4. THE RULE, and its verdict on this pipeline.  See caliper/whentocascade.py.
   Applied to CALIPER's own configuration it says: drop the AF2 stage, it is
   0.066 AUC worse at the same job.

5. DOES FOLLOWING THE RULE HELP?  Partly.  Dropping AF2 is nominally better
   (0.358 vs 0.338) but not significantly, and BOTH cascades still lose to
   simply running AF3 on everything (0.400).

Conclusion: on a fixed candidate pool with these metrics, cascading does not
pay.  It pays only when compute is the binding constraint and the pool is
large enough that screening 3x more designs outweighs the loss.

Korean note:
내 주장이 실데이터에서 무너졌고, 그 원인을 파헤친 결과가 원래 주장보다 쓸모 있었다.
"싸다고 앞에 붙이지 마라. 최종 단계와 판별력이 비슷할 때만 의미가 있다"는 규칙이 나왔다.

Data: Overath et al. 2025, Zenodo 10.5281/zenodo.15722219, CC-BY-4.0
Run:  python experiments/why_cascade_lost.py
"""


import sys
from pathlib import Path; sys.path.insert(0, __file__.rsplit("experiments",1)[0])
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from caliper.metrics import roc_auc
from caliper.metrics import spearman
from caliper.stats import bootstrap_ci, paired_bootstrap
from caliper.types import stable_hash
from caliper.whentocascade import explain_stage_order, should_cascade

DATA = str(Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv")

def _require(path):
    """Fail with instructions, not a traceback, when the dataset is absent."""
    if not path.exists():
        print("dataset not found: " + str(path), file=sys.stderr)
        print("  python scripts/get_data.py overath", file=sys.stderr)
        print("  (82 MB, CC-BY-4.0, Zenodo 10.5281/zenodo.15722219)",
              file=sys.stderr)
        raise SystemExit(2)
    return path









# ==========================================================================
# 1. STAGE CORRELATION AND MECHANISM
# ==========================================================================

df=pd.read_csv(str(Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv"),low_memory=False)
df=df[df.binder.notna()].copy(); df["y"]=df.binder.astype(bool).astype(int)
ST=[("af2_pae_interaction",True),("colab_ipSAE_min",False),("af3_ipSAE_min",False)]
for c,_ in ST: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=[c for c,_ in ST]).reset_index(drop=True)
S={c:(-df[c].to_numpy(float) if low else df[c].to_numpy(float)) for c,low in ST}
names=[c for c,_ in ST]; y=df.y.to_numpy()

print("="*74); print("1. 단계 간 스피어만 상관 (실데이터 n=%d)"%len(df)); print("="*74)
print("  %-24s %8s %8s %8s"%("", *[n.split('_')[0] for n in names]))
for a in names:
    print("  %-24s"%a.split('_')[0], end="")
    for b in names: print(" %8.3f"%spearman(S[a],S[b]),end="")
    print()
print()
print("  ※ 시뮬레이터는 이 값들을 '독립 잡음'으로 뒀다 — 즉 상관 0에 가깝게 가정")

print()
print("="*74); print("2. 기전: 값싼 1단계가 최종 우수 후보를 얼마나 죽이나"); print("="*74)
print("  %-16s %6s %10s %12s %12s"%("표적","풀","af3 상위24","af2에서 생존","실제 결합 손실"))
print("  "+"-"*62)
tot_lost=0; tot_hits=0
for t,g in df.groupby("target_id"):
    n=len(g)
    if n<100: continue
    idx=g.index.to_numpy()
    a2=S["af2_pae_interaction"][idx]; a3=S["af3_ipSAE_min"][idx]; yy=y[idx]
    top24=np.argsort(-a3,kind="mergesort")[:24]
    keep1=set(np.argsort(-a2,kind="mergesort")[:max(24,int(n/3))].tolist())
    surv=[i for i in top24 if i in keep1]
    lost_hits=int(yy[[i for i in top24 if i not in keep1]].sum())
    tot_lost+=lost_hits; tot_hits+=int(yy[top24].sum())
    print("  %-16s %6d %10d %12d %12d"%(str(t)[:16],n,24,len(surv),lost_hits))
print("  "+"-"*62)
print("  합계: af3 상위24 안의 실제 결합 %d개 중 %d개(%.0f%%)를 af2 단계가 버렸다"
      %(tot_hits,tot_lost,100*tot_lost/max(tot_hits,1)))

print()
print("="*74); print("3. 상관이 높으면 왜 캐스케이드가 손해인가 — 조건부 정보량"); print("="*74)
for c,_ in ST[:2]:
    r_y=roc_auc(S[c],y); r_last=spearman(S[c],S["af3_ipSAE_min"])
    print("  %-22s AUC %.3f | af3와의 상관 %.3f" % (c,r_y,r_last))
print()
print("  af2는 af3와 %.2f 상관이면서 AUC는 %.3f 낮다."
      %(spearman(S['af2_pae_interaction'],S['af3_ipSAE_min']),
        roc_auc(S['af3_ipSAE_min'],y)-roc_auc(S['af2_pae_interaction'],y)))
print("  → 같은 걸 보는데 더 부정확하다. 그런 필터를 먼저 걸면 정보가 아니라 잡음만 더한다.")



# ==========================================================================
# 3. DECOMPOSITION ACROSS TARGETS
# ==========================================================================

df=pd.read_csv(str(Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv"),low_memory=False)
df=df[df.binder.notna()].copy(); df["y"]=df.binder.astype(bool).astype(int)
ST=[("af2_pae_interaction",True),("colab_ipSAE_min",False),("af3_ipSAE_min",False)]
for c,_ in ST: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=[c for c,_ in ST]).reset_index(drop=True)
NF,RED=24,3.0
def orient(g,c,low):
    v=g[c].to_numpy(float); return -v if low else v

print("="*88)
print("표적별 분해 — 캐스케이드 손실이 어디서 나오나")
print("="*88)
print("  %-16s %6s %8s %8s %9s %9s %9s %8s"%("표적","n","n/3","af2 AUC","af3 AUC","캐스케이드","단일최고","차이"))
print("  "+"-"*82)
rows=[]
for t,g in df.groupby("target_id"):
    n=len(g)
    if n<60 or g.y.sum()<3: continue
    S={c:orient(g,c,low) for c,low in ST}; y=g.y.to_numpy()
    keeps=[max(NF,int(n/RED)), max(NF,int(n/RED/RED)), NF]
    alive=np.arange(n)
    for (c,_),k in zip(ST,keeps): alive=alive[np.argsort(-S[c][alive],kind="mergesort")[:k]]
    bs=np.argsort(-S["af3_ipSAE_min"],kind="mergesort")[:NF]
    ch,bh=float(y[alive].mean()),float(y[bs].mean())
    a2,a3=roc_auc(S["af2_pae_interaction"],y),roc_auc(S["af3_ipSAE_min"],y)
    rows.append(dict(t=t,n=n,n3=keeps[0],a2=a2,a3=a3,c=ch,b=bh,d=ch-bh))
    print("  %-16s %6d %8d %8.3f %9.3f %9.3f %9.3f %+8.3f"%(str(t)[:16],n,keeps[0],a2,a3,ch,bh,ch-bh))
R=pd.DataFrame(rows)
print()
print("  큰 표적(n>=400) 평균 차이: %+.3f  (n=%d)"%(R[R.n>=400].d.mean(),len(R[R.n>=400])))
print("  작은 표적(n<400) 평균 차이: %+.3f  (n=%d)"%(R[R.n<400].d.mean(),len(R[R.n<400])))
print()
print("  af2 AUC와 캐스케이드 우위의 상관: %+.3f"%spearman(R.a2,R.d))
print("  af2-af3 AUC 격차와 우위의 상관 : %+.3f"%spearman(R.a3-R.a2,R.d))
print("  풀 크기와 우위의 상관          : %+.3f"%spearman(R.n,R.d))
print()
print("  → 큰 표적일수록 캐스케이드가 유리한가? %s"%("그렇다" if R[R.n>=400].d.mean()>R[R.n<400].d.mean() else "아니다"))



# ==========================================================================
# 4. THE RULE, VALIDATED
# ==========================================================================

df=pd.read_csv(str(Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv"),low_memory=False)
df=df[df.binder.notna()].copy(); df["y"]=df.binder.astype(bool).astype(int)
ST=[("af2_pae_interaction",True),("colab_ipSAE_min",False),("af3_ipSAE_min",False)]
for c,_ in ST: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=[c for c,_ in ST]).reset_index(drop=True)
COSTS=[1.0,8.0,20.0]; NF,RED=24,3.0
def orient(g,c,low):
    v=g[c].to_numpy(float); return -v if low else v

print("="*86)
print("규칙 검증 — 규칙이 예측한 것과 실제 결과가 맞나 (실데이터 10표적)")
print("="*86)
print("  %-16s %7s %8s %10s %10s %9s %s"%("표적","n","AUC격차","규칙 예측","실제 차이","일치?",""))
print("  "+"-"*80)
ok=0; tot=0
for t,g in df.groupby("target_id"):
    n=len(g)
    if n<60 or g.y.sum()<3: continue
    S={c:orient(g,c,low) for c,low in ST}; y=g.y.to_numpy()
    a2=roc_auc(S["af2_pae_interaction"],y); a3=roc_auc(S["af3_ipSAE_min"],y)
    keeps=[max(NF,int(n/RED)),max(NF,int(n/RED/RED)),NF]; alive=np.arange(n)
    for (c,_),k in zip(ST,keeps): alive=alive[np.argsort(-S[c][alive],kind="mergesort")[:k]]
    bs=np.argsort(-S["af3_ipSAE_min"],kind="mergesort")[:NF]
    actual=float(y[alive].mean())-float(y[bs].mean())
    v=should_cascade(a2,a3,n,COSTS)
    pred="캐스케이드" if v.should_cascade else "단일단계"
    agree = (v.should_cascade and actual>=0) or ((not v.should_cascade) and actual<0)
    ok+=agree; tot+=1
    print("  %-16s %7d %8.3f %10s %+10.3f %9s"%(str(t)[:16],n,a3-a2,pred,actual,"O" if agree else "X"))
print("  "+"-"*80)
print("  일치 %d/%d = %.0f%%"%(ok,tot,100*ok/tot))
print()
print("="*86); print("이 파이프라인 자신의 설정을 규칙에 넣으면"); print("="*86)
A=[roc_auc(orient(df,c,low),df.y.to_numpy()) for c,low in ST]
print(explain_stage_order(A,COSTS,[c for c,_ in ST]))
print()
print(should_cascade(A[0],A[2],3650,COSTS))



# ==========================================================================
# 5. DOES FOLLOWING THE RULE HELP?
# ==========================================================================

df=pd.read_csv(str(Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv"),low_memory=False)
df=df[df.binder.notna()].copy(); df["y"]=df.binder.astype(bool).astype(int)
COLS={"af2":("af2_pae_interaction",1.0,True),"colab":("colab_ipSAE_min",8.0,False),
      "af3":("af3_ipSAE_min",20.0,False)}
for c,_,_ in COLS.values(): df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=[c for c,_,_ in COLS.values()]).reset_index(drop=True)
NF,RED=24,3.0
def orient(g,k):
    c,_,low=COLS[k]; v=g[c].to_numpy(float); return -v if low else v

def cascade(S,order,n):
    keeps=[]; cur=n
    for i in range(len(order)):
        cur=NF if i==len(order)-1 else max(NF,int(cur/RED)); keeps.append(min(cur,n))
    alive=np.arange(n); cost=0.0; ins=[n]+keeps[:-1]
    for k,kk,i in zip(order,keeps,ins):
        cost+=i*COLS[k][1]
        alive=alive[np.argsort(-S[k][alive],kind="mergesort")[:kk]]
    return alive,cost

res={}
for t,g in df.groupby("target_id"):
    n=len(g)
    if n<60 or g.y.sum()<3: continue
    S={k:orient(g,k) for k in COLS}; y=g.y.to_numpy()
    a3,c3=cascade(S,["af2","colab","af3"],n)
    a2,c2=cascade(S,["colab","af3"],n)          # 규칙의 권고: af2 제거
    bs=np.argsort(-S["af3"],kind="mergesort")[:NF]; cb=n*COLS["af3"][1]
    res.setdefault("3단계 (af2+colab+af3)",[]).append((float(y[a3].mean()),c3))
    res.setdefault("2단계 (colab+af3) ★규칙권고",[]).append((float(y[a2].mean()),c2))
    res.setdefault("단일 (af3만)",[]).append((float(y[bs].mean()),cb))

print("="*80)
print("규칙의 권고를 따르면 실제로 나아지는가 (실데이터 10표적)")
print("="*80)
print("  %-30s %26s %10s"%("정책","적중률","평균비용"))
print("  "+"-"*70)
for k,v in res.items():
    h=[x[0] for x in v]; c=[x[1] for x in v]
    print("  %-30s %26s %10.0f"%(k,str(bootstrap_ci(h,seed=1)),np.mean(c)))
print()
h3=[x[0] for x in res["3단계 (af2+colab+af3)"]]
h2=[x[0] for x in res["2단계 (colab+af3) ★규칙권고"]]
hb=[x[0] for x in res["단일 (af3만)"]]
print("  쌍대 검정:")
print("   ",paired_bootstrap(h2,h3,"2단계","3단계",seed=2).verdict())
print("   ",paired_bootstrap(h2,hb,"2단계","단일af3",seed=2).verdict())
print("   ",paired_bootstrap(h3,hb,"3단계","단일af3",seed=2).verdict())
c2=np.mean([x[1] for x in res["2단계 (colab+af3) ★규칙권고"]])
cb=np.mean([x[1] for x in res["단일 (af3만)"]])
print()
print("  2단계는 단일af3 대비 비용 %.2f배 (%.0f%% 절감)"%(c2/cb,100*(1-c2/cb)))
