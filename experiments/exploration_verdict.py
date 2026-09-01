"""Is an exploration quota worth its wells? Measured on real data: no.

CALIPER's most novel claim was that spending some wells on designs the filter
REJECTED buys better calibration, by supplying the low-score labels a
winners-only campaign never sees.  On the simulator it worked (out-of-fold ECE
0.220 -> 0.160 at 25% exploration, d=0.64).  On the Overath data it does not.

Aggregating by (target, budget) CELL rather than by repeat -- the exploit arm is
deterministic, so counting its repeats separately inflates one failure into
forty -- gives 19 cells:

    median ECE   exploit 0.065 | explore-25% 0.061
    maximum ECE  exploit 0.916 | explore-25% 0.167
    paired test  NO significant difference (diff +0.041, CI [-0.017, +0.142])

So there is no average benefit.  In 18 of 19 cells it is a wash.  What the
exploration quota did do was eliminate a single catastrophic failure -- and
chasing that failure produced something better than the quota.

The catastrophe, and the free fix
---------------------------------
EGFR, top 24 of 434 designs.  The labelled scores span 0.650-0.723, which is 10%
of the full range.  Inside that narrow band the score correlates NEGATIVELY with
outcome by chance, so Platt fits a slope of -17.1.  The curve then predicts
0.962 for the 410 unassayed designs whose true rate is 0.046.  ECE 0.916.  At
N=48 and N=96 the same target spans 17% and 33%, the slope comes out positive,
and ECE falls to 0.017.

**An inverted curve is detectable from the labelled data alone.**  It costs
nothing to check the sign of the fitted slope; the exploration quota costs a
quarter of the plate.  `caliper.smallsample.check_calibration` does the check,
and on these 19 cells it rejected exactly the catastrophic fit and nothing else.

Verdict: drop the exploration quota, keep the slope check.

Korean note:
탐색 할당량은 평균적으로 아무 이득이 없다 (19셀 중 18셀 무승부).  한 셀에서만
파국을 막았는데, 그 파국의 원인을 파보니 **웰을 쓰지 않고 공짜로 감지**할 수 있는
것이었다.  라벨 표본이 좁은 띠만 덮으면 기울기가 뒤집히고, 뒤집힌 곡선은
"점수가 높을수록 안 붙는다"고 말한다.  부호만 보면 된다.

Data: Overath et al. 2025, Zenodo 10.5281/zenodo.15722219, CC-BY-4.0
Run:  python experiments/exploration_verdict.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.smallsample import PlattCalibrator, check_calibration
from caliper.stats import bootstrap_ci, equal_mass_ece, paired_bootstrap, wilson
from caliper.types import stable_hash

DATA = Path(__file__).resolve().parents[1] / "data" / "overath" / "final_dataset.csv"

df=pd.read_csv(DATA,low_memory=False)
df=df[df.binder.notna()].copy(); df["y"]=df.binder.astype(bool).astype(int)
SC="af3_ipSAE_min"; df[SC]=pd.to_numeric(df[SC],errors="coerce")
df=df.dropna(subset=[SC]).reset_index(drop=True)
cells=[]
for held,g in df.groupby("target_id"):
    n=len(g)
    if n<90 or g.y.sum()<8: continue
    s=g[SC].to_numpy(float); y=g.y.to_numpy(); order=np.argsort(-s,kind="mergesort")
    for N in (24,48,96):
        if n<N*2: continue
        # exploit 은 결정적 — 한 번만 계산한다
        idx=order[:N]; lab=set(int(i) for i in idx)
        te=np.array([i for i in range(n) if i not in lab])
        if te.size<30 or len(set(y[idx].tolist()))<2: continue
        ex=equal_mass_ece(PlattCalibrator().fit(s[idx],y[idx]).predict(s[te]),y[te])
        # explore 는 무작위 — 반복 평균과 최악을 본다
        vals=[]
        for rep in range(40):
            rng=np.random.default_rng(int(stable_hash([held,N,rep,"f"]),16)%2**31)
            m=int(round(N*0.25)); top=order[:N-m]; rest=order[N-m:]
            xi=np.concatenate([top,rest[rng.choice(len(rest),size=m,replace=False)]])
            lab2=set(int(i) for i in xi); te2=np.array([i for i in range(n) if i not in lab2])
            if te2.size<30 or len(set(y[xi].tolist()))<2: continue
            vals.append(equal_mass_ece(PlattCalibrator().fit(s[xi],y[xi]).predict(s[te2]),y[te2]))
        if not vals: continue
        cells.append(dict(t=held,N=N,frac=N/n,exploit=ex,
                          explore_mean=float(np.mean(vals)),
                          explore_p90=float(np.percentile(vals,90))))
C=pd.DataFrame(cells)
print("="*86)
print("셀 단위 재집계 — (표적 x 예산) 하나를 1로 센다"); print("="*86)
print("  셀 %d개 (표적 %d x 예산)"%(len(C),C.t.nunique()))
print()
print("  %-16s %5s %7s %10s %14s %12s"%("표적","N","N/n","exploit","explore 평균","이득"))
print("  "+"-"*70)
for _,r in C.sort_values("exploit",ascending=False).iterrows():
    print("  %-16s %5d %6.1f%% %10.3f %14.3f %+12.3f"
          %(str(r.t)[:16],r.N,100*r.frac,r.exploit,r.explore_mean,r.exploit-r.explore_mean))
print("  "+"-"*70)
print()
for thr in (0.10,0.20,0.30,0.50):
    a=int((C.exploit>thr).sum()); b=int((C.explore_mean>thr).sum())
    print("  ECE > %.2f 인 셀:  exploit %d/%d   explore %d/%d"%(thr,a,len(C),b,len(C)))
print()
t=paired_bootstrap(C.exploit,C.explore_mean,"exploit","explore25",seed=5)
print("  "+t.verdict(lower_is_better=True))
print("  중앙값: exploit %.4f | explore %.4f"%(C.exploit.median(),C.explore_mean.median()))
print("  최대  : exploit %.4f | explore %.4f"%(C.exploit.max(),C.explore_mean.max()))
