"""The development harness — how the findings in `caliper/` were arrived at.

Nothing here is part of the contribution.  It is the simulator, the campaign
runner built around it, and the baselines that were raced against each other
before real data was available.

It is kept, not deleted, for one reason: **the simulator is what proved itself
wrong.**  It modelled stage errors as independent, said the cascade beat every
baseline, and was contradicted by real data.  Chasing that contradiction
produced the correlation measurement, the AUC-gap rule, and eventually the
inverted-curve detector.  A record of a method flattering its author is worth
keeping where someone can run it.

Everything a campaign should actually use lives one level up, in `caliper/`:

    caliper.audit           every surviving check, one entry point
    caliper.smallsample     calibration that refuses when the data is too thin
    caliper.hierarchical    per-target calibration with partial pooling
    caliper.whentocascade   whether a cheap stage earns its place

Korean note:
여기 있는 건 기여가 아니라 **개발 과정의 기록**이다.  시뮬레이터가 자기 결론을
스스로 반증했고, 그 모순을 쫓다가 상관 측정과 AUC 격차 규칙, 그리고 뒤집힌 곡선
감지기가 나왔다.  "방법이 저자를 편들었다"는 기록은 돌려볼 수 있게 남겨둘 값이 있다.
실제로 쓸 것은 전부 한 단계 위 `caliper/` 에 있다.
"""
