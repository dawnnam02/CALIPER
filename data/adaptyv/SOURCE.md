# 데이터 출처 — Adaptyv EGFR 경진대회

`round2.csv` (402행) · `round1.csv` (202행)

- **논문**: Adaptyv Bio EGFR binder competition, bioRxiv 10.1101/2025.04.17.648362
- **저장소**: https://github.com/adaptyvbio/egfr_competition_2
- **라이선스**: 데이터 **ODbL**, 코드 Apache-2.0
- **받은 날**: 2026-09-01
- **직접 링크**:
  https://raw.githubusercontent.com/adaptyvbio/egfr_competition_2/main/results/result_summary.csv

## 왜 이 데이터가 중요한가

**Overath와 독립적이다.** 표적도, 설계를 만든 사람도, 실험한 곳도 다르다.

| | Overath | Adaptyv |
|---|---|---|
| 설계 출처 | 여러 발표 캠페인을 모음 | 경진대회 참가팀들이 제출 |
| 표적 | 15종 | EGFR 1종 |
| 실험 | 원 논문들 각각 | 한 곳에서 일괄 |
| 점수 | AF2/ColabFold/AF3/Boltz1 재계산 | 참가자 제출 + ColabFold 1회 |

이전 판의 모든 결론이 Overath 하나에 걸려 있었다. 그게 가장 약한 지점이었다.

## 내용 (실측)

| 항목 | 값 |
|---|---|
| 라벨 있는 설계 | 380 (`unknown` 22개 제외) |
| 결합 | 55 (**13.7%**) |
| 발현 | high 281 · medium 97 · none 22 · low 2 |
| 점수 | `iptm` · `plddt` · `pae_interaction` · `esm_pll` (전부 커버 100%) |

## 발표값 재현 확인

이 저장소 코드로 다시 계산한 값이 논문 보고값과 일치한다 —
**파일을 원저자 의도대로 읽고 있다는 증거다.**

| 지표 | 내 계산 | 논문 보고 |
|---|---|---|
| ipTM AUC | **0.636** | 0.64 |
| pLDDT AUC | **0.656** | 0.66 |

> CSV는 `.gitignore` 로 커밋에서 제외돼 있다. 위 링크로 다시 받으면 된다.
