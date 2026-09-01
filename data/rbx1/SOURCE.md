# 데이터 출처 — GEM × Adaptyv RBX1 결합체 설계 대회

`rbx1.csv` (실험된 설계 321개 · 표적 1종 · 약 0.4 MB)

- **대회**: GEM × Adaptyv Bio RBX1 Binder Design Competition (ICLR 2026)
- **표적**: RBX1 (RING-box protein 1) — 컬린-RING 유비퀴틴 리가아제 구성 요소
- **호스팅**: https://proteinbase.com/collections/gem-x-adaptyv-rbx1-binder-design-competition-results
- **라이선스**: **ODC-ODbL** (Nipah와 같은 근거 — 대회 규정 문구)
- **받은 날**: 2026-09-01
- **받는 법**: `python scripts/get_data.py rbx1`

## 왜 넣었는가

**RBX1은 다른 어느 캠페인에도 없는 표적이다.** 그것 하나 때문에 넣었다.

## 이 데이터의 약점 — 먼저 밝힌다

> [!warning] 계면 점수가 없다
> 이 컬렉션은 ipSAE · ipTM · pAE 같은 **계면(interface) 신뢰 점수를 공개하지 않는다.**
> 공개된 연속 점수는 `esmfold_plddt`(단량체 접힘 신뢰도)와
> `proteinmpnn_score`(서열 설계 점수)뿐이다.
>
> 그래서 대표 지표를 `esmfold_plddt` 로 잡았다. 이건 "복합체가 잘 형성될까"가
> 아니라 "이 단백질 혼자 잘 접힐까"를 재는 값이다.
> **다른 캠페인의 대표 지표보다 약한 근거다.** 그대로 감안하고 읽어야 한다.

> [!note] 결합률이 낮아 작은 예산은 못 쓴다
> 결합체가 9개(2.8%)뿐이라 상위 12개·24개 안에 결합체가 하나도 안 들어온다.
> **한 계급만 있는 슬라이스는 교정 곡선을 못 맞추므로 건너뛴다.**
> 실제로 쓰이는 예산은 N=48 과 N=96 둘뿐이다.

## 내용 (실측)

| 항목 | 값 |
|---|---|
| 컬렉션의 설계 | 322 |
| 실험 반복이 있는 설계 | **321** (1개는 측정 없음 → 버림) |
| 결합 | **9 (2.8%)** |
| 설계 유형 | 상당수가 Nanobody |
| `esmfold_plddt` AUC | 0.715 |
| `proteinmpnn_score` AUC | 0.733 |

> 측정이 없는 설계를 "결합 안 함"으로 세지 않는다.
> **측정을 안 한 것과 결합을 안 한 것은 다르다.** 없는 라벨을 만들면 안 된다.

## 결과

대표 지표에서 **N=48·96 모두 발화했고 둘 다 파국**이었다 (ECE 0.519 · 0.792).
참 양성(true positive) 한 건이다.

## 겹침 확인

기존 데이터의 서열과 **겹치는 것이 0개**다.

> CSV는 `.gitignore` 로 커밋에서 제외돼 있다. `python scripts/get_data.py rbx1` 로 다시 받으면 된다.
