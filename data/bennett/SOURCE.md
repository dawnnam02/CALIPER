# 데이터 출처 — Bennett et al. 2023 후향 분석

`retrospective.csv` (603,178행 · 10 표적 · 약 86 MB)

- **논문**: Bennett et al., *Improving de novo protein binder design with deep
  learning*, Nature Communications 14, 2625 (2023)
- **DOI**: 10.1038/s41467-023-38328-5
- **라이선스**: **CC-BY-4.0** (Nature Communications 오픈액세스)
- **받은 날**: 2026-09-01
- **원본 위치**: Supplementary Data 4 (139 MB zip) 안의
  `all_data/retrospective_analysis/retrospective_analysis_more_scores.sc`
- **받는 법**: `python scripts/fetch_bennett.py`

## 왜 139 MB를 통째로 받지 않는가

zip 파일은 **끝에 목차(central directory)** 가 붙어 있다. 그 목차만 먼저
HTTP 범위 요청으로 받아보면, 원하는 파일이 몇 번째 바이트에 있는지 알 수 있다.
그 구간(약 27 MB)만 받아서 압축을 풀면 된다.
`scripts/fetch_bennett.py` 가 이걸 한다. 서버가 범위 요청을 거부하면
139 MB 전체를 받는 경로로 자동으로 넘어간다.

## 왜 이 데이터를 넣었는가

기존 근거는 **파국 사례 6건**뿐이었다. 신뢰구간 아래끝이 0.436이라
"동전 던지기와 다를 바 없다"는 가능성이 열려 있었다.
Bennett을 넣어 파국 11건, 단위 19개가 되면서 구간이 좁아졌다.

| | Overath | Adaptyv | **Bennett** |
|---|---|---|---|
| 설계 수 | 3,650 | 380 | **603,178** |
| 표적 | 15 (쓸 수 있는 건 10) | 1 | **10 (쓸 수 있는 건 8)** |
| 실험 | 원 논문들 각각 | 한 곳에서 일괄 | 저자 연구실 효모 표면제시 |
| 대표 점수 | `af3_ipSAE_min` | `iptm` | `pAE_interaction` |

**새 표적**: PDGFR · Tie2 · H3 는 Overath에 아예 없다.

## Overath와 겹친다 — 그리고 얼마나 겹치는지 셌다

Bennett의 "retrospective"는 **이전에 발표된 설계를 다시 채점한 것**이다.
그래서 Overath 설계 3,669개 중 **1,642개(45%)가 Bennett 표에 이름 그대로 있다.**
Overath의 `source` 열은 Bennett을 121행만 credit하지만 실제 겹침은 훨씬 크다.

> [!important] 그래도 두 단위를 따로 세는 이유
> 교정 곡선이 실제로 학습하는 건 **상위 N개 슬라이스**다.
> 겹치는 표적 7종의 상위 96개 슬라이스를 세어 보면 **1,344개 중 7개(0.5%)만 공통**이다.
> 서로 다른 점수로 서로 다른 크기의 풀을 정렬하니 슬라이스가 다른 곳에 떨어진다.
>
> `experiments/detector.py` 가 이 숫자를 **매번 다시 세어 출력한다.**
> 주장으로 남기지 않고 검사로 남겼다.

다만 표적 단백질 자체는 겹친다. 그래서 단위 19개는 **단백질 12종**을 덮는다.
단위 수를 단백질 수인 것처럼 쓰지 않는다.

## "결합했다"의 정의 — 두 가지를 다 시험했다

| 정의 | 양성 | 비율 | 결과 |
|---|---|---|---|
| `avid_ub` 유한 (효모 표면제시 avidity Kd 측정됨) | 23,044 | 3.8% | **채택** |
| `kd_ub` 유한 (SPR Kd 측정됨) | 2,374 | 0.4% | 기각 |

`kd_ub` 쪽은 8표적 8/8로 더 좋아 보이지만 **음성 사례가 하나도 없다.**
적중률이 0.02~2%라 모든 표적이 파국으로 분류되고, 그러면
**항상 발화하는 고장난 감지기도 만점을 받는다.** 아무것도 떨어질 수 없는
시험은 시험이 아니다. 그래서 양쪽 계급이 다 있는 `avid_ub` 를 쓴다.
`avid_doesnt_agree` 플래그는 603,178행 전부 `False` 였다.

## 쓰는 열

`description` `target` `avid_ub` `avid_lb` `kd_ub` `kd_lb`
`pAE_interaction` `RF2_pAE_interaction` `pAE_interaction_no_guess`
`AF2_plddt_monomer` `RF2_plddt_monomer` `AF2_complex_RMSD`
`DAN_interface_lddt` `Rosetta ddG`

`description` 은 위의 겹침 검사에 필요해서 남겼다.

## 표적별 실측

| 표적 | 설계 | avidity 양성 | % |
|---|---|---|---|
| SARS_CoV2_RBD | 98,991 | 298 | 0.30 |
| PDGFR | 98,825 | 936 | 0.95 |
| EGFR | 98,746 | 339 | 0.34 |
| Tie2 | 92,293 | 764 | 0.83 |
| FGFR2 | 59,343 | 16,341 | 27.54 |
| InsulinR | 58,731 | 724 | 1.23 |
| H3 | 57,381 | 13 | 0.02 |
| TrkA | 14,982 | 1,953 | 13.04 |
| IL7Ra | 14,888 | 471 | 3.16 |
| VirB8 | 8,998 | 1,205 | 13.39 |

H3는 양성이 13개뿐이라 상위 N개 안에 양성이 안 들어와 실제 분석에서 빠진다.
Tie2는 대표 점수(`pAE_interaction`)로는 빠지고 보조 점수에서만 잡힌다.

> CSV는 `.gitignore` 로 커밋에서 제외돼 있다. `scripts/fetch_bennett.py` 로 다시 받으면 된다.
