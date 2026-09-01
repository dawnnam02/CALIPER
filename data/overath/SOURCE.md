# 데이터 출처

`final_dataset.csv` (82 MB)

- **논문**: Overath et al. 2025, *Predicting Experimental Success in De Novo
  Binder Design: A Meta-Analysis of 3,766 Experimentally Characterised Binders*,
  bioRxiv 10.1101/2025.08.14.670059
- **데이터**: Zenodo 10.5281/zenodo.15722219
- **라이선스**: CC-BY-4.0
- **받은 날**: 2026-09-01
- **직접 링크**:
  https://zenodo.org/api/records/15722219/files/final_dataset.csv/content

## 내용 (실측)

| 항목 | 값 |
|---|---|
| 라벨 있는 설계 | 3,669 (단계 점수 결측 제외 후 3,650) |
| 결합 | 392 (**10.7%**) |
| 표적 | 15 |
| 열 | 312 |

**예측기별 열**: af2 48 · af3 60 · boltz1 66 · colabfold

**단일 지표 판별력(전체)**: `af3_ipSAE_min` **0.786** (최고) ·
`colab_ipSAE_min` 0.748 · `af2_pae_interaction` 0.721

**표적별 성공률**: 2.1% (pMHC_NY1) ~ 57.3% (Mdm2)
**표적별 AUC**: 0.573 (Mdm2) ~ 1.000 (LTK)

> 이 파일은 `.gitignore` 로 커밋에서 제외돼 있다 (82 MB).
> 위 링크로 다시 받으면 된다.
