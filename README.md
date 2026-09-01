# 단백질 결합체 설계 — 정석 파이프라인

표적 단백질에 달라붙는 새 단백질을 컴퓨터로 설계하는 표준 절차를,
**실행 가능한 코드**와 **실측으로 확인한 판정 기준**으로 정리한 것.

> **이 저장소가 다른 점**
> 파이프라인은 어디에나 있다. 여기서 다른 건 **필터 임계값을 논문에서
> 베껴 오지 않고 실측 데이터 60만 개로 다시 재봤다**는 것이다.
> 그 결과 널리 쓰이는 필터 하나가 실은 아무 일도 안 한다는 것을 확인했다.
> → [VALIDATION.md](VALIDATION.md)

---

## 전체 흐름

```
표적 구조 (PDB)
    │
 1. 표적 준비        핫스팟 고르기               ← 사람이 판단
 2. 골격 생성        RFdiffusion                 GPU · 수 시간
 3. 서열 설계        ProteinMPNN                 수 분
 4. 구조 검증        AlphaFold2 initial guess    GPU · 가장 오래
 5. 필터와 순위      pae · plddt · rmsd          ← 사람이 판단
 6. 실험 발주        다양성 + 대조군 + 배치      수 초
    │
 96웰 플레이트 → 실험실 → 결과로 임계값 재조정 → 2라운드
```

각 단계가 무엇을 왜 하는지, 어디서 틀리는지 → **[PIPELINE.md](PIPELINE.md)**

---

## 바로 해보기

```bash
git clone https://github.com/dawnnam02/CALIPER && cd CALIPER
pip install -e .

python pipeline/run_all.py --check       # 설정과 설치된 도구 확인
python scripts/get_example_target.py     # 예제 표적 (MDM2, 94 KB)
python pipeline/step1_target.py          # 1단계는 GPU 없이 바로 돈다
```

1단계 출력 예 (MDM2 의 p53 결합 주머니):

```
  지정한 핫스팟: [54, 62, 93, 99]

       잔기     이름     소수성     이웃수   판정
  ----------------------------------------------------
       54    LEU       예      15   OK
       62    MET       예      13   OK
       93    VAL       예      16   OK
       99    ILE       예      14   OK
```

2·4단계는 GPU 와 모델 가중치가 필요하다 → [PIPELINE.md 의 도구 표](PIPELINE.md#필요한-도구).
**도구가 없으면 각 단계는 무엇을 설치해야 하는지 알려주고 멈춘다.
가짜 결과를 만들어내지 않는다.**

---

## 필터를 실측으로 다시 쟀다

```bash
python scripts/get_data.py               # 공개 캠페인 6종
python validation/check_filters.py
```

교과서 필터는 `pae_interaction < 10 · plddt_binder > 80 · rmsd < 2.0 Å` 이다.
실험 결과가 붙어 있는 캠페인 넷에 그대로 걸어봤다.

| 필터 | Bennett (60만 개) | Overath (3.7천 개) |
|---|---|---|
| `pae_interaction < 10` | **2.2×** | **2.4×** |
| `binder_aligned_rmsd < 2.0` | 2.6× | 1.1× |
| `plddt_binder > 80` | **0.9×** ⚠ | 1.4× |

*(농축 배수 = 통과한 것의 적중률 ÷ 전체 적중률. 1.0배면 아무 일도 안 한 것)*

**세 가지가 나왔다.**

**1. `pae_interaction` 은 실제로 작동한다.** 두 데이터셋 모두 2.2~2.4배.
그래서 이 파이프라인의 주 필터다.

**2. `plddt > 80` 은 가장 널리 쓰이는데 가장 일을 안 한다.**
Bennett 60만 개에서 **0.9배** — 통과한 쪽이 오히려 덜 붙었다.
81.8%를 통과시키니 거른다고 하기도 어렵다.
(참고로 `Rosetta ddG` 는 1.1배, `DAN_interface_lddt > 0.8` 은 **0.6배**로 해로웠다.)

**3. 어떤 임계값도 표적을 가로질러 옮겨지지 않는다.** ← 가장 중요하다

| 표적 | 전체 | `pae<10` 통과 | 농축 | |
|---|---|---|---|---|
| Bennett/Tie2 | 92,293 | **0** | — | **라이브러리 전멸** |
| Overath/VirB8 | 99 | **0** | — | **전멸** |
| Bennett/H3 | 57,381 | 2 | **0.0×** | **해로움** |
| Bennett/SARS_CoV2_RBD | 98,991 | 69 | **14.4×** | 훌륭 |
| Overath/Pdl1 | 95 | 93 | **1.0×** | 무의미 |

같은 숫자 하나를 걸었는데 통과율이 **0%에서 98%까지** 벌어진다.

**그래서 이 파이프라인은 임계값을 조용히 적용하지 않는다.**
[`step5_filter.py`](pipeline/step5_filter.py) 가 통과 개수를 먼저 보고하고,
너무 적으면 멈춰서 데이터에서 계산한 대안 임계값을 제안한다.

전체 결과와 한계 → **[VALIDATION.md](VALIDATION.md)**

---

## 검증에 쓴 데이터

전부 공개 데이터고, 커밋하지 않고 실행할 때 받아온다.
출처·라이선스·실측 내용은 `data/*/SOURCE.md` 에 있다.

| 캠페인 | 설계 | 표적 | 라이선스 |
|---|---|---|---|
| Bennett et al. 2023 | 603,178 | 10 | CC-BY-4.0 |
| Overath et al. 2025 | 3,650 | 15 | CC-BY-4.0 |
| Adaptyv Nipah 대회 | 1,201 | 1 | ODC-ODbL |
| Adaptyv EGFR R2 | 380 | 1 | ODbL |
| GEM×Adaptyv RBX1 대회 | 321 | 1 | ODC-ODbL |
| BindCraft (Pacesa 2025) | 212 | 13 | CC BY-NC-ND |

```bash
python scripts/get_data.py          # 전부 (약 170 MB)
python scripts/get_data.py adaptyv  # 0.2 MB 만
```

---

## 폴더

| | |
|---|---|
| `pipeline/` | 6단계. `step0_config.py` 하나만 고치면 전부 그 값으로 돈다 |
| `validation/` | 필터를 실측으로 다시 재는 스크립트 |
| `data/` | 검증용 공개 데이터 (받아온다) + 출처 문서 |
| `scripts/` | 데이터·예제 표적 받기 |
| `legacy/` | 이전 프로젝트 (CALIPER). 아래 참조 |

---

## `legacy/` 는 뭔가

이 저장소는 원래 **CALIPER** 라는 다른 것이었다 — 설계 캠페인이 자기
데이터를 거꾸로 읽고 있는지 검사하는 도구. 실측 근거는 남아 있지만
(캠페인 6종, 단위 22개, 민감도 0.833), **용도가 너무 좁았다.**
계산 몇 줄로 사고를 막아주긴 하는데, 애초에 그 계산을 하는 팀이 드물다.

지우지 않고 남겼다. **왜 접었는지가 다음 판단의 근거**이기 때문이고,
자기 결론을 여섯 번 깬 기록([`legacy/CRITIQUE.md`](legacy/CRITIQUE.md))이
파이프라인 자체보다 쓸모 있을 수도 있어서다.

이 저장소의 데이터셋 여섯 개와 그 출처 문서는 전부 그때 모은 것이고,
지금 필터 검증에 그대로 쓰인다.

---

## 솔직하게 밝힐 것

- **여기서 무거운 계산을 하지 않는다.** RFdiffusion · ProteinMPNN ·
  AlphaFold2 는 남이 만든 것을 받아 쓴다. 이 저장소가 하는 건
  **어떤 순서로 엮고 어디서 자를지**다. 정석의 실체가 그것이다.
- **1·5단계는 자동화되지 않는다.** 핫스팟 선택과 임계값 판단은 사람 몫이다.
  스크립트는 판단에 필요한 숫자를 계산해 줄 뿐이다.
- **검증에 쓴 표적은 21개다.** "통과율 0%가 두 번 나왔다"는 확실하지만,
  "내 표적에서 몇 %일까"는 이 데이터로 알 수 없다.
- **검증 데이터는 전부 선별된 집합이다.** 아무도 설계 전부를 실험하지 않는다.
  → [VALIDATION.md 의 '한계'](VALIDATION.md#이-검증의-한계--반드시-같이-읽을-것)

---

## 라이선스

코드는 MIT ([LICENSE](LICENSE)).
데이터는 각자의 라이선스를 따른다 — `data/*/SOURCE.md` 를 확인할 것.
특히 BindCraft 데이터는 **CC BY-NC-ND** 라 MIT 가 적용되지 않는다.
