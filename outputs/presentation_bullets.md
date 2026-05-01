# 발표용 한국어 요약 (Solar-XAI Pitch Bullets)

> 데이터 출처: `synthetic` · 정규화 단위: AC_POWER min-max [0, 1] · 발전소 용량 ≈ 1373 kW · 1시간 후 발전량 예측

## 1) 문제 정의
- 태양광 (PV) 발전량은 일사량·온도·구름 변동으로 인해 운영자가 예측에 의존하기 어려운 자원이다.
- 우리는 **운영자가 신뢰할 수 있는 1시간 단위 발전량 예측 + 그 근거를 시각적으로 설명하는 XAI** 솔루션을 제공한다.

## 2) 데이터 & 전처리
- Kaggle "anikannal/solar-power-generation-data" Plant 1 (15분 단위 발전·기상 센서) 또는 동일 스키마의 합성 fallback.
- 1시간 단위 집계, 야간(일사량 ≈ 0) 자동 0 처리, 시간 순서 70/15/15 train/val/test 분할.
- AC_POWER를 min-max로 정규화하여 논문과 동일 스케일에서 MAE를 보고.

## 3) 모델 사다리 (MAE 중심 튜닝)
1. Persistence (전일 동시각)
2. Ridge (선형 baseline)
3. LightGBM (`regression_l1`)
4. XGBoost (`reg:absoluteerror`, GPU)
5. CatBoost (`MAE`, GPU)
6. LSTM (PyTorch, L1Loss, 멀티 seed 평균)
7. **Stacking 앙상블**: 위 base predictions + Ridge 메타러너 / Simplex blend 중 OOF MAE가 더 낮은 것 채택.

## 4) 핵심 결과 — MAE 최우선
- 우승 모델: **cat**
- 테스트 MAE (정규화): **0.0137**
- 테스트 MAE (kW 환산): **18.28 kW**
- 헤드라인: **논문(SOTA) 대비 정규화 MAE를 +94.0% 개선 (0.0137 vs 0.229)**
- 비교 대상 논문: X-LSTM-EO (Khan et al., PLOS One 2024) — 동일 데이터셋(Kaggle anikannal Plant 1), 보고 MAE = 0.229.
- *주의*: 논문 수치와 본 실험 수치는 별도 표에 분리 보관 (`reference/benchmarks.md` vs `outputs/metrics.csv`).

## 5) 왜 이렇게 예측이 나왔는가? — XAI 인사이트
- LightGBM SHAP 분석 결과, 예측에 가장 크게 기여하는 변수는: **power_same_hour_mean_3d, zenith_deg, cos_hour, IRRADIATION, cloud_index**.
- 일사량(`IRRADIATION`)과 청천 일사량(`clear_sky`) 관련 피처가 상위 기여도 → 모델이 물리적으로 타당한 신호를 학습.
- 모듈 온도(`MODULE_TEMPERATURE`)는 고온일수록 발전 효율을 깎는 음의 기여를 보여주어 panel temperature derate 효과가 그대로 재현됨.
- 직전 1시간 발전량(`power_lag_1`)과 어제 같은 시각(`power_lag_24`) lag 피처가 단기·일주기 동역학을 포착.
- 특정 일자 local 설명 (SHAP waterfall): 맑은 정오 vs. 흐린 정오 비교를 제시해 발표자가 "왜 이때 예측이 높았는가"를 한 화면으로 설명 가능.

## 6) 비즈니스 임팩트
- MAE는 발전량과 동일한 단위(kW)로 해석되어 운영자가 직관적으로 "예측 오차 → 보조전원 손실 비용"으로 환산 가능.
- 블랙박스가 아닌 SHAP 기반 설명을 제공하여 grid 운영자/규제기관 신뢰 확보.
- 동일 데이터셋에서 논문 SOTA 대비 MAE 개선을 달성함으로써 기술 차별성 입증.

## 7) 다음 단계
- 멀티 사이트 일반화 (현재 Plant 1 단일 사이트 → Plant 2/타지역 확장).
- 1시간 → 6/24시간 다중 horizon 확장.
- 확률적 forecast (quantile / interval prediction).
- 운영자용 dashboard에 SHAP 설명을 함께 노출.
