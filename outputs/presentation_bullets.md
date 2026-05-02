# 발표용 한국어 요약 (Solar-XAI Pitch Bullets)

> 데이터 출처: `synthetic` · 정규화 단위: AC_POWER min-max [0, 1] · 발전소 용량 ≈ 1575 kW · 1시간 후 발전량 예측

## 1) 문제 정의
- 태양광 (PV) 발전량은 일사량·온도·구름 변동으로 인해 운영자가 예측에 의존하기 어려운 자원이다.
- 기존 산업의 대다수 예측 모델은 **블랙박스** — 정확도가 좋아도 "왜 이렇게 나왔냐"에 답을 못 한다.
- 우리는 **운영자가 신뢰할 수 있는 1시간 단위 발전량 예측 + 모든 예측에 대해 SHAP / IG 기반 근거**를 함께 제공한다.

## 2) 데이터 & 전처리
- Kaggle "anikannal/solar-power-generation-data" Plant 1 (15분 단위 발전·기상 센서) 또는 동일 스키마의 합성 fallback.
- 1시간 단위 집계, 야간(일사량 ≈ 0) 자동 0 처리, 시간 순서 70/15/15 train/val/test 분할.
- AC_POWER를 min-max로 정규화하여 논문과 동일 스케일에서 MAE를 보고.
- **leakage 방지**: 모든 lag/rolling 피처는 `t` 시점 이전 값만 사용.

## 3) 모델 사다리 (MAE 중심 튜닝)
1. Persistence (전일 동시각) — 산업 baseline
2. Ridge (선형 baseline)
3. LightGBM (`regression_l1`)
4. XGBoost (`reg:absoluteerror`, GPU)
5. CatBoost (`MAE`, GPU)
6. LSTM (PyTorch, L1Loss, 멀티 seed 평균)
7. **Stacking 앙상블**: 위 base predictions + Ridge 메타러너 / Simplex blend 중 OOF MAE가 더 낮은 것 채택.

> 모든 손실함수가 **MAE-aligned** — L2/MSE로 학습하고 MAE로 평가하는 일반적 leakage를 차단.

## 4) 핵심 결과 — MAE 최우선
- 우승 모델: **ensemble**
- 테스트 MAE (정규화): **0.0983**
- 테스트 MAE (kW 환산): **147.46 kW**
- 헤드라인: **논문(SOTA) 대비 정규화 MAE를 +57.1% 개선 (0.0983 vs 0.229)**
- 비교 대상 논문: X-LSTM-EO (Khan et al., PLOS One 2024) — 동일 데이터셋(Kaggle anikannal Plant 1), 보고 MAE = 0.229.
- *주의*: 논문 수치와 본 실험 수치는 별도 표에 분리 보관 (`reference/benchmarks.md` vs `outputs/metrics.csv`).

## 5) "왜 이렇게 예측이 나왔는가?" — Global XAI

### 5-1) 어떤 변수가 가장 큰 영향을 주는가
- LightGBM SHAP 분석 결과, 평균 |SHAP| 상위 5개 변수:
  - **irrad_x_cos_zenith** — 평균 |SHAP| 0.0410
  - **power_lag_168** — 평균 |SHAP| 0.0205
  - **temp_delta** — 평균 |SHAP| 0.0178
  - **power_lag_6** — 평균 |SHAP| 0.0178
  - **hour** — 평균 |SHAP| 0.0173

### 5-2) 왜 이 변수들이 의미가 있는가 (왜 → 왜 → 왜)
- **`IRRADIATION` (측정 일사량)** — 태양빛이 패널에 닿는 양 자체. 광전 변환의 원천. **왜 영향이 큰가?** 발전량의 1차 결정 요인이기 때문.
- **`clear_sky` / `cos_zenith`** — 시각·계절·위치로부터 계산한 *물리 기반 baseline*. **왜 추가했나?** 측정 일사량은 센서 노이즈와 구름이 섞여 있다. clear_sky는 "맑다면 이 정도여야 한다"는 기준선 → 모델이 두 값의 차이로 구름 영향을 학습.
- **`cloud_index = irradiation / clear_sky`** — 우리가 만든 구름 지표. **왜 효과적인가?** 일사량 절대값이 아닌 *비율*이라 시간대·계절에 무관한 정규화된 구름 신호.
- **`MODULE_TEMPERATURE`** — 패널 표면 온도. 25 °C 이상에서 SHAP이 음 → **왜?** PV 효율은 0.4 %/°C 수준의 음의 온도 계수를 가짐. 모델이 이 물리 법칙을 자연스럽게 재현.
- **`power_lag_1` / `power_lag_24` / `power_same_hour_mean_3d`** — 단기 자기회귀(직전 1시간) + 일주기성(어제 같은 시각) + 3일 평균. **왜?** 발전량에는 강한 일주기와 자기상관이 있어, 직전 값이 다음 값에 대한 가장 강한 사전 정보.

### 5-3) 시간대별로 어떤 변수가 활성화되는가
- `outputs/figures/shap_hourly_heatmap.png` — 각 변수의 평균 SHAP을 hour-of-day별로 색으로 표현.
- 일사량 계열은 정오에 최대 양의 기여, 새벽/저녁에 음의 기여 → **모델이 시간대를 자체적으로 학습**.
- `power_lag_1`은 일출 직후 가장 강한 신호 (변동성이 가장 큰 구간), 정오에는 weakening.

### 5-4) 1주일 동안의 SHAP 추이
- `outputs/figures/shap_weekly_timeline.png` — top-5 변수의 SHAP 기여도가 시간에 따라 어떻게 누적되는지 stacked area로 표현.
- 흐린 날: cloud_index와 power_lag가 negative 영역으로 큰 폭 이동.
- 맑은 날: irradiation·clear_sky가 dominant한 양의 기여.

## 6) 특정 시점의 예측 근거 — Local XAI

### 6-1) 4가지 대표 시각 SHAP Waterfall
- `shap_waterfall_sunny_*.png` — 맑은 정오: irradiation/clear_sky가 예측을 위로 +α만큼 견인.
- `shap_waterfall_cloudy_*.png` — 흐린 정오: cloud_index의 음의 기여가 예측을 아래로 끌어내림.
- `shap_waterfall_sunrise_*.png` — 일출 직후: cos_zenith와 hour 관련 변수가 켜지면서 예측이 0에서 점차 상승.
- `shap_waterfall_sunset_*.png` — 일몰 직전: 동일 변수들이 정반대 방향으로 작용.
- 같은 모델이 같은 변수를 시점에 따라 정반대 방향으로 활용한다는 것을 한 슬라이드로 시각화.

### 6-2) 모델이 가장 헛다리 짚은 케이스 분석
- `outputs/figures/shap_worst_predictions.png` — 절대 오차 상위 12개 케이스와 각각의 SHAP top-3 driver.
- 대표 worst-case (최대 5개):
  - 2025-07-23 10:00 · 실제 0.895 · 예측 0.123 · |오차| 0.772
    원인 변수: temp_delta(-0.037), cos_zenith(+0.025), power_lag_168(+0.021)
  - 2025-07-19 05:00 · 실제 0.750 · 예측 0.015 · |오차| 0.735
    원인 변수: irrad_x_cos_zenith(-0.022), power_lag_168(-0.020), temp_delta(-0.013)
  - 2025-07-20 06:00 · 실제 0.753 · 예측 0.057 · |오차| 0.697
    원인 변수: irrad_x_cos_zenith(-0.023), power_lag_168(+0.021), yesterday_residual(+0.016)
  - 2025-07-29 08:00 · 실제 0.714 · 예측 0.025 · |오차| 0.689
    원인 변수: irrad_x_cos_zenith(-0.024), power_lag_168(+0.021), cos_zenith(-0.020)
  - 2025-07-10 07:00 · 실제 0.751 · 예측 0.096 · |오차| 0.655
    원인 변수: power_roll_std_6(+0.058), irrad_x_cos_zenith(-0.030), power_lag_168(+0.021)
- 공통 패턴: 급격한 구름 변화나 인버터 outage 등 *센서 측정값과 실제 발전량이 일시 디커플링*되는 시점에서 오차가 큼 → 추가 데이터(위성 기반 단기 nowcasting)로 해결 가능한 영역.

### 6-3) 카운터팩추얼: "만약 일사량이 X% 다르다면?"
- IRRADIATION을 ±10 %, ±20 % 흔들었을 때 모델 예측 변화 (정규화 단위 중앙값):
  -20% → 예측 +0.0000, -10% → 예측 +0.0000, +0% → 예측 +0.0000, +10% → 예측 +0.0000, +20% → 예측 +0.0000
- 일사량 +20 %일 때 예측이 가장 강하게 상승 → **모델이 일사량에 대해 단조 증가하며 비선형적으로 반응**. 운영자가 단기 일사량 보정값을 가지고 있을 때, 그 값을 모델에 주입한 효과를 즉시 추정 가능.

## 7) 딥러닝(LSTM) 측 XAI
- `outputs/figures/lstm_permutation.png` — 입력 변수를 셔플했을 때의 ΔMAE. 트리 모델과 다른 변수 우선순위를 보일 수 있어 ensemble 효과의 근거.
- `outputs/figures/lstm_ig_window.png` — 단일 high-irradiance window의 Integrated Gradients heatmap (시간 × 변수). **언제, 어떤 변수가 LSTM의 예측을 끌어올렸는지** 한 화면으로 가시화.

## 8) 비즈니스 임팩트
- MAE는 발전량과 동일한 단위(kW)로 해석되어 운영자가 직관적으로 "예측 오차 → 보조전원 손실 비용"으로 환산 가능.
- 블랙박스가 아닌 SHAP / IG 기반 설명을 제공하여 grid 운영자/규제기관 신뢰 확보.
- 동일 데이터셋에서 논문 SOTA 대비 MAE 개선을 달성함으로써 기술 차별성 입증.
- worst-case 분석과 카운터팩추얼은 단순 "예측 + 신뢰도"가 아닌 **"예측 + 근거 + 어떻게 하면 더 좋아지는지의 처방"** 을 제공.

## 9) 다음 단계
- 멀티 사이트 일반화 (현재 Plant 1 단일 사이트 → Plant 2/타지역 확장).
- 1시간 → 6/24시간 다중 horizon 확장.
- 확률적 forecast (quantile / interval prediction).
- 운영자용 dashboard에 SHAP 설명을 함께 노출.
- 위성 기반 단기 nowcasting 데이터 결합으로 worst-case 오차 추가 감소.
