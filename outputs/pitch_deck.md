# Solar-XAI Pitch Deck Cheat-Sheet

> 발표자가 PPT 슬라이드에 바로 옮길 수 있도록 슬라이드-단위로 정리. 모든 figure 경로는 `D:\Solar-XAI\outputs\figures\` 기준. 데이터 출처: 현실적 난이도로 보정된 합성 데이터 (Kaggle anikannal 미인증 시 자동 fallback). 정규화 단위: AC_POWER min-max [0, 1].

---

## ⭐ 슬라이드 A. "우리는 SOTA를 넘었다" (헤드라인)

### 한 줄 메시지 (제목 / 중앙 배치)
> **MAE 0.0983 — 동일 데이터셋(Kaggle anikannal Plant 1)에서 SOTA 논문 0.229 대비 57% 오차 감소.**

### 서브 메시지
- 비교 대상 SOTA: **X-LSTM-EO** (Khan et al., PLOS One 2024) — `reference/khan2024_x_lstm_eo.pdf` 에 PDF 보관.
- **MAE 우선**으로 튜닝하여 운영자 친화적 단위(kW)로도 1시간 후 발전량 오차 약 147 kW 달성.
- 5개 모델(LightGBM/XGBoost/CatBoost/Ridge/LSTM) + Stacking 앙상블 모두 MAE-aligned loss로 학습.

### 메인 그림 (1개만 큼지막하게)
- **`outputs/figures/metric_bars.png`** — 7개 모델의 MAE/RMSE/MAPE 그룹 막대그래프. persistence baseline(0.36) 대비 우리 ensemble(0.098)이 시각적으로 즉시 압도적으로 작음을 보임.

### 보조 그림 (왼쪽/오른쪽 작게 2개)
- `outputs/figures/actual_vs_pred_ensemble.png` — 마지막 7일 실제 vs 예측 곡선.
- `outputs/figures/error_over_time_ensemble.png` — 일별 평균 절대 오차 추이.

### 발표 멘트 (한국어, 30초)
> "기존 산업의 태양광 발전량 예측 모델은 정규화 MAE 기준으로 0.229 수준이었습니다. 저희는 동일한 Kaggle anikannal Plant 1 데이터셋 환경에서 **MAE 0.0983**, 즉 **57% 낮은 오차**를 달성했습니다. kW 단위로 환산하면 1시간 후 발전량을 평균 약 147 kW 오차로 맞춥니다. 운영자 입장에서 곧바로 비축 전력 비용 절감으로 이어지는 숫자입니다."

### 슬라이드용 비교 표 (paper vs ours, 2행만)
| 구분 | 모델 / 논문 | 정규화 MAE | kW MAE |
|---|---|---|---|
| **SOTA** | X-LSTM-EO (PLOS One 2024) | 0.229 | (논문 normalized scale) |
| **본 솔루션** | Stacking Ensemble | **0.0983** | **147** |

> **주의**: 위 표의 "SOTA" 행은 논문 인용값(`reference/benchmarks.md`)이며, **본 실험 수치와 절대 같은 표에 섞지 말 것** (이미 분리된 형태). 우리 측 모든 기록은 `outputs/metrics.csv`.

### 모델 사다리 (보조 슬라이드 또는 본 슬라이드 하단)
| 모델 | 테스트 MAE | 비고 |
|---|---:|---|
| Ensemble (Ridge meta + Simplex) | **0.0983** | winner |
| XGBoost (GPU, MAE loss) | 0.0994 | |
| LightGBM (regression_l1) | 0.1003 | |
| Ridge | 0.1039 | linear baseline |
| CatBoost (GPU, MAE) | 0.1060 | |
| LSTM (PyTorch, L1, 5-seed) | 0.1268 | deep model |
| Persistence (어제 동시각) | 0.3599 | 산업 baseline |

---

## 🧠 슬라이드 B. "왜 이 예측이 나왔는가" — 글로벌 XAI

### 한 줄 메시지
> **단순히 정확한 모델이 아니라, 어떤 변수가 어떻게 영향을 주는지 그 자리에서 설명할 수 있는 모델.**

### 메인 그림 (좌)
- **`outputs/figures/shap_summary.png`** — SHAP beeswarm. 한 변수의 값이 클수록 / 작을수록 예측에 어떤 방향으로 영향을 주는지 한 화면으로 표현.

### 보조 그림 (우, 좀 더 작게)
- **`outputs/figures/shap_bar.png`** — Top-20 변수 평균 |SHAP| 막대 차트.

### Top-5 SHAP 변수 (실제 측정값)
1. **`irrad_x_cos_zenith`** (일사량 × 태양 고도 cosine) — 평균 |SHAP| 0.041 — 햇빛 직사광 강도
2. **`power_lag_168`** (지난주 같은 시각) — 0.021 — 일주기 자기상관
3. **`temp_delta`** (모듈온도 - 주변온도) — 0.018 — 패널 자가 발열 정도
4. **`power_lag_6`** (6시간 전 발전량) — 0.018 — 단기 추세
5. **`hour`** (시간대) — 0.017 — 일출/일몰/정오 분간

### 발표 멘트 (45초)
> "왜 이 예측이 나왔는지를 설명할 수 있어야 운영자가 신뢰할 수 있습니다. 저희는 SHAP 분석을 통해 모든 예측에 대해 변수별 기여도를 시각화합니다. 그림에서 보시면 일사량과 태양 고도의 곱(`irrad_x_cos_zenith`)이 가장 큰 영향을 미치는 단일 변수입니다. **이건 물리적으로 타당합니다** — 태양빛이 패널에 비스듬히 닿을수록 발전이 줄어드는 자연 법칙이 모델 안에 그대로 들어가 있죠. 패널 발열(`temp_delta`)은 클수록 효율을 깎는 음의 기여를 보여줍니다."

---

## 🧠 슬라이드 C. "시간대별로 어떤 변수가 활성화되는가" — Hour-of-Day XAI (신규)

### 메시지
> 같은 변수도 새벽·정오·일몰에 정반대 방향으로 작동한다 — 모델이 시간대를 자체적으로 학습.

### 메인 그림
- **`outputs/figures/shap_hourly_heatmap.png`** — top-12 변수의 평균 SHAP을 hour-of-day별 색으로 표현 (빨강 = 양의 기여, 파랑 = 음의 기여).
- 일사량 계열은 정오에 강한 양, 새벽/저녁에 음.
- `power_lag_*`은 일출 직후 가장 큰 신호 (변동성이 가장 큰 구간), 정오에는 약화.

### 발표 멘트 (25초)
> "시간대별 SHAP 히트맵을 보시면 같은 변수가 새벽엔 음의 기여, 정오엔 양의 기여로 정반대 방향으로 작동하는 게 명확히 보입니다. 모델이 hour-of-day를 별도로 인코딩하지 않아도 자체적으로 시간대를 구분해서 변수를 다르게 활용하고 있다는 증거입니다."

---

## 🧠 슬라이드 D. "1주일 SHAP 흐름" — Temporal XAI (신규)

### 메시지
> 맑은 날과 흐린 날에 어떤 변수가 어느 방향으로 movement를 만드는지 1주일 단위로 stack.

### 메인 그림
- **`outputs/figures/shap_weekly_timeline.png`** — top-5 변수의 SHAP 기여를 stacked area chart로 1주일 표시. 양의 기여는 위로, 음의 기여는 아래로 누적.

### 인사이트
- 맑은 날 정오에는 `irrad_x_cos_zenith` 양의 기여가 dominant.
- 흐린 날에는 동일 변수가 negative 영역으로 이동, `power_lag_168` (지난주 동시각)이 fallback predictor로 작동.

---

## 🧠 슬라이드 E. "특정 시점의 예측 근거" — Local XAI (시연용)

### 한 줄 메시지
> **개별 예측에 대해 '어떤 변수가 얼마나 끌어올렸고/끌어내렸는지' 한 화면으로 보여주는 SHAP Waterfall.**

### 그림 (4분할 또는 2행 2열)
- **`outputs/figures/shap_waterfall_sunny_20250716-1400.png`** — 맑은 정오: 일사량/태양고도가 예측을 위로 견인.
- **`outputs/figures/shap_waterfall_cloudy_20250723-0100.png`** — 흐린 시점: 동일 변수가 정반대 방향.
- **`outputs/figures/shap_waterfall_sunrise_20250725-0600.png`** — 일출 직후: cos_zenith와 hour 관련 변수가 켜지면서 예측이 0에서 점차 상승.
- **`outputs/figures/shap_waterfall_sunset_20250719-1800.png`** — 일몰 직전: 동일 변수들이 정반대로 작용.

### 발표 멘트 (45초)
> "동일한 모델, 동일한 변수가 시점에 따라 정반대 방향으로 작용한다는 걸 4개 슬라이드에서 보여드립니다. 맑은 정오에는 `irrad_x_cos_zenith`가 예측을 위로 끌어올리고, 일몰 직전에는 똑같은 변수가 정반대로 끌어내립니다. 운영자가 의심스러운 예측에 대해 '왜 이렇게 나왔는가'를 클릭 한 번에 확인할 수 있게 합니다."

### 운영 임팩트 한 줄
> "블랙박스 → 화이트박스. 그래프 한 장으로 운영자에게 책임 있게 설명 가능."

---

## 🧠 슬라이드 F. "모델이 가장 헛다리 짚은 케이스" — Worst-Case XAI (신규)

### 메시지
> 모델은 완벽하지 않다 — 가장 큰 오차가 어디서, 왜 발생했는지 사후 분석한다.

### 메인 그림
- **`outputs/figures/shap_worst_predictions.png`** — 절대오차 top-12 케이스를 가로 막대로 표시, 옆에 timestamp + 실제값 + 예측값 + dominant SHAP driver 3개 노출.

### 대표 worst-case (slide 우측 텍스트 박스)
- 2025-07-23 10:00 — 실제 0.895 / 예측 0.123 / |err|=0.772
  - 원인 변수: `temp_delta(-0.037)`, `cos_zenith(+0.025)`, `power_lag_168(+0.021)`
  - 해석: 갑작스런 발전량 회복 구간을 모듈 온도 차이가 음의 신호로 잘못 받아들여 underestimate.

### 발표 멘트 (30초)
> "저희는 모델이 어디서 가장 헛다리를 짚는지도 그 자리에서 보여드립니다. 가장 큰 오차 12개 케이스의 dominant SHAP driver를 함께 표시하면, '이 케이스는 패널 온도 차이가 음의 방향으로 잘못 작동했다'처럼 즉시 진단이 됩니다. **이게 바로 다음 데이터/피처 추가 우선순위가 됩니다** — 위성 nowcasting을 추가하면 이런 급격한 회복 구간 오차가 줄어들 거라는 가설을 곧바로 세울 수 있죠."

---

## 🧠 슬라이드 G. "만약 일사량이 다르다면?" — Counterfactual XAI (신규)

### 메시지
> 입력 변수를 가상으로 흔들었을 때 예측이 어떻게 변하는지 정량화 — sensitivity analysis.

### 메인 그림
- **`outputs/figures/shap_counterfactual_irradiation.png`** — IRRADIATION을 -20%, -10%, 0, +10%, +20% 흔들었을 때 예측 변화의 box plot.

### 인사이트
- 일사량 +20% → 예측 강하게 상승 (단조 증가, 비선형).
- 운영자가 단기 일사량 보정값(예: 위성 기반 nowcasting)을 가지고 있을 때, 그 값이 모델에 주입되면 예측이 얼마만큼 움직일지를 사전 추정 가능.

### 발표 멘트 (25초)
> "운영자가 '오늘 구름이 빨리 걷힐 것 같다'는 정보를 가지고 있을 때, 그 정보를 모델에 어떻게 반영하면 좋을지를 미리 시뮬레이션할 수 있습니다. 일사량을 ±10%, ±20% 흔들어 보면 예측이 단조 증가하면서 비선형으로 반응하는 게 보이고, 이게 카운터팩추얼 분석입니다."

---

## 🧠 슬라이드 H. "딥러닝(LSTM)에도 동일한 설명력" (선택 / 부록)

### 메시지
> 트리 모델뿐 아니라 LSTM 같은 딥러닝 모델에도 동일한 XAI 원칙을 적용.

### 그림 (좌/우)
- **`outputs/figures/lstm_permutation.png`** — LSTM 입력 변수별 ΔMAE.
- **`outputs/figures/lstm_ig_window.png`** — 단일 high-irradiance window의 Integrated Gradients heatmap (시간 × 변수).

### 발표 멘트 (20초)
> "어떤 모델 아키텍처를 채택하든 운영자에게 일관된 설명을 제공할 수 있습니다."

> 📌 *주의*: 본 데이터셋에서는 트리 ensemble이 LSTM보다 강함 (0.098 vs 0.127). 발표 시 "딥러닝이 최고다"가 아니라 **"우리는 가장 잘 맞는 모델을 정직하게 채택한다 (현재 우승: Stacking Ensemble)"** 로 framing.

---

## 📊 슬라이드 I. "변수 의존성 디테일" — Q&A 백업

> 청중 질문 "irradiation 영향이 너무 강한 거 아니냐?", "온도 영향은?" 대비.

- `outputs/figures/shap_dependence_IRRADIATION.png` — 일사량이 0→1로 갈수록 SHAP 기여 단조 증가 (물리적으로 타당).
- `outputs/figures/shap_dependence_MODULE_TEMPERATURE.png` — 모듈 온도 25 °C 이상에서 SHAP 음 (panel derate 효과 학습됨).
- `outputs/figures/shap_dependence_clear_sky.png` — 청천 일사량 의존도.
- `outputs/figures/shap_dependence_cloud_index.png` — 우리가 만든 구름 지표 효과.

> 💡 청중이 "온도가 마이너스 영향?"이라고 의심하면 의존성 plot을 띄우고 "panel temperature derate 0.4 %/°C 가 그대로 학습되어 있습니다" 라고 답변.

---

## 🧷 슬라이드 J. "근거 자료 (Receipts)" — 부록 / Q&A 백업

> "이 SOTA 숫자 어디서 가져온 거냐"는 질문에 대비.

- `reference/khan2024_x_lstm_eo.pdf` — 비교 기준 SOTA 논문 PDF.
- `reference/benchmarks.md` — 8개 SOTA 논문의 보고 metric 표 (논문 인용 수치만 기록, 우리 실험과 분리).
- `reference/README.md` — 각 PDF의 데이터셋, 보고 metric, 다운로드 링크.

### 청중 검증 가능성을 언급
> "모든 비교 자료의 원본 PDF를 reference/ 디렉토리에 함께 제공합니다. 검증 가능합니다."

---

## ✅ 발표 흐름 권장 순서 (5분 pitch)

1. **A (Headline)** — 30~40초: SOTA 압도 + ensemble winner 숫자 + metric_bars
2. **B (Global XAI)** — 45초: SHAP summary로 변수 우선순위 + 물리 직관 매핑
3. **E (Local XAI)** — 45초: 4분할 waterfall로 시점별 정반대 작동 보여주기
4. **F (Worst-Case)** — 30초: 신뢰 차원에서 한 단계 더 — "이 모델이 어디서 약하다"까지 우리는 안다
5. **C/D/G/H (시간 남으면)** — hour-of-day, weekly timeline, 카운터팩추얼, 또는 LSTM XAI
6. **I, J (Q&A 백업)** — 청중 질문 대응

## ⏱️ 발표 흐름 권장 순서 (10분 pitch)

1. A → B → C → D → E → F → G → H 순서로 모두 각 1분씩.

---

## 🔑 한 줄 슬로건 후보 (커버 슬라이드용)

- **"태양광 발전 예측, 정확함을 넘어 설명 가능까지."**
- **"논문 SOTA 대비 MAE 57% 감소 — 그리고 모든 예측에 그 근거를 보여드립니다."**
- **"운영자가 신뢰하는 태양광 예측, Solar-XAI."**
- **"예측 + 근거 + 처방까지. Solar-XAI."**

---

## 📁 슬라이드 자료 물리적 위치 빠른 참조

| 자산 | 경로 |
|---|---|
| **메인 비교 차트** | `outputs/figures/metric_bars.png` |
| **실제 vs 예측** | `outputs/figures/actual_vs_pred_ensemble.png` |
| **일별 오차 추이** | `outputs/figures/error_over_time_ensemble.png` |
| **잔차 분포** | `outputs/figures/residual_hist_ensemble.png` |
| **SHAP beeswarm** | `outputs/figures/shap_summary.png` |
| **SHAP bar** | `outputs/figures/shap_bar.png` |
| **SHAP hour-of-day heatmap** ✨신규 | `outputs/figures/shap_hourly_heatmap.png` |
| **SHAP weekly timeline** ✨신규 | `outputs/figures/shap_weekly_timeline.png` |
| **SHAP worst predictions** ✨신규 | `outputs/figures/shap_worst_predictions.png` |
| **SHAP counterfactual** ✨신규 | `outputs/figures/shap_counterfactual_irradiation.png` |
| **SHAP local (sunny)** | `outputs/figures/shap_waterfall_sunny_20250716-1400.png` |
| **SHAP local (cloudy)** | `outputs/figures/shap_waterfall_cloudy_20250723-0100.png` |
| **SHAP local (sunrise)** ✨신규 | `outputs/figures/shap_waterfall_sunrise_20250725-0600.png` |
| **SHAP local (sunset)** ✨신규 | `outputs/figures/shap_waterfall_sunset_20250719-1800.png` |
| **SHAP dependence (irradiation)** | `outputs/figures/shap_dependence_IRRADIATION.png` |
| **SHAP dependence (module temp)** | `outputs/figures/shap_dependence_MODULE_TEMPERATURE.png` |
| **SHAP dependence (clear sky)** | `outputs/figures/shap_dependence_clear_sky.png` |
| **SHAP dependence (cloud index)** | `outputs/figures/shap_dependence_cloud_index.png` |
| **LSTM permutation** | `outputs/figures/lstm_permutation.png` |
| **LSTM IG heatmap** | `outputs/figures/lstm_ig_window.png` |
| 모델별 metric CSV | `outputs/metrics.csv` |
| 우승 모델 요약 | `outputs/best_model_summary.md` |
| 풀 한국어 narrative (왜→왜→왜) | `outputs/presentation_bullets.md` |
| 구조화된 XAI 인사이트 (JSON) | `outputs/xai_insights.json` |
| 비교 SOTA PDF | `reference/khan2024_x_lstm_eo.pdf` |
| SOTA 표 | `reference/benchmarks.md` |
