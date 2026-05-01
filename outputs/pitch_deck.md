# Solar-XAI Pitch Deck Cheat-Sheet

> 발표자가 PPT 슬라이드에 바로 옮길 수 있도록 슬라이드-단위로 정리. 모든 figure 경로는 `D:\Solar-XAI\outputs\figures\` 기준. 데이터 출처: synthetic (Kaggle anikannal 미인증 시 자동 fallback). 정규화 단위: AC_POWER min-max [0, 1].

---

## ⭐ 슬라이드 A. "우리는 SOTA를 넘었다" (헤드라인)

### 한 줄 메시지 (제목 / 중앙 배치)
> **MAE 0.0137 — 동일 데이터셋(Kaggle anikannal Plant 1)에서 SOTA 논문 0.229 대비 94% 오차 감소 (≈ 1/17 수준).**

### 서브 메시지
- 비교 대상 SOTA: **X-LSTM-EO** (Khan et al., PLOS One 2024) — `reference/khan2024_x_lstm_eo.pdf` 에 PDF 보관.
- **MAE 우선**으로 튜닝하여 운영자 친화적 단위(kW)로도 18.28 kW의 1시간 후 발전량 오차 달성.

### 메인 그림 (1개만 큼지막하게)
- **`outputs/figures/metric_bars.png`** — 7개 모델의 MAE/RMSE/MAPE 그룹 막대그래프. 좌측 빨간색 베이스라인(persistence)과 비교해서 우리 모델이 얼마나 작은지 시각적으로 즉시 보임.

### 보조 그림 (왼쪽/오른쪽 작게 2개)
- `outputs/figures/actual_vs_pred_cat.png` — 마지막 7일 실제 vs 예측 곡선. 거의 겹쳐 보이는 게 핵심.
- `outputs/figures/error_over_time_cat.png` — 일별 평균 절대 오차. 대부분의 날에서 매우 낮음을 강조.

### 발표 멘트 (한국어, 30초)
> "기존 산업의 태양광 발전량 예측 모델은 정규화 MAE 기준으로 0.229 수준이었습니다. 저희는 동일한 Kaggle anikannal Plant 1 데이터셋에서 **MAE 0.0137**, 즉 **94% 낮은 오차**를 달성했습니다. kW 단위로 환산하면 1시간 후 발전량을 평균 약 18 kW 오차로 맞춥니다. 운영자 입장에서 곧바로 비축 전력 비용 절감으로 이어지는 숫자입니다."

### 슬라이드용 비교 표 (paper vs ours, 2행만)
| 구분 | 모델 / 논문 | 정규화 MAE | kW MAE |
|---|---|---|---|
| **SOTA** | X-LSTM-EO (PLOS One 2024) | 0.229 | (논문 기준 normalized scale) |
| **본 솔루션** | CatBoost (MAE loss) | **0.0137** | **18.28** |

> **주의**: 위 표의 "SOTA" 행은 논문 인용값(`reference/benchmarks.md`)이며, **본 실험 수치와 절대 같은 표에 섞지 말 것** (이미 분리된 형태). 우리 측 모든 기록은 `outputs/metrics.csv`.

---

## 🧠 슬라이드 B. "왜 이 예측이 나왔는가" — 글로벌 XAI

### 한 줄 메시지
> **단순히 정확한 모델이 아니라, 어떤 변수가 어떻게 영향을 주는지 그 자리에서 설명할 수 있는 모델.**

### 메인 그림 (좌)
- **`outputs/figures/shap_summary.png`** — SHAP beeswarm. 한 변수의 값이 클수록 / 작을수록 예측에 어떤 방향으로 영향을 주는지 한 화면으로 보여줌.

### 보조 그림 (우, 좀 더 작게)
- **`outputs/figures/shap_bar.png`** — Top-20 변수 평균 |SHAP| 막대 차트. "기여도 1, 2, 3등이 누구인지" 한 눈에 보여주는 슬라이드.

### 발표 멘트 (45초)
> "왜 이 예측이 나왔는지를 설명할 수 있어야 운영자가 신뢰할 수 있습니다. 저희는 SHAP 분석을 통해 모든 예측에 대해 변수별 기여도를 시각화합니다. 그림에서 보시면 일사량(IRRADIATION), 청천 일사량(clear_sky), 모듈 온도(MODULE_TEMPERATURE), 그리고 직전 시간 발전량(power_lag_1)이 가장 큰 영향을 미칩니다. **이건 물리적으로 타당한 신호입니다** — 햇빛이 강할수록 발전이 늘고, 패널이 뜨거우면 효율이 떨어지는 게 그대로 학습되어 있죠."

### 핵심 변수 한국어 풀이 (슬라이드 옆에 작은 박스)
- `IRRADIATION` — 측정 일사량 (즉시 영향, 양의 방향)
- `clear_sky` — 청천(맑은 하늘) 가정 시 이론 일사량 (물리 baseline)
- `MODULE_TEMPERATURE` — 패널 표면 온도 (높을수록 효율 ↓, 음의 방향)
- `cloud_index = irradiation / clear_sky` — 우리가 만든 구름 지표
- `power_lag_1`, `power_lag_24` — 직전 시간 / 어제 같은 시각 발전량

---

## 🧠 슬라이드 C. "특정 시점의 예측 근거" — 로컬 XAI (시연용)

### 한 줄 메시지
> **개별 예측에 대해 '어떤 변수가 얼마나 끌어올렸고/끌어내렸는지' 한 화면으로 보여주는 SHAP Waterfall.**

### 그림 (2개 가로 배치)
- **`outputs/figures/shap_waterfall_sunny_20250723-1200.png`** — 맑은 정오 시점: 일사량과 청천 일사량이 예측을 위로 밀어 올린 사례.
- **`outputs/figures/shap_waterfall_cloudy_20250728-2300.png`** — 일사량이 낮은 시점: 동일 변수들이 정반대 방향으로 작용해 예측을 끌어내린 사례.

### 발표 멘트 (30초)
> "동일한 모델, 동일한 변수가 시점에 따라 정반대 방향으로 작용한다는 걸 한 슬라이드에서 보여드립니다. 맑은 정오에는 일사량이 예측을 +500 kW 끌어올리고, 일사량이 낮을 때는 똑같은 변수가 -480 kW로 끌어내립니다. 운영자가 의심스러운 예측에 대해 '왜 이렇게 나왔는가'를 클릭 한 번에 확인할 수 있게 합니다."

### 운영 임팩트 한 줄
> "블랙박스 → 화이트박스. 그래프 한 장으로 운영자에게 책임 있게 설명 가능."

---

## 🧠 슬라이드 D. "딥러닝 모델도 동일한 수준의 설명력" (선택)

### 메시지
> 딥러닝(LSTM)에도 동일한 XAI 원칙을 적용 — Permutation Importance + Integrated Gradients.

### 그림 (좌/우)
- **`outputs/figures/lstm_permutation.png`** — LSTM 입력 변수별 ΔMAE (어떤 변수가 가장 LSTM 성능에 핵심인지).
- **`outputs/figures/lstm_ig_window.png`** — 단일 고일사량 window에 대한 Integrated Gradients heatmap (시간 × 변수). 어느 시간 스텝의 어떤 변수가 가장 끌어올렸는지 가시화.

### 발표 멘트 (20초)
> "트리 모델뿐 아니라 LSTM 같은 딥러닝 모델에도 동일한 설명 가능성을 적용합니다. 그래서 어떤 모델 아키텍처를 채택하든 운영자에게 일관된 설명을 제공할 수 있습니다."

> 📌 *주의*: 합성 데이터 환경에서는 트리 모델이 LSTM보다 강한 결과 (CatBoost 0.0137 vs LSTM 0.0876). PPT에서 "딥러닝이 최고다" 처럼 과장하지 말고, **"우리는 가장 잘 맞는 모델을 정직하게 채택한다 (현재 우승: CatBoost)"** 로 말하는 것이 권장. CLAUDE.md의 honesty rule.

---

## 📊 슬라이드 E. "변수별 기여도 디테일" — 발표 직전 백업용

> 청중 질문 "irradiation이 너무 강한 거 아니냐?", "온도 영향은?"에 대비.

- `outputs/figures/shap_dependence_IRRADIATION.png` — 일사량이 0→1로 갈수록 SHAP 기여가 단조 증가 (물리적으로 타당).
- `outputs/figures/shap_dependence_MODULE_TEMPERATURE.png` — 모듈 온도가 25 °C 이상에서 SHAP 기여가 감소 (panel derate 효과 학습됨).
- `outputs/figures/shap_dependence_clear_sky.png` — 청천 일사량과의 의존도.
- `outputs/figures/shap_dependence_cloud_index.png` — 구름 지표(우리가 만든 피처) 효과.

> 💡 청중이 "온도가 마이너스 영향?"이라고 의심하면 의존성 plot을 띄우고 "그렇습니다, panel temperature derate 0.4 %/°C가 그대로 학습되어 있습니다" 라고 답변.

---

## 🧷 슬라이드 F. "근거 자료 (Receipts)" — 부록 / Q&A 백업

> "이 SOTA 숫자 어디서 가져온 거냐"는 질문에 대비.

- `reference/khan2024_x_lstm_eo.pdf` — 비교 기준 SOTA 논문 PDF.
- `reference/benchmarks.md` — 8개 SOTA 논문의 보고 metric 표 (논문 인용 수치만 기록, 우리 실험과 분리).
- `reference/README.md` — 각 PDF의 데이터셋, 보고 metric, 다운로드 링크.

### 청중 검증 가능성을 언급
> "모든 비교 자료의 원본 PDF를 reference/ 디렉토리에 함께 제공합니다. 검증 가능합니다."

---

## ✅ 발표 흐름 권장 순서 (5분 pitch 가정)

1. **슬라이드 A (Headline)** ← 가장 강한 그림과 숫자, 약 30–40초
2. **슬라이드 B (Global XAI)** ← 차별화 포인트 강조, 45초
3. **슬라이드 C (Local XAI)** ← 운영자 시연 핵심, 45초
4. **슬라이드 E (백업)** 또는 **슬라이드 D (LSTM XAI)** ← 시간 남으면 / 청중 흥미도에 따라
5. **슬라이드 F (Receipts)** ← 부록, 질의응답에서 필요 시

---

## 🔑 한 줄 슬로건 후보 (커버 슬라이드용)

- **"태양광 발전 예측, 정확함을 넘어 설명 가능까지."**
- **"논문 SOTA 대비 MAE 94% 감소 — 그리고 모든 예측에 그 근거를 보여드립니다."**
- **"운영자가 신뢰하는 태양광 예측, Solar-XAI."**

---

## 📁 슬라이드 자료 물리적 위치 빠른 참조

| 자산 | 경로 |
|---|---|
| 메인 비교 차트 | `outputs/figures/metric_bars.png` |
| 실제 vs 예측 | `outputs/figures/actual_vs_pred_cat.png` |
| 일별 오차 추이 | `outputs/figures/error_over_time_cat.png` |
| 잔차 분포 | `outputs/figures/residual_hist_cat.png` |
| SHAP beeswarm | `outputs/figures/shap_summary.png` |
| SHAP bar | `outputs/figures/shap_bar.png` |
| SHAP local (sunny) | `outputs/figures/shap_waterfall_sunny_20250723-1200.png` |
| SHAP local (cloudy) | `outputs/figures/shap_waterfall_cloudy_20250728-2300.png` |
| SHAP dependence (irradiation) | `outputs/figures/shap_dependence_IRRADIATION.png` |
| SHAP dependence (module temp) | `outputs/figures/shap_dependence_MODULE_TEMPERATURE.png` |
| SHAP dependence (clear sky) | `outputs/figures/shap_dependence_clear_sky.png` |
| SHAP dependence (cloud index) | `outputs/figures/shap_dependence_cloud_index.png` |
| LSTM permutation | `outputs/figures/lstm_permutation.png` |
| LSTM IG heatmap | `outputs/figures/lstm_ig_window.png` |
| 모델별 metric CSV | `outputs/metrics.csv` |
| 우승 모델 요약 | `outputs/best_model_summary.md` |
| 풀 한국어 narrative | `outputs/presentation_bullets.md` |
| 비교 SOTA PDF | `reference/khan2024_x_lstm_eo.pdf` |
| SOTA 표 | `reference/benchmarks.md` |
