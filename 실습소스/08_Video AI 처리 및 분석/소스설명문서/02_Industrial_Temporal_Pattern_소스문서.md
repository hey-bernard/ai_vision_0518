# 02 Industrial Temporal Pattern — 소스문서

> **대상:** Video AI 초보자 (01 Temporal Vision 수강 후)  
> **원본:** `02_Industrial_Temporal_Pattern.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` → `IPAD_Sample/` (R01, 클립 01)

---

## 강의 전체 흐름

| 섹션 | 주제 | 한 줄 요약 |
|------|------|-----------|
| 0~1 | IPAD 데이터 | 설비 프레임 시퀀스 구조·로드 |
| 2 | Motion Energy | 프레임 차분으로 동작 강도 측정 |
| 3~4 | 주기·사이클 | Autocorr/FFT로 1사이클 분할 |
| 5 | Optical Flow | peak 구간 부품 이동 시각화 |
| 6~11 | 이상 탐지 | Z-score, STFT, DTW, CUSUM, Mahalanobis, Phase |

---

## 0~1. IPAD 데이터셋

### Vision·Video AI 전문 용어 (강의용)

**IPAD(Industrial Process Anomaly Detection)**  
산업 공정(설비) 영상에서 **이상(anomaly)** 을 탐지하기 위한 공개 데이터셋입니다(Liu et al., arXiv:2404.15033). 교통·보행 VAD와 달리, 설비가 **주기적으로 같은 동작을 반복**한다는 점이 핵심 가정입니다. mp4가 아니라 `000.jpg, 001.jpg, …` **프레임 이미지 시퀀스**로 제공되어, 프레임 단위 라벨·분석에 유리합니다.

**Video Anomaly Detection, VAD(비디오 이상 탐지)**  
정상 패턴을 학습하거나 기준(baseline)으로 삼고, 그와 다른 구간·프레임을 찾는 기술입니다. 산업 VAD는 "사람이 이상 행동"보다 "설비 사이클이 어긋남·동작 강도 이상"을 탐지하는 경우가 많습니다.

**Periodicity(주기성)**  
같은 동작 패턴이 일정 간격으로 반복되는 성질입니다. 컨베이어 1회전, 프레스 1사이클처럼 IPAD 설비 VAD에서 **가장 중요한 Temporal 특징**입니다. 주기 T를 알면 "정상 1사이클"과 test 사이클을 비교할 수 있습니다.

**Temporal Pattern(시간 패턴)**  
시간에 따라 반복·변화하는 동작 패턴 전체를 가리킵니다. Motion Energy 시계열의 peak/trough 반복, Optical Flow 방향 패턴 등이 temporal pattern의 표현입니다.

### IPAD 구조

```
IPAD_Sample/R01/
  training/frames/01/*.jpg   ← 정상 baseline
  testing/frames/01/*.jpg    ← 이상 포함 가능
```

### 소스코드 핵심

- `resolve_ipad_session()` — zip 해제, train/test 클립
- `IpadClip` — device, split, clip_id, frame paths
- `try_load_anomaly_labels()` — frame_labels_{device}.npy (선택)

---

## 2. Motion Energy — 설비 동작 강도

### Vision·Video AI 전문 용어 (강의용)

**Motion Energy(모션 에너지)**  
연속 프레임 그레이스케일 차분 `|F(t+1) − F(t)|`의 평균으로, "그 순간 화면이 얼마나 변했는가"를 하나의 숫자로 요약합니다. 값이 크면 움직임 peak, 작으면 정지 trough입니다. 01번과 동일 개념이며, 산업 설비에서는 **주기적으로 peak/trough가 반복** → 이후 Autocorr·사이클 분할의 1차 입력이 됩니다.

**Flow Magnitude(플로우 매그니튜드)**  
Farneback Dense Optical Flow로 구한 이동 벡터의 **크기(magnitude)** 평균입니다. Motion Energy가 "픽셀 값이 얼마나 달라졌는가"라면, Flow Magnitude는 "픽셀이 실제로 얼마나, 어느 방향으로 움직였는가"에 가깝습니다. 두 지표를 함께 보면 단순 조명 변화 vs 실제 기구 이동을 구분하는 데 도움이 됩니다.

### 소스코드

- `compute_motion_energy()`, `compute_flow_magnitude_mean()`
- `plot_motion_signals()` → `01_motion_energy_train.png`

---

## 3~4. 주기 추정 · 사이클 분할

### Vision·Video AI 전문 용어 (강의용)

**Period / Cycle(주기 / 사이클)**  
Period는 같은 패턴이 반복되는 **간격**(프레임 수 또는 초). Cycle은 그 간격 1회분의 동작 전체(예: 리프트 1회 상승·하강)입니다. IPAD 설비 1사이클은 보통 5~30초(30 FPS 기준 150~900프레임)입니다.

**Period Memory(IPAD 논문 개념)**  
한 주기 단위로 정상 패턴을 "기억"하고, 다음 사이클이 그 패턴과 얼마나 다른지 비교하는 VAD 아이디어입니다. 본 실습의 사이클 overlay·DTW template이 이 개념의 규칙 기반 버전입니다.

**Keyframe(키프레임, 대표 프레임)**  
사이클 내 대표 순간(시작, 1/4, 1/2, 3/4)의 프레임입니다. 긴 사이클을 4장으로 요약해 육안 확인할 때 씁니다.

> Autocorrelation·FFT·segment_cycles 등 **신호 처리 기법**은 노트북 코드 주석과 그래프를 함께 설명하세요. lag=T에서 peak → T프레임마다 반복.

### 소스코드

- `estimate_period_autocorr()`, `estimate_period_fft()`
- `segment_cycles()`, `plot_cycle_overlay()`, `visualize_cycle_keyframes()`

---

## 5. Optical Flow — 설비 부품 이동

### Vision·Video AI 전문 용어 (강의용)

**Optical Flow(광학 흐름)**  
01번과 동일. 연속 프레임에서 픽셀 이동 벡터 (dx, dy)를 추정합니다.

**Farneback Dense Flow**  
모든 픽셀의 이동량을 계산합니다. Motion Energy **peak 구간**(설비가 가장 활발히 움직일 때)에서 flow를 계산하면, 부품 이동 **방향·속력 패턴**이 가장 뚜렷합니다.

**HSV Flow 시각화**  
Hue=이동 **방향**, Value=이동 **크기(속력)**. Pseudo-color로 흑백 flow를 색상 맵에 올려, "어느 방향으로 얼마나 빠르게 움직이는가"를 한 장에 표현합니다.

**ROI(Region of Interest, 관심 영역)**  
화면 중 특정 부품·작업 구역만 분석할 때 쓰는 영역입니다. 전체 화면 변화 대신 ROI 내부 flow만 보면 특정 부품 이상에 집중할 수 있습니다.

### 소스코드

- `visualize_peak_motion_flow()` — peak_idx에서 Farneback + HSV
- 저장: `05_optical_flow_peak.png`

---

## 6. 정상 vs 이상 — Z-score

### Vision·Video AI 전문 용어 (강의용)

**Baseline(기준선)**  
training(정상) 클립에서 계산한 평균·표준편차 분포입니다. test 사이클이 이 baseline에서 얼마나 벗어났는지가 이상 점수가 됩니다.

**Reconstruction Error(복원 오차)**  
딥러닝 VAD(IPAD 논문 등)에서 정상만으로 학습한 autoencoder가 test를 복원할 때의 오차입니다. 본 실습의 Z-score(사이클 평균 motion 편차)는 이 개념과 **직관적으로 대응**하는 규칙 기반 버전입니다.

**Period Classification(IPAD 논문 VAD)**  
추정한 주기 T 단위로 사이클 패턴을 분류·비교해 이상을 찾는 논문의 핵심 아이디어입니다.

판정: train 사이클 motion 평균 → μ, σ → test z = (x−μ)/σ → |z|>2 이상 후보

### 소스코드

- `cycle_energy_profile()`, `compare_train_test_patterns()`

---

## 7~11. STFT · DTW · CUSUM · Mahalanobis · Phase

### Vision·Video AI / 산업 VAD 맥락 용어

**STFT(단시간 푸리에 변환) 스펙트로그램**  
Motion Energy를 짧은 윈도우마다 FFT해 "그 순간 어떤 주파수(반복 속도)가 강한가" 2D 히트맵으로 봅니다. **전역 Autocorr**(영상 전체 대표 주기) vs **국소 STFT**(시간 구간별 주기 변화) 차이가 핵심입니다. stripe가 흐릿해지면 주기 drift·과도(transient) 이상 단서입니다.

**DTW(Dynamic Time Warping, 동적 시간 정렬)**  
두 시계열 길이·속도가 달라도 시간축을 **비선형으로 늘리고 줄여** 맞춘 뒤 유사도를 측니다. Z-score가 사이클 **평균 1개 숫자**만 비교한다면, DTW는 **사이클 전체 곡선 shape**를 비교합니다. peak가 몇 프레임 밀려도 warp로 흡수해 산업 주기 비교에 적합합니다.

**CUSUM(누적합 변화점 탐지)**  
baseline에서 한쪽으로 **지속적으로** 벗어나면 누적합 S+, S−가 threshold를 넘어 **change-point(변화점)** alarm을 냅니다. "어느 사이클이 이상인가"보다 "**언제부터** 패턴이 달라졌는가"에 답합니다.

**Mahalanobis Distance(마할라노비스 거리)**  
한 사이클을 9차원 feature(mean motion, peak 위치, flow 통계 등)로 요약한 뒤, train 분포의 **평균·공분산**을 반영한 통계적 거리입니다. feature 간 **상관관계**까지 고려해 다변량 이상을 탐지합니다.

**Operational Phase(운전 단계)**  
한 사이클 Motion Energy에서 peak를 기준으로 Idle→Accel→Peak/Work→Decel 4단계로 나눕니다. "전체 사이클이 이상" vs "**Peak/Work 단계만** 이상"을 구분할 수 있습니다.

### 이상 탐지 기법 비교

| 기법 | 잡아내는 이상 |
|------|-------------|
| Z-score | 사이클 전체 강도 편차 |
| STFT | 국소 주기 drift |
| DTW | 곡선 형태·타이밍 어긋남 |
| CUSUM | change-point (시점) |
| Mahalanobis | 다변량 복합 이상 |
| Phase | 특정 운전 단계 이상 |

---

## 01과의 관계

| | 01 Temporal Vision | 02 Industrial Pattern |
|---|-------------------|----------------------|
| 목적 | 일반 행동 인식 | 설비 **주기·이상** |
| 공통 | Motion Energy, Optical Flow | 동일 |
| 01만 | 3D CNN, TimeSformer, ViViT | — |
| 02만 | — | Autocorr, STFT, DTW, CUSUM, Phase |

---

## 부록: 출력 파일

`output_ipad/figures/` — `01_motion_energy_train.png` ~ `15_phase_heatmap.png`

---

## FAQ

**Q. training vs testing?**  
A. training=정상 baseline, testing=이상 포함 가능. test를 train baseline과 비교합니다.

**Q. DTW가 Z-score보다 나은 경우?**  
A. peak가 몇 프레임 밀려도 warp로 흡수. **곡선 형태**가 중요한 산업 설비에 유리합니다.

---

*본 문서는 `02_Industrial_Temporal_Pattern.ipynb` 소스문서로 작성되었습니다.*
