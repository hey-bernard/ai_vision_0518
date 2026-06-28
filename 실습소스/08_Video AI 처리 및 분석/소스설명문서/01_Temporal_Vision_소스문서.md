# 01 Temporal Vision — 소스문서

> **대상:** Video AI를 처음 접하는 초보자  
> **원본:** `01_Temporal_Vision.ipynb`  
> **샘플 영상:** PyTorchVideo 공개 `archery.mp4` (양궁 동작 클립)

---

## 강의 전체 흐름

| 섹션 | 주제 | 한 줄 요약 |
|------|------|-----------|
| 0 | 환경 및 데이터 | 비디오를 읽고, 분석할 준비를 한다 |
| 1 | Spatiotemporal(시공간) | 공간(한 장면)과 시간(움직임)을 구분한다 |
| 2 | Optical Flow(광학 흐름) | 픽셀이 어디로 움직였는지 계산한다 |
| 3 | SlowFast | 느린 경로 + 빠른 경로로 효율적으로 본다 |
| 4 | 3D CNN | 시간 축까지 포함한 합성곱으로 행동을 인식한다 |
| 5 | Video Transformer | Attention으로 시공간 패턴을 학습한다 |
| 6 | 모델 비교 요약 | Classical → 3D CNN → Transformer 진화를 정리한다 |

---

## 0. 환경 및 데이터 준비

### 이 섹션에서 배우는 것

- 비디오 AI 실습에 필요한 라이브러리와 샘플 영상을 준비한다.
- 비디오를 프레임 단위로 읽고, 모델 입력 형식으로 변환하는 공통 도구 함수를 이해한다.

### Vision·Video AI 전문 용어 (강의용)

**Temporal Vision(템포럴 비전)**  
시간에 따라 변하는 영상(비디오)을 이해하는 컴퓨터 비전 분야입니다. 단일 사진은 "지금 화면에 무엇이 있는가"만 답하지만, Temporal Vision은 "어떻게 움직이고 변하는가"까지 함께 다룹니다. 행동 인식, 이상 탐지, 영상 요약 등 대부분의 Video AI가 이 범주에 속합니다.

**Spatiotemporal(시공간, 스페이시오템포럴)**  
Spatial(공간: 한 장면 안의 위치·형태)과 Temporal(시간: 프레임 간 변화)을 합친 개념입니다. "어디에 무엇이 있고, 시간이 지나면서 어떻게 바뀌는가"를 동시에 표현할 때 씁니다. 3D CNN, Video Transformer, Optical Flow 모두 시공간 정보를 다루는 방법입니다.

**Kinetics-400(키네틱스-400)**  
걷기, 양궁, 춤 등 400가지 사람 행동 클립으로 학습된 대표 비디오 데이터셋입니다. 본 실습의 R3D, TimeSformer, ViViT 등 대부분 모델이 이 데이터로 사전학습(pretrain)되어 있어, archery 영상에 "archery" 라벨을 예측할 수 있습니다.

### 소스코드 핵심 기능

| 변수/함수 | 역할 |
|-----------|------|
| `TIMESFORMER_NUM_FRAMES = 8` | TimeSformer 입력 프레임 수 |
| `VIVIT_NUM_FRAMES = 32` | ViViT 입력 프레임 수 |
| `R3D_NUM_FRAMES = 16` | R3D/MC3 입력 (112×112) |
| `SLOWFAST_ALPHA = 4` | Fast 경로가 Slow보다 4배 많은 프레임 |
| `download_sample_video()` | archery.mp4 다운로드 |
| `read_all_frames_bgr()` | 전체 프레임 BGR 배열 |
| `sample_frame_indices()` | 균등 간격 N개 인덱스 |
| `analyze_video_metadata()` | 해상도, FPS, 재생 시간 |

> **강의 포인트:** 모델마다 입력 프레임 수·해상도가 다릅니다. 영상 전체가 아니라 **고정 길이 clip**으로 잘라 넣습니다.

---

## 1. 시공간(Spatiotemporal) 정보

### 이 섹션에서 배우는 것

- 비디오를 **공간(Spatial)** 과 **시간(Temporal)** 두 관점으로 나누어 본다.
- 프레임 차분과 **Motion Energy**로 가장 단순한 시간 특징을 계산한다.

### Vision·Video AI 전문 용어 (강의용)

**Spatial Context(공간 맥락)**  
한 장의 프레임 안에서 보이는 정적 정보입니다. 사람 모양, 활의 형태, 배경 등 "지금 이 순간 화면에 무엇이 있는가"에 해당합니다. 2D CNN·ViT가 잘 추출하는 정보입니다.

**Temporal Context(시간 맥락)**  
프레임과 프레임 사이의 변화, 즉 동적 정보입니다. 팔을 당기는 순서, 화살이 날아가는 타이밍처럼 "어떻게 변하는가"를 담습니다. 비디오 AI가 사진 AI와 다른 이유가 바로 이 정보를 쓰기 때문입니다.

**Frame Difference(프레임 차분)**  
연속 두 프레임의 픽셀 값 차이 `|F(t+1) − F(t)|`입니다. 차이가 큰 영역은 움직임이 있다는 뜻입니다. Optical Flow보다 단순하지만, Motion Energy·이상 탐지의 1차 신호로 널리 씁니다.

**Motion Energy(모션 에너지)**  
프레임 차분(또는 Optical Flow 크기)을 하나의 숫자로 요약한 "그 순간 얼마나 움직였는가" 지표입니다. 시간축 그래프로 그리면 동작 peak·정지 trough를 볼 수 있어, 이후 주기 분석·이상 탐지의 기초가 됩니다.

**CNN(Convolutional Neural Network, 합성곱 신경망)**  
2D 이미지에서 가장자리·질감·형태 같은 **공간 특징**을 계층적으로 추출하는 신경망입니다. 프레임 한 장의 Spatial 정보에는 강하지만, 시간 축은 별도 설계(3D Conv, RNN, Transformer 등)가 필요합니다.

**ViT(Vision Transformer, 비전 트랜스포머)**  
이미지를 16×16 같은 **패치(patch)** 로 잘라 각 패치를 token으로 Transformer에 넣는 모델입니다. CNN은 지역적 convolution, ViT는 **Self-Attention**으로 패치 간 관계를 학습합니다. ImageNet 이후 비전 분야의 대표 아키텍처 중 하나입니다.

### 개념 비유

| 관점 | 질문 | 양궁 영상 예시 |
|------|------|---------------|
| Spatial | "지금 화면에 **무엇**이 있나?" | 활, 과녁, 양궁 선수 |
| Temporal | "**어떻게** 변하나?" | 시위 당기기 → 발사 → 화살 비행 |

### 소스코드 핵심

- `visualize_spatiotemporal_overview()` — 8프레임 격자 + 차분 맵
- `compute_motion_energy_timeline()` — Frame Diff Mean + Flow Magnitude Mean
- `plot_motion_energy_timeline()` — 시간축 그래프

---

## 2. 광학 흐름 (Optical Flow)

### 이 섹션에서 배우는 것

- Optical Flow로 픽셀 이동 벡터를 추정한다.
- **Dense(Farneback)** 와 **Sparse(Lucas-Kanade)** 방식의 차이를 이해한다.

### Vision·Video AI 전문 용어 (강의용)

**Optical Flow(광학 흐름, 옵티컬 플로우)**  
연속 프레임에서 각 픽셀(또는 특징점)이 **어디로, 얼마나** 이동했는지 (dx, dy) 벡터로 추정하는 기법 family입니다. 밝기 패턴이 프레임 간 어떻게 "흘러가는지"를 가정해 움직임을 계산합니다. 규칙 기반이라 학습 데이터 없이도 쓸 수 있어, 설비 모니터링·행동 분석 전처리에 자주 등장합니다.

**Farneback(파르너백) — Dense Optical Flow**  
**모든 픽셀**에 대해 이동량을 추정하는 Dense 방식입니다. 다중 해상도 피라미드를 쓰며, 주변 15×15 영역의 밝기 변화를 보고 (dx, dy)를 구합니다. HSV 컬러맵(방향=색, 크기=밝기)으로 시각화하면 전체 움직임 패턴을 한눈에 볼 수 있습니다. 연산량은 크지만 정보가 풍부합니다.

**Lucas-Kanade(루카스-카나데) — Sparse Optical Flow**  
**코너 특징점**만 골라 빠르게 추적하는 Sparse 방식입니다. Shi-Tomasi로 코너를 찾고 `calcOpticalFlowPyrLK`(피라미드 LK)로 다음 프레임에서 같은 점을 따라갑니다. 화살표로 방향·크기를 그려 직관적이며, 실시간 추적에 적합합니다.

**Dense vs Sparse**  
Dense는 모든 픽셀, Sparse는 일부 점만 다룹니다. Dense는 "전체 장면이 어떻게 움직이는가", Sparse는 "이 코너들이 어디로 갔는가"에 초점을 둡니다.

**HSV 시각화 (Optical Flow용)**  
Flow 벡터의 **방향(angle)** 을 Hue(색상), **크기(magnitude)** 를 Value(밝기)에 매핑합니다. `cartToPolar`로 (dx, dy) → (magnitude, angle) 변환 후 HSV 이미지를 만듭니다.

**Shi-Tomasi 코너 검출**  
`goodFeaturesToTrack`의 기반 알고리즘입니다. 밝기가 사방으로 크게 변하는 "코너" 점을 찾아, LK 추적의 시작점으로 씁니다.

**FlowNet / PWC-Net / RAFT (딥러닝 Optical Flow)**  
Classical(Farneback, LK) 이후 딥러닝으로 end-to-end flow를 예측하는 모델들입니다. RAFT는 현재 SOTA급으로, occlusion·큰 움직임에서 Classical보다 정확하지만 GPU·학습 데이터가 필요합니다.

### Farneback vs Lucas-Kanade

| | Farneback | Lucas-Kanade |
|---|-----------|--------------|
| 유형 | Dense | Sparse |
| 대상 | 모든 픽셀 | 코너 ~200개 |
| 시각화 | HSV 컬러맵 | 녹색 화살표 |
| 장점 | 전체 패턴 | 빠르고 직관적 |

### 소스코드 핵심

- `compute_dense_optical_flow()` — `calcOpticalFlowFarneback`
- `flow_to_hsv_image()` — angle→Hue, magnitude→Value
- `compute_sparse_optical_flow_lk()` — goodFeaturesToTrack + PyrLK
- `visualize_optical_flow_comparison()` — 2×2 비교 그림

---

## 3. SlowFast Network — Dual-pathway

### 이 섹션에서 배우는 것

- SlowFast가 비디오를 **두 갈래 경로**로 나눠 처리하는 이유를 이해한다.

### Vision·Video AI 전문 용어 (강의용)

**SlowFast(슬로우패스트)**  
Facebook AI Research(FAIR)가 제안한 **이중 경로(dual-pathway) 3D CNN**입니다. 같은 클립을 Slow·Fast 두 갈래로 나눠 처리합니다. Slow는 적은 프레임·고해상도로 "무엇인지(객체·형태)", Fast는 많은 프레임·저해상도로 "어떻게 움직이는지"를 담당합니다. 인간 망막의 P-cell(고해상·저속) / M-cell(저해상·고속)에서 영감을 받았다고 알려져 있습니다.

**Dual-pathway(이중 경로)**  
하나의 입력을 서로 다른 시간·공간 해상도로 두 경로에 동시에 넣는 설계입니다. 단일 3D CNN이 모든 프레임을 고해상도로 처리하면 연산량이 폭증하는데, SlowFast는 **효율과 정확도의 균형**을 노립니다.

**Slow Pathway**  
프레임 수 적음(예: 8), **원본 해상도** 유지. Spatial(형태·세부) 정보를 담습니다.

**Fast Pathway**  
프레임 수 많음(α=4배 → 32), **112×112 저해상도**. Temporal(빠른 동작) 정보를 담습니다.

**Alpha(α)**  
Fast 경로의 프레임 밀도 배율입니다. α=4이면 Slow 8프레임 대비 Fast 32프레임을 사용합니다.

### Slow vs Fast

| | Slow | Fast |
|---|------|------|
| 프레임 | 8 (적음) | 32 (α=4배) |
| 해상도 | 원본 | 112×112 |
| 역할 | Spatial — 형태 | Temporal — 동작 |

### 소스코드 핵심

- `build_slowfast_pathways()` — Slow 균등 8 + Fast 32 (resize 112)
- `visualize_slowfast_pathways()` — 두 경로 모자이크
- `print_architecture_comparison_table()` — C3D~Swin3D 비교표

---

## 4. 3D CNN Architectures

### 이 섹션에서 배우는 것

- **3D CNN**으로 Action Recognition(행동 인식)을 수행한다.
- R3D, MC3, Swin3D 추론 결과를 비교한다.

### Vision·Video AI 전문 용어 (강의용)

**3D CNN(3차원 합성곱 신경망)**  
Conv3D 커널이 **(시간 × 높이 × 너비)** 3축을 동시에 훑습니다. 2D CNN이 한 프레임의 공간만 본다면, 3D CNN은 **여러 프레임에 걸친 시공간 패턴**(손 흔들기, 달리기 등)을 직접 학습합니다. 비디오 행동 인식의 핵심 아키텍처 family입니다.

**C3D(2014)**  
3D Convolution을 비디오에 처음 본격 적용한 대표 모델입니다. 시간·공간을 같은 convolution으로 처리해 "작은 시공간 큐브" 단위 특징을 쌓습니다.

**R3D-18**  
ResNet-18 구조를 3D Conv로 확장한 모델입니다. torchvision에서 C3D 계열 대표로 제공되며, Kinetics-400으로 사전학습된 weight를 씁니다.

**I3D(Inflated 3D)**  
ImageNet에서 학습된 **2D CNN 가중치**를 시간 축 방향으로 복제·확장(Inflation)해 3D 필터로 만드는 전략입니다. 이미지에서 배운 특징 추출 능력을 비디오에 **전이(transfer)** 합니다.

**Inflation(인플레이션)**  
2D 필터 (H×W)를 (T×H×W) 3D 필터로 "팽창"시키는 기법입니다. T=1이면 2D와 동일, T>1이면 시간 방향으로 같은 패턴을 반복 적용한 것과 같습니다.

**MC3-18(Mixed Convolution 3D)**  
I3D 계열로, 채널별로 다른 Conv3D 구조를 섞어 효율을 높인 모델입니다.

**Swin3D(Video Swin Transformer)**  
2D Swin의 **3D Window Attention**을 비디오에 적용한 Transformer 계열 모델입니다. 작은 시공간 윈도우 안에서 Self-Attention을 계산해, 전역 attention보다 연산 효율이 좋습니다. "3D CNN 섹션"과 "Video Transformer 섹션"의 경계에 있는 모델입니다.

**Self-Attention(자기 주의)**  
입력 token(패치)들끼리 "서로 얼마나 관련 있는지" 가중치를 계산하는 메커니즘입니다. Transformer의 핵심으로, 먼 프레임·먼 공간 위치 간 관계도 직접 연결할 수 있습니다.

**Action Recognition(행동 인식)**  
비디오 클립에서 "걷기", "양궁", "춤" 등 **행동 클래스**를 분류하는 과제입니다. 본 실습은 archery 클립에 대해 Top-k 클래스를 출력합니다.

### 3D CNN vs 2D CNN

```
2D Conv:  [높이 × 너비]         → 한 프레임 공간만
3D Conv:  [시간 × 높이 × 너비]   → 여러 프레임 시공간 패턴
```

### 모델 비교

| Model | 교재 대응 | 입력 |
|-------|-----------|------|
| R3D-18 | C3D | 16프 × 112² |
| MC3-18 | I3D | 16프 × 112² |
| Swin3D-T | Video Swin | 32프 × 224² |

### 소스코드 핵심

- `_prepare_torchvision_clip()` — RGB → (C,T,H,W) tensor
- `predict_torchvision_video_model()` — softmax top-k
- `run_3d_cnn_benchmark()` — R3D, MC3, Swin3D 순차 추론

---

## 5. Video Transformer

### 이 섹션에서 배우는 것

- CNN 대신 **Transformer Attention**으로 비디오를 이해한다.
- TimeSformer와 ViViT의 Space-Time Attention 차이를 이해한다.

### Vision·Video AI 전문 용어 (강의용)

**Transformer**  
"Attention is All You Need" 논문의 Self-Attention 기반 신경망입니다. NLP에서 출발했으나, ViT 이후 **Vision Transformer**로 이미지·비디오에 확장되었습니다. CNN의 inductive bias(지역성) 대신, 데이터와 attention으로 패턴을 학습합니다.

**TimeSformer(타임스포머)**  
Facebook이 제안한 Video Transformer입니다. **Divided Space-Time Attention**을 씁니다. 모든 시공간 패치에 한꺼번에 attention(Joint)을 걸면 연산량이 폭발하므로, **Spatial Attention → Temporal Attention**을 순차 분리해 Joint 대비 약 10배 효율을 얻습니다. 입력 8프 × 224², Kinetics-400 fine-tune 모델이 Hugging Face에 공개되어 있습니다.

**ViViT(Video Vision Transformer)**  
Google이 제안한 Video Transformer입니다. 연속 2프레임 × 16×16 패치를 묶은 **Tubelet**을 3D token으로 쓰고, **Factorized Attention**으로 시간·공간 축을 분리 처리합니다. 입력 32프 × 224²로 TimeSformer보다 긴 시간 맥락을 다룹니다.

**Tubelet(튜블릿)**  
ViViT의 3D token 단위입니다. "2프레임 × 16×16 패치"를 하나의 작은 **시공간 덩어리**로 묶어, 프레임 간 움직임이 token 안에 포함되도록 합니다.

**Token / Patch**  
ViT·ViViT에서 Transformer 입력의 최소 단위입니다. 이미지는 2D patch, ViViT는 3D tubelet이 token이 됩니다.

**Attention(어텐션)**  
Query-Key-Value 구조로 "어떤 token에 집중할지" 가중치를 학습합니다. 비디오에서는 "발사 순간 프레임"에 높은 attention이 가는 식으로 **중요 시공간 위치**를 자동 선택합니다.

**Space-Time Attention 3가지**

| 방식 | 설명 | 대표 |
|------|------|------|
| Joint (ST) | 모든 시공간 패치 동시 attention | 연산 최대 |
| Divided (T+S) | Spatial → Temporal 순차 | TimeSformer |
| Factorized | 시간·공간 독립 encoder | ViViT |

### TimeSformer vs ViViT

| | TimeSformer | ViViT |
|---|-------------|-------|
| Attention | Divided (T+S) | Factorized + Tubelet |
| 입력 | 8프 × 224² | 32프 × 224² |

### 소스코드 핵심

- `predict_timesformer()` — 8프, `facebook/timesformer-base-finetuned-k400`
- `predict_vivit()` — 32프, tubelet [2,16,16]
- `print_space_time_attention_guide()` — Joint/Divided/Factorized 설명

---

## 6. 모델 비교 요약

### Vision 기술 진화

```
Optical Flow (규칙 기반 픽셀 변위)
        ↓
3D CNN — C3D → I3D → SlowFast
        ↓
Video Transformer — TimeSformer / ViViT / Swin3D
```

| 단계 | Temporal 정보 추출 |
|------|-------------------|
| Classical | Farneback, LK — 픽셀/코너 이동 벡터 |
| 3D CNN | 3D Conv / 3D Window Attention |
| Transformer | Divided / Factorized Self-Attention |

### 소스코드

- `print_model_comparison_summary()` — 모든 모델 Top-1 한 표

---

## 부록: 출력 파일

| 폴더 | 파일 |
|------|------|
| `output/spatiotemporal/` | spatial_vs_temporal.png, motion_energy_timeline.png |
| `output/optical_flow/` | flow_compare_0010.png |
| `output/slowfast/` | slowfast_pathways.png |

---

## FAQ

**Q. Optical Flow와 3D CNN의 차이?**  
A. Optical Flow는 **규칙 기반**으로 "픽셀이 어디로 갔는지" 계산합니다. 3D CNN·Transformer는 **데이터로부터** "어떤 행동인지"를 학습합니다.

**Q. TimeSformer vs ViViT?**  
A. Attention **분리 방식**이 다릅니다. TimeSformer는 공간→시간 순차(Divided), ViViT는 Tubelet 3D token + Factorized. 입력 프레임 수도 8 vs 32입니다.

**Q. Swin3D는 CNN인가 Transformer인가?**  
A. **Transformer 계열**입니다. 3D Window Attention을 사용합니다.

---

*본 문서는 `01_Temporal_Vision.ipynb` 소스문서로 작성되었습니다.*
