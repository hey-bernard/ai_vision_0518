# 03 Frame Sampling & Swin/DETR — 소스문서

> **대상:** Video AI 초보자 (02 IPAD 수강 후)  
> **원본:** `03_Frame_Sampling_Swin_DETR.ipynb`  
> **실습 데이터:** `IPAD_sample.zip` (R01, 클립 01)

---

## 강의 전체 흐름

| 섹션 | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1.5 | 디코딩 병목 | OpenCV vs Decord 배치 로딩 |
| 2 | 프레임 샘플링 | Uniform / Motion / Cycle-aware 6종 |
| 3 | Swin Transformer | Cycle-aware 프레임 Spatial 분류 |
| 4 | DETR | Motion top-k 객체 검출 |
| 5 | 통합 파이프라인 | sampling_manifest.json |

---

## 1.5 디코딩 병목

### Vision·Video AI 전문 용어 (강의용)

**Frame Sampling(프레임 샘플링)**  
긴 영상(수천 프레임)에서 모델이 처리할 **일부 프레임만** 고르는 기법입니다. Video Transformer·VLM은 입력 token 수에 한계가 있어, "어떤 프레임을 넣을 것인가"가 성능·비용의 핵심입니다. Uniform(균등), Motion top-k(움직임 큰 구간), Cycle-aware(주기 기반) 등 전략이 있습니다.

**Decord(디코드)**  
딥러닝·대규모 학습용 **고속 비디오 디코더**입니다. OpenCV가 순차 읽기·seek에 약한 반면, Decord는 **임의 인덱스 배치(`get_batch`)** 를 GPU(NVDEC)로 빠르게 디코딩합니다. 교재 기준 디코딩 병목 해소 시 throughput 최대 약 7배 향상 사례가 있습니다.

**Inference Gap(추론 갭)**  
GPU에서 모델 inference는 빠른데, CPU에서 프레임 디코딩·전처리가 느려 **GPU가 데이터를 기다리는** 현상입니다. 샘플링 + 배치 디코딩으로 이 gap을 줄입니다.

### 병목 대응

| 병목 | 대응 |
|------|------|
| CPU 순차 디코딩 | 배치 인덱스 로딩 |
| I/O 대용량 | 필요한 인덱스만 읽기 |
| Inference Gap | Decord get_batch + 샘플링 |

### 소스코드

- `benchmark_opencv_*()`, `benchmark_decord_batch()`
- `IpadBatchReader.get_batch()` — (N,H,W,C) RGB

---

## 2. 프레임 샘플링 6종

### Vision·Video AI 전문 용어 (강의용)

**Uniform Sampling(균등 샘플링)**  
전체 구간을 균등 간격으로 추출합니다. 긴 영상의 **전역(global) 맥락**을 대표하는 프레임을 얻을 때 씁니다. VLM 요약·장면 스키밍의 기본 전략입니다.

**Motion top-k Sampling**  
Motion Energy가 **상위인 프레임** + 양끝 프레임을 선택합니다. "움직임이 큰 순간"에 모델 compute를 집중해, 설비 동작 peak·이상 구간을 놓치지 않으려는 전략입니다. 본 실습 DETR 입력(6프)에 사용합니다.

**Cycle-aware Sampling(주기 인식 샘플링)**  
02번 Autocorr로 추정한 주기 T를 기준으로, **사이클마다 대표 프레임**을 고릅니다. 산업 설비처럼 반복 동작이 있을 때, 매 사이클의 "같은 위상" 프레임을 모아 Swin 입력(8프)에 씁니다.

**Consecutive / Strided**  
특정 시점부터 **연속 N프**(action clip) vs **stride 간격** 추출(연산·시간 범위 균형).

### 프리셋

| 모델 | 샘플링 | 프레임 |
|------|--------|--------|
| Swin-T | Cycle-aware | 8 |
| DETR | Motion top-k | 6 |

### 소스코드

- `FrameSampler`, `SamplingStrategy` enum
- `sampling_manifest.json` — preset별 인덱스

---

## 3. Swin Transformer — Spatial 분류

### Vision·Video AI 전문 용어 (강의용)

**ViT(Vision Transformer)**  
이미지를 패치(16×16 등)로 잘라 token화하고 Transformer Self-Attention으로 처리합니다. 전역 attention은 연산량이 O(n²)이라 고해상도·대형 이미지에 부담이 큽니다.

**Swin Transformer(스윈 트랜스포머)**  
**S**hifted **Win**dow Attention 기반 ViT입니다. 이미지를 작은 **윈도우** 단위로 나눠 윈도우 **내부**에서만 attention을 계산해 ViT 대비 연산량을 크게 줄입니다. 다음 layer에서 윈도우를 **shift**해 인접 윈도우 패치 간 정보도 교환합니다. 계층적으로 윈도우를 합치며(m merge) ImageNet 분류 SOTA급 성능을 냈습니다.

**Window Attention / Shifted Window**  
Window Attention: 7×7 같은 고정 윈도우 안에서만 Q-K-V attention. Shifted Window: 윈도우 경계를 half-patch만큼 밀어 **크로스-윈도우** 연결. "지역 attention + 점진적 전역화"가 Swin의 핵심입니다.

**Feature Drift(피처 드리프트)**  
Cycle-aware 8프를 Swin에 넣었을 때, **프레임 간 embedding cosine 유사도**가 갑자기 떨어지는 현象입니다. ImageNet Top-1 라벨("screwdriver" 등) 자체보다, 산업 영상에서는 **유사도 급변 프레임**이 이상·공정 변화 단서가 됩니다.

### 추론 흐름

```
Cycle-aware 8프 → AutoImageProcessor → Swin-T → Top-k + cosine heatmap
```

### 소스코드

- `microsoft/swin-tiny-patch4-window7-224`
- `swin_classify_frames()`, `plot_swin_results()`

---

## 4. DETR — 객체 검출

### Vision·Video AI 전문 용어 (강의용)

**Object Detection(객체 검출)**  
이미지에서 "무엇이 어디에 있는가" — **바운딩 박스(bbox) + 클래스 + confidence**를 출력하는 과제입니다. YOLO는 anchor 기반 one-stage, DETR은 Transformer 기반 end-to-end입니다.

**DETR(DEtection TRansformer, 디터)**  
Facebook의 **Transformer Encoder-Decoder** 객체 검출 모델입니다. CNN backbone(ResNet)으로 feature map을 만들고, **learnable object query** N개(예: 100개)가 decoder에서 각각 "객체 하나"를 예측합니다. **NMS(Non-Maximum Suppression) 없이** end-to-end로 학습·추론합니다. 겹치는 박스는 Hungarian matching으로 학습 시 1:1 매칭됩니다.

**Object Query(객체 쿼리)**  
DETR decoder의 learnable embedding 슬롯입니다. 각 query가 "이미지 어디를 볼지" attention으로 결정하며, 하나의 detection 후보(bbox+class)를 출력합니다. query 수 = 최대 검출 객체 수 상한입니다.

**NMS(Non-Maximum Suppression)**  
전통 detector(YOLO, Faster R-CNN)에서 **겹치는 bbox** 중 confidence 낮은 것을 제거하는 후처리입니다. DETR은 set prediction + bipartite matching으로 NMS가 **구조적으로 불필요**하다는 것이 핵심 주장입니다.

**COCO pretrained 한계**  
DETR-ResNet-50은 COCO 80+ 클래스(사람, 컵, toothbrush 등)로 학습됩니다. IPAD **설비 부품**은 COCO에 없어 **오탐·엉뚱한 라벨** 가능 → 실무에서는 **도메인 fine-tune** 또는 motion blob 보조(09번)가 필요합니다.

### 추론 흐름

```
Motion top-k 6프 → DetrImageProcessor → DetrForObjectDetection → bbox 시각화
```

### 소스코드

- `facebook/detr-resnet-50`, `DETR_SCORE_THRESHOLD=0.35`
- `detect_objects()`, `draw_detr_boxes()`

---

## 5. 통합 파이프라인

- `run_spatial_pipeline()` → Swin + DETR + `sampling_manifest.json`
- 08번 `MANIFEST_SOURCES["sampling"]` 입력

---

## Swin vs DETR

| | Swin-T | DETR |
|---|--------|------|
| 과제 | Image Classification | Object Detection |
| 샘플링 | Cycle-aware 8f | Motion top-k 6f |
| 산업 신호 | feature drift | bbox (fine-tune 필요) |

---

*본 문서는 `03_Frame_Sampling_Swin_DETR.ipynb` 소스문서로 작성되었습니다.*
