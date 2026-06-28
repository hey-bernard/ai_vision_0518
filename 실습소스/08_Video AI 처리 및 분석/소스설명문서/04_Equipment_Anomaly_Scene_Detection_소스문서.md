# 04 Equipment Anomaly Scene Detection — 소스문서

> **대상:** Video AI 초보자  
> **원본:** `04_Equipment_Anomaly_Scene_Detection.ipynb`  
> **실습 데이터:** IPAD (training=정상, testing=이상 포함)

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1~2 | Motion · Appearance | 픽셀 기반 이상 신호 |
| 3 | CLIP embedding | 의미·장면 baseline 편차 |
| 4 | VLM query contrast | 정상 vs 이상 텍스트 대조 |
| 5~6 | 융합 · manifest | 4신호 fused score → anomaly_manifest |

---

## Part 1~2 — 픽셀 기반 이상 신호

### Vision·Video AI 전문 용어 (강의용)

**Video Anomaly Detection, VAD(비디오 이상 탐지)**  
정상 패턴(baseline)을 기준으로 test 영상에서 **다른 구간·프레임**을 찾습니다. 본 Part는 training(정상) 클립의 motion·appearance 분포를 baseline으로, testing 프레임별 편차를 측정합니다.

**Motion Z-score(모션 지-스코어)**  
02번과 동일 개념. training motion 분포의 μ, σ 대비 test 프레임 motion이 `(x−μ)/σ`로 얼마나 벗어났는지입니다. 설비 **동작 강도·리듬 이상**을 잡습니다.

**Appearance Change(외관 변화)**  
연속 프레임의 **색·밝기 histogram 분포**가 급변했는지 봅니다. Motion이 작아도 카메라 전환·조명 변화·장면 컷이 있으면 appearance score가 올라갑니다. Motion과 **상호 보완**합니다.

**Bhattacharyya Distance(바타차리야 거리)**  
두 **색 histogram(HSV)** 분포가 얼마나 다른지 0~1로 재는 거리입니다. 0에 가까우면 비슷한 장면, 1에 가까우면 다른 장면·조명입니다. 장면 전환(change-point) 탐지에 씁니다.

---

## Part 3 — CLIP embedding baseline

### Vision·Video AI 전문 용어 (강의용)

**CLIP(Contrastive Language-Image Pre-training, 클립)**  
OpenAI의 **이미지·텍스트 공동 embedding** 모델입니다. 수억 장의 (이미지, caption) 쌍으로 contrastive learning해, 같은 의미의 이미지·문장은 embedding 공간에서 가깝게, 다른 의미는 멀게 배치합니다. **zero-shot**으로 텍스트 prompt만으로 분류·검색이 가능해 VLM·이상 탐지·요약의 기반 모델이 되었습니다.

**Image Embedding(이미지 임베딩)**  
CLIP ViT backbone이 프레임을 **512차원( base 모델)** 벡터로 압축한 표현입니다. 픽셀 histogram과 달리 "장면·객체·스타일" 같은 **고수준 의미**를 담습니다.

**Centroid(센트로이드, 중심 벡터)**  
training(정상) 프레임 embedding들의 **평균 벡터**입니다. "정상 설비 장면"의 의미적 중심으로, test 프레임 embedding이 centroid에서 **cosine distance**로 멀수록 시각·의미적으로 다른 장면(이상 후보)입니다.

**Cosine Distance / Similarity(코사인 거리·유사도)**  
두 embedding 벡터의 **방향(각도)** 차이입니다. 크기(norm)는 무시하고 "의미 방향"만 비교해, 조명 밝기 차이에 덜 민감한 편입니다.

---

## Part 4 — VLM query contrast

### Vision·Video AI 전문 용어 (강의용)

**Vision-Language Model, VLM(비전-언어 모델)**  
이미지(또는 비디오)와 **자연어**를 함께 이해하는 모델 family입니다. CLIP은 embedding·유사도, Video-LLaVA·LLaVA는 LLM과 결합해 **자연어 생성**까지 합니다. 본 Part는 full VLM 대신 CLIP으로 **텍스트 query와 프레임의 relevance**를 계산합니다.

**Query Contrast(쿼리 대조)**  
**정상 query**("normal industrial machine operation")와 **이상 query**("malfunction abnormal vibration stuck") 각각에 대한 CLIP 유사도를 구하고, `이상 유사도 − 정상 유사도`를 점수로 씁니다. 단일 query보다 **정상 대비 이상 쪽으로 기울었는지**를 명시적으로 봅니다.

**Relevance Score(관련도 점수)**  
텍스트 prompt와 이미지 embedding의 cosine similarity입니다. "이 프레임이 이 문장과 얼마나 맞는가"를 0~1 근처 값으로 표현합니다.

---

## Part 5~6 — 세그멘테이션 · 융합 · manifest

### Vision·Video AI 전문 용어 (강의용)

**Change-Point Segmentation(변화점 세그멘테이션)**  
histogram·motion Z가 **급변하는 프레임**을 경계로 영상을 구간 `[start, end)`로 나눕니다. 프레임 단위 점수를 **장면(scene) 단위**로 묶어 알람·요약에 씁니다.

**Multi-signal Fusion(다중 신호 융합)**  
Motion 30% + Appearance 20% + CLIP embedding 35% + Query contrast 15% 가중 합산으로 **fused anomaly score**를 만듭니다. 한 신호만으로는 놓치는 이상(조명 변화 vs 동작 이상)을 **상호 보완**합니다.

**AKS(Adaptive Keyframe Sampling)**  
05번과 동일. fused score가 높은 구간에서 **대표 이상 프레임**을 고릅니다. relevance + temporal coverage를 동시에 고려하는 키프레임 선별(교재 CVPR 2025 개념).

**Anomaly Scene(이상 장면)**  
fused score가 threshold(상위 percentile) 이상인 **연속 프레임 구간**입니다. 대시보드 주황 음영·`anomaly_scenes` JSON에 기록됩니다.

### 산출물

- `anomaly_manifest.json` — top_frame_indices, anomaly_scenes, scores
- 08번 unified manifest의 `anomaly_scene` 소스

---

## 05 VLM 요약 vs 04 이상 탐지

| | 05 요약 | 04 이상 |
|---|---------|---------|
| Baseline | 전체 relevance | **정상 training centroid** |
| CLIP | summary query | **정상 vs 이상 contrast** |
| 산출 | summary_manifest | **anomaly_manifest** |

---

*본 문서는 `04_Equipment_Anomaly_Scene_Detection.ipynb` 소스문서로 작성되었습니다.*
