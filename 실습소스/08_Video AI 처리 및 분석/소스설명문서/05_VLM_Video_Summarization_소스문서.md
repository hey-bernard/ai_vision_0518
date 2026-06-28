# 05 VLM Video Summarization — 소스문서

> **대상:** Video AI 초보자  
> **원본:** `05_VLM_Video_Summarization.ipynb`  
> **실습 데이터:** IPAD_sample.zip (R01, 클립 01)

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | 정적 vs 동적 요약 | Keyframe · Video Skimming |
| 2 | VLM 융합 | CLIP · Spatiotemporal Projector |
| 3 | 적응형 키프레임 | AKS · T* · K-Medoids |
| 4 | 통합 파이프라인 | summary_manifest.json |

---

## Part 1 — 정적 vs 동적 요약

### Vision·Video AI 전문 용어 (강의용)

**Video Summarization(비디오 요약)**  
긴 영상을 짧은 **대표 프레임(정적)** 또는 **중요 구간 클립(동적)** 으로 압축하는 과제입니다. VLM·관제·검색 전에 "무엇을 모델에 넣을 것인가"를 결정하는 전처리 단계입니다.

**Keyframe(키프레임)**  
영상 전체를 대표하는 **핵심 정지 화면** 몇 장입니다. 정적 요약(static summary)의 출력입니다. histogram 다양성, SSIM/MSE 차이, Motion peak 등으로 선별합니다.

**Video Skimming(비디오 스키밍)**  
중요 **구간(clip)** 을 이어 붙인 동적 요약입니다. Shot boundary로 나눈 뒤 motion importance로 상위 구간(예: 35%)만 선택합니다.

**Shot Boundary(샷 경계)**  
카메라 컷·장면 전환 지점입니다. histogram Bhattacharyya 거리가 threshold를 넘으면 새 shot으로 분할합니다.

**SSIM(Structural Similarity, 구조적 유사도)**  
두 프레임의 **구조(밝기·대비·구조)** 유사도 0~1입니다. MSE(픽셀 차이 제곱)보다 사람 눈에 가깝게 "비슷한 장면"을 판단해, **서로 다른 키프레임**을 greedy하게 고를 때 씁니다.

---

## Part 2 — VLM 융합 & 토큰화

### Vision·Video AI 전문 용어 (강의용)

**Vision-Language Model, VLM(비전-언어 모델)**  
이미지·비디오와 **자연어**를 함께 입력·출력하는 모델입니다. "이 영상에 무슨 일이 일어나는가?"에 답하거나 캡션·요약을 생성합니다. Video-LLaVA, LLaVA-Next-Video, Video-LLaMA 등이 대표입니다.

**CLIP(Contrastive Language-Image Pre-training)**  
04번과 동일. 본 Part에서는 프레임별 **relevance score**(텍스트 query와의 cosine similarity)로 "요약에 넣을 가치가 있는 프레임"을 ranking합니다.

**ViT(Vision Transformer)**  
CLIP·LLaVA의 visual encoder로 쓰이는 패치 기반 Transformer입니다. 프레임 → patch token → visual feature sequence.

**Spatiotemporal Projector(시공간 프로젝터)**  
N개 프레임의 visual feature `(B,T,D)`를 VLM LLM이 받을 수 있는 **K개 token** `(B,K,D)`으로 압축하는 모듈입니다. 비디오는 프레임 수가 많아 token 폭발을 막기 위해 **Cross-Attention pooling** 등으로 차원을 줄입니다. Video-LLaVA·LLaVA-Next-Video의 핵심 구성 요소입니다.

**Q-Former( Query Transformer)**  
BLIP-2에서 도입된 learnable query token이 visual feature에 **cross-attention**하는 모듈입니다. visual 정보를 고정 개수 query로 압축해 LLM에 넘깁니다.

**Token Budget(토큰 예산)**  
LLM context window 한도 때문에 VLM에 넣을 **최대 프레임/token 수**입니다. 본 실습 `NUM_VLM_FRAMES=8` — 긴 영상은 키프레임 선별(Part 3)이 필수입니다.

**LLM Token Alignment(LLM 토큰 정렬)**  
Visual projector 출력을 LLM word embedding space에 맞춰, "이미지 token + 텍스트 prompt"를 하나의 sequence로 LLM에 입력합니다.

### VLM 아키텍처 비교

| Model | Token 전략 |
|-------|-----------|
| Video-LLaVA | Uniform sample + projection |
| Video-LLaMA | Visual + Audio 병렬 |
| LLaVA-Next-Video | AnyRes + spatiotemporal pooling |

---

## Part 3 — 적응형 키프레임

### Vision·Video AI 전문 용어 (강의용)

**Adaptive Keyframe Sampling, AKS(적응형 키프레임 샘플링)**  
CVPR 2025 등에서 제안된 방식으로, CLIP **relevance**가 높으면서 동시에 시간축 **coverage(고르게 분포)** 가 좋은 프레임 subset을 고릅니다. relevance만 maximization하면 키프레임이 한 구간에 몰릴 수 있어 coverage 제약을 함께 둡니다.

**T*(Temporal Search)**  
relevance 배열 위에서 **coarse grid → promising cell zoom-in**으로 최적 프레임 subset을 탐색합니다. brute-force보다 긴 영상에서 효율적 키프레임 탐색(교재 CVPR 2025).

**K-Medoids(케이-메도이드)**  
change-point로 나눈 **구간마다** histogram 거리 기준 **medoid(실제 존재하는 프레임)** 를 대표로 선택합니다. K-Means의 centroid가 "가상 평균"인데 반해, medoid는 **실제 프레임**이라 VLM 입력으로 바로 쓸 수 있습니다.

**Change-Point(변화점)**  
histogram·motion Z 급변 지점. 영상을 의미 구간으로 나눈 뒤 구간별 medoid를 뽑습니다.

**Long-form VLM(장영상 VLM)**  
수십 분~수 시간 영상을 다루는 VLM입니다. 전체를 넣을 수 없어 **키프레임 선별·압축·projector** 전처리 파이프라인이 필수입니다.

---

## Part 4 — 통합 VLM 요약 파이프라인

### Vision·Video AI 전문 용어 (강의용)

**Rule-based Summary(규칙 기반 요약)**  
LLM API 없이 segment·motion·relevance 통계로 narrative를 생성합니다. PoC·오프라인 환경용. 실무에서는 Video-LLaVA API로 대체합니다.

**BLIP(Bootstrapping Language-Image Pre-training)**  
이미지 **캡션 생성** 모델입니다. `RUN_BLIP=True` 시 키프레임마다 짧은 영문 캡션을 붙여 요약 품질을 보강합니다.

**VLM Input Card( VLM 입력 카드)**  
선별된 키프레임 + 메타(relevance, segment, query)를 한 장 그림으로 시각화한 것입니다. "VLM에 실제로 무엇이 들어가는가"를 강의·디버깅할 때 씁니다.

### 파이프라인

```
IPAD → Motion/Shot/Change-point → CLIP relevance
     → AKS 키프레임 → Projector(K tokens)
     → Rule-based 요약 (+ BLIP 선택)
     → summary_manifest.json
```

---

*본 문서는 `05_VLM_Video_Summarization.ipynb` 소스문서로 작성되었습니다.*
