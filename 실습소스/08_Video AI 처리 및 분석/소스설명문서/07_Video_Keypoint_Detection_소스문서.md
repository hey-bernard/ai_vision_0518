# 07 Video Keypoint Detection — 소스문서

> **대상:** Video AI 초보자  
> **원본:** `07_Video_Keypoint_Detection.ipynb`  
> **실습 데이터:** IPAD_sample.zip + 포즈 샘플

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | 키포인트 개념 | Feature vs Pose |
| 2 | Harris · ORB · RANSAC | 특징점 검출·매칭 |
| 3 | Keypoint R-CNN | COCO 17 관절 포즈 |
| 4 | LK 추적 · manifest | displacement z-score |
| 5 | 포즈 시간 안정성 | 관절 각도 변화 |

---

## Part 1 — 키포인트 개념

### Vision·Video AI 전문 용어 (강의용)

**Keypoint(키포인트, 특징점)**  
이미지·프레임에서 **의미 있는 (x,y) 좌표**입니다. 두 종류가 본 실습에 등장합니다.

**Feature Keypoint(특징점 / 코너)**  
코너·텍스처 돌출부 등 **반복 추적에 적합한 점**입니다. Harris, Shi-Tomasi, ORB로 검출. **설비 부품·고정 구조** 추적, Optical Flow sparse 버전과 연결됩니다.

**Pose Keypoint(포즈 키포인트 / 관절)**  
사람 **신체 관절** (어깨, 팔꿈치, 무릎 등). Keypoint R-CNN으로 COCO 17점 검출. 작업자 자세·안전 분석에 씁니다.

**Descriptor(디스크립터)**  
키포인트 **주변 패치의 지문(fingerprint)** 벡터입니다. ORB·SIFT 등이 제공. 다음 프레임에서 **같은 점인지** 매칭할 때 descriptor 거리로 판단합니다.

**Tracking(추적)**  
연속 프레임에서 **동일 physical point**에 ID·좌표를 유지하는 것. Lucas-Kanade optical flow 또는 ORB re-detection으로 구현합니다.

---

## Part 2 — Harris · ORB · RANSAC

### Vision·Video AI 전문 용어 (강의용)

**Harris Corner Detector(해리스 코너 검출)**  
윈도우를 조금 움직였을 때 **밝기 변화가 큰** 점(코너)을 찾습니다. `cornerHarris` heatmap — flat·edge보다 corner가 cornerness score가 큽니다.

**Shi-Tomasi Corner**  
Harris 개선. `goodFeaturesToTrack` — **추적에 안정적인** 상위 코너 N개 선택. LK optical flow 시작점으로 널리 씁니다.

**ORB(Oriented FAST and Rotated BRIEF)**  
FAST로 keypoint, **회전 불변 BRIEF** descriptor. **Binary** descriptor라 매칭이 빠르고, 회전·스케일 변화에 Harris보다 강합니다. 실시간 SLAM·추적에 많이 씁니다.

**Lowe Ratio Test(로우 ratio test)**  
BFMatcher knn(k=2)에서 **최근접/차근접 거리 ratio**가 threshold(예: 0.75) 이하인 매칭만 채택. **ambiguous match(outlier)** 를 걸러 descriptor 매칭 품질을 올립니다.

**RANSAC(Random Sample Consensus, 랜섹)**  
매칭 쌍 중 **outlier**가 많을 때, random subset으로 모델(여기선 **Homography**)을 반복 추정해 **inlier**만 남깁니다. 두 프레임이 **같은 평면·같은 장면**인지, inlier ratio 90%↑면 거의 동일 시점·시점입니다.

**Homography(호모그래피, 투영 변환)**  
평면 A → 평면 B를 나타내는 3×3 행렬 H. 카메라 시점이 바뀌었지만 **같은 평면**(바닥, 벽) 위 점들의 대응 관계를 설명합니다. `findHomography` + RANSAC.

---

## Part 3 — Keypoint R-CNN (인체 포즈)

### Vision·Video AI 전문 용어 (강의용)

**Keypoint R-CNN**  
torchvision의 **2-stage** 모델: Faster R-CNN으로 person bbox 검출 → RoIAlign → **관절 heatmap/keypoint** 예측. COCO keypoints로 pretrained (`keypointrcnn_resnet50_fpn`).

**COCO Keypoints(17점)**  
nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles 등. 각 점에 **visibility** (0=없음, 1=가려짐, 2=보임).

**Skeleton(스켈레톤)**  
관절을 **뼈대 선**으로 연결한 표현. 자세 시각화·각도 계산의 기반.

**Joint Angle(관절 각도)**  
팔꿈치·무릎·어깨 등 세 점으로 이루어진 **각도(도)**. 인체공학·작업 자세 이상(과도한 굽힘) 탐지에 씁니다.

---

## Part 4 — LK 추적 · manifest

### Vision·Video AI 전문 용어 (강의용)

**Lucas-Kanade Optical Flow(루카스-카나데)**  
`calcOpticalFlowPyrLK` — Part 1에서 고른 feature point를 **다음 프레임에서 추적**. 국소 brightness constancy 가정 하에 (dx,dy) 추정.

**Adaptive Tracking(적응형 추적)**  
추적 점이 부족해지면 ORB로 **재검출**해 track pool을 보충. 긴 IPAD 클립에서 drift 방지.

**Displacement(변위)**  
프레임마다 추적 점들의 **평균 이동 거리(px)**. 02 Motion Energy와 유사한 "움직임 크기" 신호이나, **특징점 subset** 기준입니다.

**Motion Z-score**  
02·04와 동일. displacement 시계열 baseline 대비 편차. `motion_spike_frames` — z≥2.0.

**Density Heatmap(밀도 히트맵)**  
키포인트 누적 Gaussian — **자주 지나가는 vs 고정** 영역. 설비 고정부·반복 궤적 시각화.

---

## Part 5 — 포즈 시간 안정성

- `analyze_pose_temporal_stability()` — shoulder center displacement, angle delta
- IPAD에 인물 없으면 synthetic pose sequence fallback

---

## 시리즈 연계

02 Motion · 04 z-score · **08 unified manifest** (`keypoint_manifest.json`)

---

*본 문서는 `07_Video_Keypoint_Detection.ipynb` 소스문서로 작성되었습니다.*
