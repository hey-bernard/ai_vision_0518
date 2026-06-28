# 08 Integrated Manifest — 소스문서

> **대상:** Video AI 초보자 (03~07 수강 후)  
> **원본:** `08_Integrated_Manifest.ipynb`  
> **실습 데이터:** IPAD + 03~07 manifest JSON

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | manifest 스캔 | 소스 존재 · IPAD 타임라인 |
| 2 | 신호 추출 | normalize → FrameRecord |
| 3 | 조인 · 스코어 | unified_score · 알람 후보 |
| 4~7 | export · Query · 분석 | JSON/CSV · 모자이크 |

---

## 통합 Manifest 개요

03~07 각 파이프라인이 만든 JSON을 **프레임 단위 frame_key**로 조인해 `unified_manifest.json` 하나로 정리합니다. 운영 대시보드·알람은 **한 타임라인**에서 sampling·이상·OCR·키포인트·VLM 신호를 같이 봐야 합니다.

| 노트북 | manifest | 주요 신호 |
|--------|----------|-----------|
| 03 | sampling_manifest.json | swin/detr 선별 인덱스 |
| 05 | summary_manifest.json | vlm_keyframe, text_summary |
| 04 | anomaly_manifest.json | anomaly_candidate, top_frame |
| 06 | ocr_manifest_ipad.json | full_text, detections |
| 07 | keypoint_manifest.json | displacement, motion_z |

---

## Part 1 — frame_key · 스키마

### Vision·Video AI / MLOps 용어 (강의용)

**Manifest(매니페스트)**  
Video AI 파이프라인 **중간 산출물 JSON**입니다. 원본 영상 대신 **frame_index, score, bbox, text** 같은 메타데이터·신호를 담습니다. 모듈 간 **데이터 contract** 역할을 해 재현·병합·알람 연동이 가능합니다.

**Frame Key(프레임 키)**  
서로 다른 manifest가 같은 프레임을 가리키도록 통일한 **조인 키**입니다. `C:\...\frames\01\042.jpg` → `frames/01/042.jpg`. sampling은 index만, OCR은 path만 줄 때 IPAD 타임라인으로 **index↔path** 연결합니다.

**Schema Validation(스키마 검증)**  
소스별 필수 JSON 키(`presets`, `vlm_frames`, `top_frame_indices` 등) 존재·타입 확인. 파일은 있는데 구조가 바뀐 **silent failure**를 사전에 경고합니다.

**Coverage Report(커버리지 리포트)**  
어떤 manifest가 `[ OK ]` / `[MISS]` 인지, IPAD 프레임 수, schema 경고를 기록한 `manifest_coverage.json`.

---

## Part 2 — Signal Extraction

### Vision·Video AI 용어 (강의용)

**Signal Normalization(신호 정규화)**  
형식이 제각각인 manifest에서 **프레임 단위 공통 필드**만 추출합니다. `extract_sampling_signals` → `sampling_swin=True`, `extract_anomaly_signals` → `anomaly_candidate=True` 등.

**FrameRecord**  
frame_key 하나에 붙는 **통합 레코드**: signals dict + sources list. DB의 row와 유사.

**Clip Meta(_clip_meta)**  
프레임이 아닌 **클립 전체** 정보 — VLM text_summary, anomaly_scenes, motion_spike_frames.

---

## Part 3 — unified_score

### Vision·Video AI / Fusion 용어 (강의용)

**Multi-pipeline Fusion(다중 파이프라인 융합)**  
anomaly(0.45) + vlm_keyframe(0.20) + motion_z(0.20) + sampling(0.10) + ocr(0.05) 등 **가중 합산** → `unified_score` 0~1. 04번 4신호 융합의 **운영 레이어** 버전입니다.

**Signal Agreement(신호 합의도)**  
5개 독립 파이프라인(anomaly, vlm, sampling, motion_z, ocr) 중 몇 개가 "이 프레임 주목"에 동의하는지 0~1. agreement 높은 프레임 = **다중 AI 모듈이 동시에 주목** → 알람 신뢰도↑.

**Alert Candidate(알람 후보)**  
`unified_score ≥ 0.55` 또는 상위 K=8개. threshold·가중치는 현장 도메인에 맞게 조정.

**OCR Source Priority**  
같은 프레임에 `ocr_ipad`(실제 설비 프레임) vs `ocr_hub`(AI Hub 샘플) → **ipad 우선**.

---

## Part 4~7 — Export · Query · 분석

### 용어 (강의용)

**ManifestQuery API**  
`unified_manifest.json`에 SQL-like **조건 검색** 계층. `by_score`, `by_source`, `with_ocr_keyword`, `top_alerts`, `summary`. 웹 대시보드·리포트 재사용.

**Co-occurrence / Signal Correlation**  
소스 쌍이 **같은 프레임**에 동시 기록된 횟수, anomaly↔motion_z 등 **이진 신호 Pearson 상관**. "여러 모듈이 같은 구간을 가리키는가" 분석.

**Alert Mosaic(알람 모자이크)**  
알람 후보 프레임 썸네일 그리드 — 숫자·표만으로는 부족한 **육안 1차 검증**.

### 출력

- `unified_manifest.json`, `unified_frames.csv`
- `03_unified_timeline.png`, `06_alert_mosaic.jpg`
- `signal_cooccurrence.json`

---

## FAQ

**Q. manifest MISSING?** → 해당 번호 노트북(03~07) 먼저 실행.  
**Q. unified_score?** → `SCORE_WEIGHTS`, `ALERT_SCORE_THRESHOLD` 조정.

---

*본 문서는 `08_Integrated_Manifest.ipynb` 소스문서로 작성되었습니다.*
