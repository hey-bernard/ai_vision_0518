# 09 Video Object Tracking & Equipment State — 소스문서

> **대상:** Video AI 초보자 (02 Motion · 08 Manifest 수강 후)  
> **원본:** `09_Video_Object_Tracking_Equipment_State.ipynb`  
> **실습 데이터:** IPAD_sample.zip 또는 합성 시퀀스

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | 영상 수집 | IPAD / synthetic · ROI 확인 |
| 2 | YOLOv8 | Local Detection · motion blob |
| 3 | ByteTrack | MOT · ID 유지 |
| 4 | 설비 상태 | RUNNING / IDLE / ANOMALY |
| 5~6 | 요약 · 검색 | Situation Report · ChromaDB |
| 7 | 대시보드 | MOT 지표 · tracking_manifest |

---

## PoC 파이프라인

교재 **Video AI Service PoC**: 검출 → 추적 → 상태 추론 → 요약 → 시맨틱 검색

```
프레임 시퀀스 → YOLOv8/motion 검출 → ByteTrack ID
            → ROI·velocity·motion_z → 상태 분류
            → 한국어 리포트 → ChromaDB 검색
```

---

## Part 1 — 영상 수집 · 원본 시각화

### Vision·Video AI 전문 용어 (강의용)

**Decord**  
03번과 동일. mp4·프레임 시퀀스 **고속 배치 디코딩**. 없으면 OpenCV jpg 시퀀스로 진행.

**ROI(Region of Interest, 관심 영역)**  
화면 비율(0~1)로 정의한 **operation_zone**(정상 작업), **idle_zone**(대기). `EQUIPMENT_ZONES` — 해상도가 바뀌어도 같은 **상대 위치**를 가리킵니다. Part 4 zone violation의 기준.

**Motion Energy Preview**  
02번 동일 — 인접 프레임 차분으로 **움직임 강도 곡선**. 추적·분류 전 원본·ROI·motion 리듬을 눈으로 확인합니다.

---

## Part 2 — YOLOv8 객체 검출

### Vision·Video AI 전문 용어 (강의용)

**Object Detection(객체 검출)**  
프레임에서 **bbox + class + confidence** 출력. Local Detection = PoC 1단계 "무엇이 어디에 있는가".

**YOLOv8(You Only Look Once v8)**  
Ultralytics **one-stage** real-time detector. `yolov8n.pt`(nano) — 가볍고 빠름. COCO 80 클래스 pretrained.

**COCO Class Limitation(COCO 클래스 한계)**  
YOLOv8n은 person, car, truck 등 **COCO**로 학습. IPAD **컨베이어 부품**은 클래스에 없어 검출 0건이 **정상**일 수 있습니다.

**Motion Blob Detection(모션 블롭 검출)**  
연속 프레임 **그레이 차분 + threshold + contour**로 움직이는 영역 bbox. **형태·클래스를 몰라도** 움직임만 있으면 검출. IPAD에서 **주 검출**로 사용 (`class_name=motion_part`, `source=motion`).

**Hybrid Detection(하이브리드 검출)**  
IPAD: motion only in operation_zone ROI. Synthetic: YOLO + motion, IoU로 중복 제거.

**NMS / YOLO_CONF / YOLO_IOU**  
YOLO 내부 Non-Maximum Suppression — 겹치는 bbox 제거. conf=0.25, iou=0.45는 검출 수·정밀도 튜닝 knob.

---

## Part 3 — ByteTrack MOT

### Vision·Video AI 전문 용어 (강의용)

**MOT(Multi-Object Tracking, 다중 객체 추적)**  
Detection만으로는 프레임마다 bbox가 **독립**입니다. MOT는 **track_id**를 유지해 "ID3번 부품이 구역을 벗어났다"처럼 **시간에 따른 행동**을 설명합니다.

**ByteTrack**  
2022 ECCV. **고신뢰 detection 먼저** 기존 track과 IoU 매칭, **남은 저신뢰 detection도 2차 매칭**. 잠깐 가려짐(occlusion)·score dip에도 ID 유지. Ultralytics `model.track(tracker='bytetrack.yaml')`.

**IoU Matching(IoU 매칭)**  
연속 프레임 bbox **겹침(IoU)** 이 threshold(예: 0.3) 이상이면 같은 track으로 연결. MOT association의 기본 척도.

**SimpleByteTracker(교육용)**  
ByteTrack 2단계 high/low 매칭 아이디어를 IoU tracker로 구현. IPAD motion hybrid detection 입력용.

**Occlusion(가림)**  
객체가 일시적으로 가려져 detection confidence가 떨어져도, ByteTrack은 **low-confidence box**로 association해 ID switch를 줄입니다.

**Track Series**  
`track_series[i]` = i번째 프레임의 `Track` 리스트. Part 4 velocity·zone 판별 입력.

---

## Part 4 — 설비 동작 상태

### Vision·Video AI 전문 용어 (강의용)

**Track Velocity(추적 속도)**  
bbox **중심**이 프레임마다 이동한 px/frame 거리. `VELOCITY_RUN_THRESH=3.5` 이상 → RUNNING, `≤0.8` → IDLE.

**Zone Violation(구역 위반)**  
`motion_part`·`person` track 중심이 **operation_zone** 밖 — `ZONE_VIOLATION_FRAMES=2` 연속이면 ANOMALY 후보.

**Operation State(운전 상태)**  
`RUNNING` / `IDLE` / `ANOMALY` — 규칙 기반 state machine. `FrameState` = 프레임별 관제 레코드.

**Motion Z-score**  
02·07 동일. `MOTION_Z_ANOMALY=2.2` 초과 시 동작 급변 이상.

---

## Part 5 — 상황 요약 에이전트

### Vision·Video AI 전문 용어 (강의용)

**Global Reasoning(전역 추론)**  
교재 Video-LLaVA 단계 — 영상 맥락을 **자연어**로 설명. 본 실습은 **템플릿 에이전트**로 FrameState → 한국어 `SituationReport` (GPU·API 불필요 PoC).

**Situation Report(상황 리포트)**  
"프레임 42: operation_zone 이탈, ID2 motion_part" 등 **관제 문장**. ANOMALY는 즉시, 그 외 1/6 간격 샘플링.

**Video-LLaVA 연동**  
05번 VLM + ANOMALY 키프레임 → 풍부한 자연어 요약으로 고도화 가능.

---

## Part 6 — ChromaDB 시맨틱 검색

### Vision·Video AI / NLP 용어 (강의용)

**Semantic Search(시맨틱 검색)**  
키워드 exact match 없이 **의미 유사** 문장 검색. "컨베이어 멈춤" → "저활동/대기 상태" 리포트 매칭.

**Text Embedding(텍스트 임베딩)**  
문장 → 고차원 벡터. 교육용 `_hash_embed`, 실무 **sentence-transformers**.

**Vector DB / ChromaDB**  
embedding 저장·cosine similarity 검색. `USE_CHROMA=1` 시 `output_tracking/chroma_db/` 영속.

---

## Part 7 — MOT 지표 · manifest

### Vision·Video AI 전문 용어 (강의용)

**ID Switch(ID 전환)**  
주요 track_id가 프레임 간 **바뀐 횟수**. 높으면 occlusion·detection miss로 궤적 단절. MOT 품질 모니터링 지표.

**Track Continuity(추적 연속성)**  
track_id별 **평균 관측 프레임 수**. 높을수록 안정적 association.

**MOTA / MOT Metrics(근사)**  
본 실습은 GT 없이 ID Switch·Continuity·Processing FPS로 **세션 품질** 모니터링. 실 GT 있으면 MOTA·IDF1 등 정식 MOT metric 가능.

**Tracking Manifest**  
`tracking_manifest.json` — frame별 bbox, track_id, operation_state, motion_z, report. **08번 MANIFEST_SOURCES** 확장 입력.

---

## 08번 연동

`MANIFEST_SOURCES`에 `tracking_manifest.json` 추가 → `operation_state`, `motion_z` unified_score 병합.

---

*본 문서는 `09_Video_Object_Tracking_Equipment_State.ipynb` 소스문서로 작성되었습니다.*
