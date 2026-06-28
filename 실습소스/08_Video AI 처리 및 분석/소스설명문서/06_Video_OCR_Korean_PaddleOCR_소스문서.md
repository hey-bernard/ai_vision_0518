# 06 Video OCR Korean PaddleOCR — 소스문서

> **대상:** Video AI 초보자  
> **원본:** `06_Video_OCR_Korean_PaddleOCR.ipynb`  
> **실습 데이터:** AI_Hub_한글_OCR.zip

---

## 강의 전체 흐름

| Part | 주제 | 한 줄 요약 |
|------|------|-----------|
| 1 | AI Hub 데이터 | Scene Text GT |
| 2 | Detection | Morphology proposal · IoU |
| 3 | PaddleOCR | pretrained det+rec · CER |
| 4 | 통합 파이프라인 | ocr_manifest |

---

## OCR 개요

영상 OCR = **프레임마다 2D OCR 반복**. VLM 요약과 달리 **정확한 문자열 + bbox 좌표**가 필요합니다(HMI·라벨·경고문).

---

## Part 1 — AI Hub 한글 OCR

### Vision·Video AI 전문 용어 (강의용)

**OCR(Optical Character Recognition, 광학 문자 인식)**  
이미지·프레임에서 **글자 위치(detection)** 와 **글자 내용(recognition)** 을 읽는 기술입니다. Document OCR(스캔 문서)과 **Scene Text OCR**(간판·HMI·현장 촬영)은 난이도·전처리가 다릅니다. 본 데이터셋은 Scene Text입니다.

**Scene Text(씬 텍스트)**  
자연 장면·설비 화면에 **임의 각도·크기·조명**으로 나타나는 텍스트입니다. 배경 복잡·왜곡·저해상도로 Document OCR보다 어렵습니다.

**Ground Truth, GT(정답 라벨)**  
사람이标注한 bbox + transcription입니다. AI Hub JSON의 wordbox `[x,y,w,h]` + `value`.

**TextBox / wordbox**  
글자 영역 bounding box. Detection 평가·Recognition crop 입력으로 씁니다.

---

## Part 2 — 텍스트 검출

### Vision·Video AI 전문 용어 (강의용)

**Text Detection(텍스트 검출)**  
"글자가 **어디** 있는가" — bbox만 찾습니다. Recognition과 분리된 2단계 OCR의 1단계입니다.

**Text Recognition(텍스트 인식)**  
Detection으로 crop한 영역에서 **어떤 문자열**인지 읽습니다.

**IoU(Intersection over Union, 교집합/합집합)**  
예측 bbox와 GT bbox 겹침 비율 0~1입니다. ≥0.5면 "같은 글자 영역을 맞췄다"고 보는 관례가 많습니다. Detection **Precision/Recall** 계산의 기본입니다.

**Precision / Recall (검출)**  
Precision = proposal 중 GT 맞은 비율(허위 alarm↓). Recall = GT 중 proposal이 잡은 비율(놓침↓).

**Morphology-based Proposal(형태학 기반 후보)**  
adaptive threshold + MORPH_CLOSE + findContours로 bbox 후보를 만드는 **전통 CV** baseline입니다. 교육용·베이스라인으로 F1이 낮을 수 있음 → Part 3 PaddleOCR와 비교.

---

## Part 3 — PaddleOCR

### Vision·Video AI 전문 용어 (강의용)

**PaddleOCR**  
Baidu 오픈소스 OCR. **한글 det+rec pretrained** weight 제공. 본 실습은 직접 학습 없이 사전학습 모델로 inference합니다.

**DB(D Differentiable Binarization)**  
PaddleOCR **검출** 백엔드. 텍스트 영역을 differentiable하게 이진화해 bbox polygon을 예측합니다. curved·dense text에 강한 편입니다.

**Angle Classifier(방향 분류기)**  
기울어진·회전된 글자 crop의 **읽기 방향(0°/180° 등)** 을 보정합니다. Scene Text에서 recognition 정확도 향상에 기여합니다.

**CRNN / SVTR(인식 백엔드)**  
CRNN: CNN feature + RNN sequence decoding. SVTR: Transformer 기반 recognition. PaddleOCR 한글 rec 모델 내부 아키텍처 후보입니다.

**CER(Character Error Rate, 글자 오류율)**  
Levenshtein **편집 거리**를 글자 수로 나눈 값. 0=완벽, 1=전부 틀림. Recognition 품질의 표준 지표입니다.

---

## Part 4 — 통합 OCR 파이프라인

- `run_ocr_pipeline()` → `ocr_manifest.json` + predictions PNG
- IPAD 프레임: `ocr_manifest_ipad.json` → 08번 manifest 병합

---

## 05 VLM vs 06 OCR

| | VLM | OCR |
|---|-----|-----|
| 출력 | 자연어 요약 | **정확한 문자열+좌표** |
| 용도 | 장면 설명 | HMI·라벨·표 읽기 |

---

*본 문서는 `06_Video_OCR_Korean_PaddleOCR.ipynb` 소스문서로 작성되었습니다.*
