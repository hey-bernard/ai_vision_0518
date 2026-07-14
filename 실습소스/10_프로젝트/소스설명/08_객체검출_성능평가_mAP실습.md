# 08. 객체검출 성능평가 mAP 실습 — 소스 설명

대응 노트북: `notebooks/08_객체검출_성능평가_mAP실습.ipynb`

## 한눈에 보기

객체 검출 점수의 핵심인 **IoU → TP/FP → Precision-Recall → AP → mAP**를  
직접 구현해 이해합니다. (데모용으로 예측을 시뮬레이션)

산출물: `outputs/detection_eval/`

## 초보자를 위한 용어 사전

| 용어 | 쉬운 설명 |
|------|-----------|
| **GT (Ground Truth)** | 정답 박스. 라벨에 적힌 ‘진짜 위치’. |
| **Prediction** | 모델이 예측한 박스 + 신뢰도(confidence). |
| **IoU** | 두 박스의 겹친 면적 ÷ 합친 면적. 1이면 완전 일치, 0이면 안 겹침. |
| **IoU threshold** | ‘맞췄다’고 인정하는 최소 IoU. 보통 0.5. |
| **TP / FP / FN** | 정탐 / 오탐 / 미검. |
| **AP (Average Precision)** | 한 클래스의 PR 곡선 아래 면적. 그 클래스 검출 실력. |
| **mAP** | 클래스별 AP의 평균. |
| **mAP@0.5** | IoU≥0.5 기준 mAP. 비교적 관대한 기준. |
| **NMS** | 겹치는 중복 박스를 정리하는 후처리. Non-Maximum Suppression. |
| **xyxy** | 왼쪽위(x1,y1) ~ 오른쪽아래(x2,y2) 좌표 형식. |

## IoU 그림으로 이해하기

```text
        ┌─────┐
        │  겹침 │
   ┌────┼──┐  │
   │    └─────┘
   │  박스 A   │
   └─────────┘
IoU = 겹침 / (A면적 + B면적 - 겹침)
```

## AP 계산 절차 (노트북과 동일)

1. 해당 클래스 예측을 confidence 높은 순으로 정렬
2. 각 예측에 대해 같은 이미지 GT 중 최대 IoU 찾기
3. IoU≥0.5 이고 아직 안 쓰인 GT면 TP, 아니면 FP
4. 누적 TP/FP로 Precision, Recall 곡선
5. 곡선 면적 = AP
6. 클래스 평균 = mAP@0.5

## 실제 모델로 바꾸는 법

`simulate_predictions()` 대신 아래 형식의 `preds`를 넣으면 됩니다.

```python
{"image_id": "파일stem", "class_id": 2, "conf": 0.87, "bbox": (x1,y1,x2,y2)}
```
