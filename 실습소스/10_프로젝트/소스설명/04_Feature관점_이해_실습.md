# 04. Feature 관점 이해 실습 — 소스 설명

대응 노트북: `notebooks/04_Feature관점_이해_실습.ipynb`

## 한눈에 보기

AI는 원본 픽셀을 그대로 “이해”하기보다, **다른 숫자 표현(Feature)** 으로 바꾼 뒤 구분합니다.

비유: 사람을 구분할 때 “눈·코·입 비율”처럼 **특징만 뽑아서** 비교하는 것과 비슷합니다.

## 초보자를 위한 용어 사전

| 용어 | 쉬운 설명 |
|------|-----------|
| **Feature (특징)** | 원본 데이터를 요약한 숫자들. 모델이 실제로 보는 입력의 핵심. |
| **Convolution (합성곱)** | 작은 필터(커널)를 이미지 위에 밀며 반응 값을 만드는 연산. |
| **커널 / 필터** | 3×3 같은 작은 숫자 격자. 엣지·블러 등을 강조하는 안경 역할. |
| **Feature Map** | 커널을 적용한 결과 이미지(반응 지도). |
| **엣지 (edge)** | 밝기가 급격히 바뀌는 경계선. 배선·결함 윤곽에 자주 나타남. |
| **고차원** | Feature가 많은 상태(예: 7개, 100개…). 직접 그리기 어려움. |
| **PCA** | 여러 숫자를 정보를 최대한 유지하며 2~3개로 줄여 그리는 방법. |
| **Feature Engineering** | 도메인 지식으로 유용한 파생 특징을 만드는 작업. |
| **Rolling mean/std** | 최근 N개 구간의 평균/표준편차. 부드럽게 보거나 흔들림을 봄. |
| **z-score** | (값-평균)/표준편차. ‘평균에서 몇 걸음 떨어졌나’. |
| **라벨 누수 (leakage)** | 미래 정보나 정답을 몰래 Feature에 넣어 시험 점수가 가짜로 높아지는 실수. |

## 코드 핵심

### `conv2d_valid`

순수 numpy로 2D 합성곱을 직접 구현합니다.  
Identity / Edge / Blur / Sharpen 커널별 Feature Map을 시각화합니다.

해석 포인트:

- Vertical/Horizontal Edge → 배선 경계에서 반응↑
- Laplacian → 급격한 변화(결함 경계) 강조
- Blur → 노이즈↓, 미세 결함도 약해질 수 있음

### `extract_basic_image_features`

이미지 1장 → 7개 숫자(평균, 표준편차, 분위수, gradient energy, edge ratio).

### PCA (SVD 사용)

7차원 Feature를 2차원으로 투영해 산점도로 봅니다.

### 센서 Feature Engineering

lux/temp/vibration에 rolling, diff, z-score, 조합 feature를 만들고  
결함 이벤트 구간과 얼마나 차이나는지 Top feature를 고릅니다.

## 실행 흐름

```text
샘플 이미지 → 커널별 Feature Map
→ 통계 Feature 추출 → PCA 2D
→ 센서 시계열 생성 → 파생 Feature → 이벤트 구분력 확인
```

## 실무 체크리스트 (노트북 요약)

- 현업 말로 Feature를 설명할 수 있는가?
- 노이즈/드리프트에 강한가?
- 학습과 서비스에서 **같은 계산식**인가?
- Feature를 늘린 만큼 성능이 올랐는가?
