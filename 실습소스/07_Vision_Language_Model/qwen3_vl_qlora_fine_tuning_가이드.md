# Qwen3-VL-4B QLoRA 파인튜닝 노트북 상세 가이드

대상 파일: [`08_qwen3_vl_qlora_fine_tuning.ipynb`](./08_qwen3_vl_qlora_fine_tuning.ipynb)

이 문서는 노트북의 **사용법**, **전체 실행 흐름**, **섹션별 기능**, **주요 함수·변수**, **VRAM·재실행 시나리오**를 상세히 설명합니다.

---

## 목차

1. [개요](#1-개요)
2. [사전 요구사항](#2-사전-요구사항)
3. [전체 실행 흐름](#3-전체-실행-흐름)
4. [산출물(저장 파일)](#4-산출물저장-파일)
5. [섹션별 상세 설명](#5-섹션별-상세-설명)
6. [주요 함수 레퍼런스](#6-주요-함수-레퍼런스)
7. [하이퍼파라미터 레퍼런스](#7-하이퍼파라미터-레퍼런스)
8. [부분 재실행 가이드](#8-부분-재실행-가이드)
9. [VRAM·성능 참고](#9-vram성능-참고)
10. [트러블슈팅](#10-트러블슈팅)
11. [관련 자료](#11-관련-자료)

---

## 1. 개요

### 1.1 이 노트북이 하는 일

| 항목 | 내용 |
|------|------|
| 베이스 모델 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (4B VLM) |
| 학습 방식 | **4bit QLoRA** + TRL `SFTTrainer` 지도학습(SFT) |
| 데이터셋 | [mosshoon/KoLLaVA-Instruct-1.5k](https://huggingface.co/datasets/mosshoon/KoLLaVA-Instruct-1.5k) (~1,500 한국어 이미지 VQA) |
| LoRA 대상 | attention `q_proj`, `v_proj` |
| 목표 GPU | RTX 3060 **12GB** 기준 |
| 예상 학습 시간 | 약 **80분~2시간** (1 epoch, 최초 모델·데이터 다운로드 제외) |

### 1.2 파이프라인 한 줄 요약

```
환경 설정 → 데이터 로드·포맷 → 학습 전 fp16 추론(baseline)
  → QLoRA SFT 학습 → LoRA 저장 → 학습 전·후 비교 → (선택) LoRA merge → 병합 모델 추론
```

### 1.3 선행·후속 노트북

| 노트북 | 역할 |
|--------|------|
| `06_qwen3_vl_example.ipynb` | Qwen3-VL 추론 기초 |
| `07_qwen3_vl_qlora_mini_train.ipynb` | QLoRA 미니 체험 |
| `vlm101-sft-hands-on-main/ko_sft_lora_qwen2.5vl.ipynb` | 원본 ko_sft 스타일 참고 |

---

## 2. 사전 요구사항

### 2.1 하드웨어

- **NVIDIA GPU + CUDA** 필수 (QLoRA 학습은 CPU 불가)
- 권장: **12GB VRAM** 이상 (3060, 4060 8GB는 매우 빡빡)
- 디스크: 모델·데이터·체크포인트 합쳐 **20GB+** 여유 권장

### 2.2 소프트웨어

프로젝트 루트에서:

```bash
pip install -r requirements.txt
```

핵심 패키지:

| 패키지 | 최소 버전 | 역할 |
|--------|-----------|------|
| `transformers` | 4.57.0+ | Qwen3-VL 모델·프로세서 |
| `trl` | 0.9.0+ | `SFTTrainer`, `SFTConfig` |
| `peft` | 0.10.0+ | LoRA 어댑터 |
| `bitsandbytes` | 0.43.0+ | 4bit NF4 양자화 (Windows 주의) |
| `datasets` | 2.18.0+ | Hugging Face 데이터셋 |
| `torch` | 2.0+ | CUDA 연산 |
| `accelerate` | 0.27+ | `device_map` 분산 로드 |

### 2.3 Hugging Face 접근

- `mosshoon/KoLLaVA-Instruct-1.5k`, `Qwen/Qwen3-VL-4B-Instruct` Hub 다운로드 필요
- 최초 실행 시 인터넷 연결·충분한 디스크 공간 필요
- gated 모델/데이터셋이면 HF 로그인·약관 동의 필요

---

## 3. 전체 실행 흐름

### 3.1 처음부터 끝까지 (권장 순서)

노트북을 **위에서 아래로 순서대로** 실행합니다.

| 순서 | 섹션 | 필수 | 대략 소요 |
|------|------|------|-----------|
| 1 | §1 환경 설정 | ✅ | 1분 |
| 2 | §2 KoLLaVA 데이터셋 로드 | ✅ | 1~3분 |
| 3 | §3 학습 전 베이스 추론 | ✅ | 5~10분 (fp16 로드) |
| 4 | §4 QLoRA 파인튜닝 | ✅ | **~80분** |
| 5 | §5 파인튜닝 결과 비교 | 권장 | 2~5분 |
| 6 | §5 URL 테스트 | 선택 | ~1.2분 |
| 7 | §6 LoRA merge | 선택 | 5~15분 |
| 8 | §6 병합 모델 추론 | 선택 | 3~10분 |

### 3.2 §4 내부 실행 순서

§4는 여러 코드 블록으로 나뉘어 있으며, **반드시 아래 순서**를 지킵니다.

1. `clear_memory()` 정의 및 실행 — §3 fp16 모델 해제
2. 4bit QLoRA 모델 로드 (`freeze_vision_modules`, `patch_input_embeddings`)
3. `LoraConfig` + `get_peft_model`
4. `SFTConfig` (training_args)
5. `collate_fn` 정의
6. `SFTTrainer` 생성 → `trainer.train()`
7. `trainer.save_model(OUTPUT_DIR)` + processor 저장

### 3.3 §5 비교 실행 순서

1. `clear_memory()` (선택, GPU 정리)
2. **파인튜닝 결과 비교 블록** 셀 실행 (LoRA 4bit 로드 + 헬퍼 정의)
3. train 샘플 비교 (#33)
4. eval 샘플 비교 (#27)
5. (선택) URL 테스트

> **중요:** URL 테스트·eval 비교 전에 **비교 블록 셀**을 먼저 실행해야 `compare_sample`, `model`, `processor`가 정의됩니다.

---

## 4. 산출물(저장 파일)

학습·merge 완료 후 생성되는 경로:

```
./qwen3-vl-4b-qlora-sft-ko-1.5k/
├── adapter_config.json          # LoRA 설정
├── adapter_model.safetensors    # LoRA 가중치 (~수십 MB)
├── tokenizer 관련 파일
├── processor 설정
├── checkpoint-*/                # 학습 중간 체크포인트 (save_steps마다)
└── merged_model/                # §6 merge 후 (선택)
    ├── model-*.safetensors      # fp16 병합 전체 가중치 (~8~9GB)
    ├── config.json
    └── tokenizer·processor
```

| 산출물 | 용도 |
|--------|------|
| `OUTPUT_DIR` (LoRA만) | 12GB에서 **일상 추론**에 적합, VRAM ~4GB |
| `merged_model/` | 배포·공유·다른 환경 이전용 **단일 체크포인트** |

**학습은 디스크에 저장되므로** 커널 재시작 후에도 `OUTPUT_DIR`이 있으면 **80분 학습을 다시 할 필요 없습니다.**

---

## 5. 섹션별 상세 설명

### §1. 환경 설정

#### 목적

- Python 패키지 import
- 전역 하이퍼파라미터·경로 정의
- GPU·dtype 환경 확인

#### 코드 블록

**1) 패키지 설치 셀 (주석 처리된 `%pip install`)**

- 최초 환경에서만 주석 해제 후 실행
- `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets` 설치

**2) import + 하이퍼파라미터 셀**

정의되는 핵심 변수:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODEL_ID` | `Qwen/Qwen3-VL-4B-Instruct` | 베이스 VLM |
| `DATASET_ID` | `mosshoon/KoLLaVA-Instruct-1.5k` | 학습 데이터 HF repo |
| `OUTPUT_DIR` | `./qwen3-vl-4b-qlora-sft-ko-1.5k` | LoRA·processor 저장 경로 |
| `TEST_SIZE` | `0.1` | eval 10% |
| `NUM_EPOCHS` | `1` | 학습 epoch |
| `BATCH_SIZE` | `1` | GPU 배치 |
| `GRAD_ACCUM` | `8` | 유효 배치 = 8 |
| `MIN_PIXELS` / `MAX_PIXELS` | `256×28×28` | vision 토큰 상한 (OOM 방지) |
| `LORA_R` / `LORA_ALPHA` | `8` / `16` | LoRA rank·스케일 |
| `LEARNING_RATE` | `2e-4` | AdamW 학습률 |
| `infer_dtype` | fp16 또는 bf16 | fp16 **전체 모델** 추론용 |
| `compute_dtype` | fp16 | 4bit QLoRA 연산 dtype |

**`make_bnb_config()`**

- NF4 4bit 양자화 설정 팩토리
- `load_in_4bit=True`, `double_quant=True`
- §4 QLoRA 로드·§5 비교 블록에서 재사용

---

### §2. KoLLaVA 데이터셋 로드

#### 목적

- Hugging Face에서 KoLLaVA 1.5k 다운로드
- Qwen3-VL `apply_chat_template` 호환 **messages** 형식으로 변환
- train/eval 분할

#### 원본 데이터 스키마

KoLLaVA 각 행:

| 필드 | 타입 | 설명 |
|------|------|------|
| `images` | PIL Image | 질문 대상 이미지 |
| `questions` | str | 사용자 질문 (`<image>` 포함 가능) |
| `answers` | str | 한국어 정답 |

#### `system_message`

모든 샘플에 동일 적용되는 영문 시스템 프롬프트:

- 이미지 기반 답변
- **한국어**, 상세·이해하기 쉬운 설명 요구

#### `format_data(sample)`

KoLLaVA 1행 → Qwen3 messages 리스트 3턴:

```
[
  { role: "system",    content: [text] },
  { role: "user",      content: [image, text] },   # <image> 토큰 제거
  { role: "assistant", content: [text] },
]
```

**왜 `<image>`를 제거하나?**

- Qwen3 processor는 `content`의 `{"type":"image"}` 항목으로 이미지를 처리
- 텍스트에 남은 `<image>` 플레이스홀더는 중복·혼란 유발

#### 데이터 로드·분할 셀

```python
ds = load_dataset(DATASET_ID)
dataset = ds["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = [format_data(s) for s in train_raw]   # ~1350
eval_dataset  = [format_data(s) for s in eval_raw]    # ~150
```

#### 탐색 셀

- `train_dataset[0]` — messages 구조 확인
- `train_dataset[0][1]['content'][0]['image']` — PIL 이미지 접근 예시

---

### §3. 학습 전 베이스 모델 추론

#### 목적

- 파인튜닝 **전** fp16/bf16 **전체 모델**로 답변 품질 확인
- `output_before` 저장 → §5에서 fp16 baseline과 비교

#### 추론 헬퍼 (이 섹션에서 정의, 이후 전 구간 재사용)

| 함수 | 역할 |
|------|------|
| `configure_qwen3_processor()` | `min_pixels`/`max_pixels`로 vision 해상도 제한 |
| `move_inputs_to_model()` | `pixel_values` 등 텐서를 모델 device·dtype으로 이동 |
| `generate_text_from_sample()` | chat sample → `apply_chat_template` → `generate` → assistant 텍스트만 반환 |

**`generate_text_from_sample` 동작 상세**

1. sample에 system 턴이 있으면 **user 턴만** 사용 (`sample[1:2]`)
2. `add_generation_prompt=True`로 assistant 응답 시작 토큰 추가
3. `do_sample=False` (greedy)
4. 선택: `repetition_penalty`, `no_repeat_ngram_size` (§5 LoRA 반복 방지)
5. 생성된 토큰 중 **입력 길이 이후**만 decode

#### 베이스 모델 로드

```python
model_org = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=infer_dtype
)
```

- **QLoRA 아님** — fp16 전체 가중치 (~8~10GB VRAM)
- §4 직전 `clear_memory()`로 **반드시 해제** 필요

#### 샘플 추론

- `sample_idx = 33` — 스케이트보드 안전 VQA (train)
- `output_before` — 이후 §5 train 비교의 **original (fp16)** 기준

---

### §4. QLoRA 파인튜닝 (TRL SFTTrainer)

#### 목적

- fp16 베이스를 GPU에서 내리고 4bit QLoRA로 재로드
- KoLLaVA train set SFT
- LoRA 어댑터 + processor를 `OUTPUT_DIR`에 저장

#### `clear_memory()`

삭제 대상: `model_org`, `model`, `peft_model`, `trainer`, `processor`, `base_4bit` 등

- `gc.collect()` + `torch.cuda.empty_cache()`
- §3 → §4, §5 전, merge 전 등 **모델 교체 지점**에서 호출

#### Qwen3-VL 학습 전용 패치

| 함수 | 이유 |
|------|------|
| `freeze_vision_modules()` | visual encoder `requires_grad=False` — LLM LoRA만 학습, VRAM·시간 절약 |
| `patch_input_embeddings()` | Qwen3-VL + gradient checkpointing 시 `get_input_embeddings` NotImplementedError 방지 |

#### 4bit 모델 로드

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=compute_dtype,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False
```

- VRAM: 약 **6~10GB**
- `processor.tokenizer.padding_side = "right"` — 배치 padding 방향

#### LoRA 설정

```python
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
```

- `get_peft_model()`로 학습 가능 파라미터 비율 출력 (교육용)

#### `SFTConfig` 핵심 포인트

| 설정 | 값 | 이유 |
|------|-----|------|
| `optim` | `paged_adamw_8bit` | VRAM 절약 |
| `fp16`/`bf16` | `False` | 4bit QLoRA와 AMP 충돌 방지 |
| `max_length` | `None` | truncation 시 vision 토큰·input_ids 불일치 방지 |
| `dataset_kwargs` | `skip_prepare_dataset=True` | VLM은 `collate_fn`이 직접 처리 |
| `assistant_only_loss` | `False` | collate_fn에서 labels 직접 구성 |
| `report_to` | `"none"` | W&B 등 로깅 비활성 |
| `load_best_model_at_end` | `True` | eval_loss 최저 체크포인트 복원 |

#### `collate_fn` — VLM 학습의 핵심

배치 1 step 처리:

1. `extract_images_from_messages()` — messages에서 PIL 이미지 추출
2. `apply_chat_template(..., tokenize=False)` — 텍스트 프롬프트 문자열
3. `processor(text=..., images=..., padding=True)` — **truncation 없음**
4. `labels = input_ids.clone()`
5. pad 토큰 → `-100` (loss 제외)
6. `mask_vision_tokens()` — vision 특수 토큰 → `-100`

마스킹 대상 vision 토큰:

- `<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`, `<|image_pad|>`, `<|video_pad|>`

#### 학습 실행

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor.tokenizer,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
```

- RTX 3060 기준 약 **80분**
- `save_steps=20`마다 `OUTPUT_DIR/checkpoint-*` 생성

---

### §5. 파인튜닝 결과 비교

#### 목적

- 학습 전·후 답변 품질 시각적 비교
- 12GB VRAM에서 **CPU offload 없이** 비교

#### 12GB 비교 전략

| 비교 대상 | train #33 | eval #27 / URL |
|-----------|-----------|----------------|
| baseline | §3 `output_before` (fp16) | `model.disable_adapter()` (4bit base) |
| finetuned | LoRA ON (4bit) | LoRA ON (4bit) |

> fp16 베이스를 다시 로드하면 12GB에서 CPU offload·`meta device` 경고·극심한 저속이 발생합니다.  
> 따라서 eval/URL은 **adapter OFF/ON** 방식을 사용합니다.

#### 파인튜닝 결과 비교 블록 (단독 실행 가능)

이 셀 하나에 비교에 필요한 것이 모두 포함됩니다:

- 전역 변수 폴백 (`MODEL_ID`, `OUTPUT_DIR`, …)
- `_configure_qwen3_processor`, `_generate`, `compare_sample`
- 4bit base + `PeftModel.from_pretrained(OUTPUT_DIR)` 로드

**`compare_sample(sample)`**

```python
with model.disable_adapter():
    output_org = _generate(...)          # 4bit 베이스
output_ft = _generate(..., repetition_penalty=1.12, no_repeat_ngram_size=3)
```

- LoRA 반복 출력 방지: `repetition_penalty=1.12`, `no_repeat_ngram_size=3`

#### train 샘플 비교 (#33, 스케이트보드)

- `output_before`가 메모리에 있으면 fp16 baseline 재사용
- 없으면 `compare_sample()` (adapter off/on)

#### eval 샘플 비교 (#27, 북극곰)

- `max_new_tokens=256`
- `compare_sample(eval_dataset[27])`

#### URL 테스트 (커스텀 이미지)

**`load_image_from_url(url)`**

- `requests` + PIL로 JPEG/PNG 다운로드 → RGB
- Qwen3 processor는 `messages`의 `url` 키 직접 사용 시 Wikimedia 등에서 실패할 수 있음 → **PIL `image` 키 사용**

**`compare_external_vqa(image, question)`**

- messages 구성 후 `compare_sample()` 위임
- 테스트 이미지: COCO val2017 곰 (`000000439715.jpg`)
- RTX 3060 기준 약 1.2분

---

### §6. LoRA 가중치 병합 (merge)

#### 목적

- LoRA 어댑터를 베이스 fp16 가중치에 **영구 합침**
- 단일 `merged_model/` 체크포인트로 배포·이전

#### merge 셀 동작

```python
clear_memory()
base_model = Qwen3VLForConditionalGeneration.from_pretrained(...)  # fp16
peft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{OUTPUT_DIR}/merged_model", ...)
processor.save_pretrained(merged_dir)
```

- merge 후 모델 크기: **~8.9GB** (4B 전체)
- LoRA만 쓸 때: **수십 MB**

#### merge 직후 OOM·재시작 안내 셀

커널 재시작 없이 merge만 다시 할 때 필요한 섹션:

```
§1 환경설정 → §2 데이터셋 → §3 헬퍼함수 → §4 clear_memory 정의 → §6 merge
```

학습(`trainer.train()`)은 **다시 하지 않음** — `OUTPUT_DIR`에 LoRA가 이미 있으면 됨.

#### 병합 모델 로드 및 추론

merge 직후 **커널 재시작 없이** 실행 가능하도록 설계:

1. `base_model`, `peft_model`, `merged_model` 등 **삭제**
2. `clear_memory()` 또는 `gc` + `empty_cache`
3. `merged_dir`에서 fp16 모델 로드 — `device_map={"": 0}` (GPU 전용, offload 방지)
4. `train_dataset[33]`으로 `generate_text_from_sample` 추론

> 12GB에서 fp16 병합 모델 단독 추론은 여전히 빡빡합니다.  
> 일상 사용은 §5 **4bit LoRA** 추론을 권장하고, merge는 저장·배포용으로 활용하세요.

---

### §7. 트러블슈팅

노트북 마지막 섹션 표 요약 + 아래 확장 설명.

| 증상 | 원인 | 해결 |
|------|------|------|
| `get_input_embeddings` NotImplementedError | Qwen3-VL + checkpointing | `patch_input_embeddings(model)` |
| CUDA OOM | vision 토큰 과다·배치 | `MAX_PIXELS` 유지, `BATCH_SIZE=1` |
| `meta device` / CPU offload | fp16 모델 중복·VRAM 부족 | GPU 비우기, `device_map={"":0}`, LoRA 추론 사용 |
| `Unsupported image file` (URL) | processor가 URL 직접 디코딩 실패 | PIL 선로드 + `image` 키 |
| LoRA 반복 출력 | 짧은 eval·긴 토큰 | `repetition_penalty=1.12`, `no_repeat_ngram_size=3` |
| `train_dataset` NameError | §2 미실행 | §2 데이터 로드 셀 실행 |
| `processor` NameError (merge) | `clear_memory()`가 processor 삭제 | `OUTPUT_DIR`에서 processor 재로드 (merge 셀에 포함) |
| `compare_sample` 없음 | 비교 블록 미실행 | §5 비교 블록 셀 먼저 실행 |

#### 품질 향상 (VRAM·시간 증가)

| 조정 | 기본 | 향상 예시 |
|------|------|-----------|
| `target_modules` | q, v | +k, o |
| `LORA_R` / `ALPHA` | 8/16 | 16/32 |
| `NUM_EPOCHS` | 1 | 2~3 (과적합 주의) |
| `MAX_PIXELS` | 256×28² | 384×28² |

---

## 6. 주요 함수 레퍼런스

### 데이터·포맷

| 함수 | 입력 | 출력 | 섹션 |
|------|------|------|------|
| `format_data(sample)` | KoLLaVA dict | messages list | §2 |
| `extract_images_from_messages(messages)` | messages | PIL list | §4 |
| `mask_vision_tokens(labels, tokenizer)` | labels tensor | masked labels | §4 |
| `collate_fn(examples)` | messages 배치 | processor batch dict | §4 |

### 모델·메모리

| 함수 | 역할 | 섹션 |
|------|------|------|
| `make_bnb_config()` | 4bit NF4 설정 | §1 |
| `clear_memory()` | GPU 변수 해제 | §4 |
| `freeze_vision_modules(model)` | vision 동결 | §4 |
| `patch_input_embeddings(model)` | embedding 패치 | §4 |
| `configure_qwen3_processor(processor, min, max)` | pixel 상·하한 | §3 |

### 추론·비교

| 함수 | 역할 | 섹션 |
|------|------|------|
| `move_inputs_to_model(inputs, model, dtype)` | 텐서 device 이동 | §3 |
| `generate_text_from_sample(...)` | 단일 샘플 추론 | §3 |
| `_generate(...)` | 비교 블록 추론 (폴백 포함) | §5 |
| `compare_sample(sample)` | adapter OFF vs ON | §5 |
| `load_image_from_url(url)` | URL → PIL RGB | §5 |
| `compare_external_vqa(image, question)` | 외부 이미지 VQA 비교 | §5 |

---

## 7. 하이퍼파라미터 레퍼런스

### 학습 규모 계산

```
유효 배치 크기 = BATCH_SIZE × GRAD_ACCUM = 1 × 8 = 8
optimizer step ≈ len(train_dataset) × NUM_EPOCHS / 유효 배치
               ≈ 1350 × 1 / 8 ≈ 169 step (1 epoch)
```

### Vision 토큰·해상도

```
MAX_PIXELS = 256 × 28 × 28 = 200,704 pixels
```

- Qwen3 기본 HD 설정은 vision 토큰 **8000+** → 12GB OOM
- `configure_qwen3_processor`로 상·하한 고정 필수

### dtype 정리

| 변수 | 용도 | 대상 |
|------|------|------|
| `infer_dtype` | fp16/bf16 전체 모델 | §3 `model_org`, §6 merge |
| `compute_dtype` | 4bit 연산 | §4 QLoRA, §5 LoRA 추론 |

---

## 8. 부분 재실행 가이드

### 8.1 학습 완료 후 — 비교만 (§5)

```
§1 import 셀 → §2 전체 → §3 헬퍼함수 셀 → §5 clear_memory(선택) → §5 비교 블록 → 비교·URL 셀
```

`trainer.train()` **실행 안 함**.

### 8.2 학습 완료 후 — merge만 (§6)

```
§1 → §3 헬퍼함수 → §4 clear_memory 정의 → §6 merge
```

### 8.3 merge 완료 후 — 병합 모델 추론만

```
§1 → §2 (train_dataset 필요) → §3 헬퍼함수 → §4 clear_memory 정의
→ §6 병합 모델 추론 셀
```

merge 셀은 **이미 `merged_model/`이 있으면 생략** 가능.

### 8.4 merge 직후 — 재시작 없이 추론

```
§6 merge 셀 완료 → §6 병합 모델 추론 셀
```

추론 셀이 merge 잔여 모델을 삭제·GPU 비운 뒤 `merged_dir`에서 단독 로드합니다.

### 8.5 메모리 오류 시 (노트북 내 안내)

커널 재시작 후:

1. §1 환경 설정 — import 셀
2. §2 KoLLaVA — 모든 코드 셀
3. §3 — 헬퍼함수 셀만
4. §4 — `clear_memory` 정의 셀만
5. 이후 목적지 섹션 (merge / 추론 등)

---

## 9. VRAM·성능 참고

| 작업 | 대략 VRAM | 12GB (3060) |
|------|-----------|-------------|
| QLoRA 학습 (4bit) | 6~10GB | ✅ |
| 4bit LoRA 추론 | ~4~5GB | ✅ |
| fp16 베이스 추론 (§3) | 8~10GB | ⚠️ 단독만 |
| fp16 merge + 추론 | 8~12GB | ⚠️ 빡빡 |
| fp16 + 4bit 동시 | 14GB+ | ❌ |

**권장 GPU (병합 fp16 추론까지 여유):** 16GB+  
**권장 GPU (이 노트북 전체 여유):** 24GB+

---

## 10. 트러블슈팅

§7 표와 동일하며, 실무에서 자주 나오는 케이스만 추가 정리합니다.

### 학습 중 끊겼을 때

- `OUTPUT_DIR/checkpoint-*` 확인
- `trainer` 재생성 시 `resume_from_checkpoint=True` 또는 checkpoint 경로 지정 가능 (노트북 기본은 미포함 — 수동 추가 필요)

### `DATASET_ID` 변경

- `mosshoon/KoLLaVA-Instruct-1.5k` 외 mirror 사용 가능
- **필수 스키마:** `images`, `questions`, `answers`
- 로컬 경로도 가능: `DATASET_ID = r"C:\data\KoLLaVA-Instruct-1.5k"`

### W&B / 로깅

- `report_to="none"` 고정 — 외부 실험 추적 없음
- 필요 시 `report_to="wandb"` 등으로 변경 (별도 설정 필요)

---

## 11. 관련 자료

- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [mosshoon/KoLLaVA-Instruct-1.5k](https://huggingface.co/datasets/mosshoon/KoLLaVA-Instruct-1.5k)
- [TRL Qwen3-VL SFT 노트북](https://github.com/huggingface/trl/blob/main/examples/notebooks/sft_qwen_vl.ipynb)
- [HF VLM SFT Cookbook](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_trl.ipynb)
- [QLoRA 논문/레포](https://github.com/artidoro/qlora)
- [bitsandbytes 문서](https://huggingface.co/docs/bitsandbytes)

---

*문서 버전: `08_qwen3_vl_qlora_fine_tuning.ipynb` 기준 (mosshoon 데이터셋, adapter OFF/ON 비교, merge 후 GPU 비우기 추론 반영)*
