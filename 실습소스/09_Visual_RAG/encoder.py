"""SigLIP2 인코더 — 이미지·텍스트를 동일 벡터 공간으로 매핑."""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from conveyor_rag.config import SIGLIP2_MODEL_ID


class SigLIP2Encoder:
    """SigLIP2 So400M 이미지/텍스트 임베딩."""

    def __init__(self, model_id: str = SIGLIP2_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        # GPU 사용 가능 시 cuda + float16으로 VRAM 절약
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = None
        self.processor = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """모델/프로세서가 해제된 경우 재로드."""
        if self.model is None:
            self.model = (
                AutoModel.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device).eval()
            )
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id)

    @staticmethod
    def _pool_features(output) -> torch.Tensor:
        """모델 출력 형태(텐서 / pooler_output)를 통일된 feature 텐서로 변환."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        raise TypeError(f"Unexpected feature output type: {type(output)}")

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        """PIL Image 리스트 → L2 정규화된 임베딩 배열 (N, dim)."""
        self._ensure_loaded()
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self._pool_features(self.model.get_image_features(**inputs))
        # 코사인 유사도 = 내적 이므로 L2 정규화 필수
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def unload(self) -> None:
        """Qwen3-VL 로드 전 GPU 메모리 해제."""
        # SigLIP2와 Qwen을 동시에 GPU에 올리면 12GB에서 OOM 발생 가능
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """한글/영어 텍스트 → L2 정규화 임베딩 (ChromaDB 쿼리·분석 텍스트 저장용)."""
        self._ensure_loaded()
        inputs = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)
        feats = self._pool_features(self.model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()
