"""MediaPipe Selfie Multiclass로 이미지에서 "피부만" 마스크를 뽑는 유틸.

`texture_crop.py`가 다중 뷰 피부분할 투표로 스캔에서 발 부위를 크롭할 때 쓴다.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# MediaPipe Selfie Multiclass 클래스 id. body-skin/face-skin만 "피부"로 취급 —
# clothes(옷)/hair(머리카락)/others(장신구)는 제외한다. 주의: 손도 body-skin으로
# 잡히므로 이걸로 "발에 닿은 손" 오염은 못 막는다
SKIN_CLASS_IDS = frozenset({2, 3})
SELFIE_MULTICLASS_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
def _default_skin_model_path() -> Path:
    """PyInstaller로 얼린(frozen) 상태에서도 모델 파일을 찾는다.
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "data" / "models" / "selfie_multiclass_256x256.tflite"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[3] / "data" / "models" / "selfie_multiclass_256x256.tflite"


DEFAULT_SKIN_MODEL_PATH = _default_skin_model_path()


def load_skin_segmenter(model_path: Path = DEFAULT_SKIN_MODEL_PATH):
    """MediaPipe Selfie Multiclass 세그멘터를 로드한다. 모델 파일이 없으면 내려받는다."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[안내] 피부 정제 모델이 없어 내려받습니다: {model_path}")
        urllib.request.urlretrieve(SELFIE_MULTICLASS_URL, model_path)

    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_category_mask=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def skin_only_mask(segmenter, bgr_image: np.ndarray, *, erode: int = 8) -> np.ndarray:
    """이미지에서 "피부만" 바이너리 마스크(0/255)를 만든다.
    경계에서 `erode`px만큼 안쪽으로 깎아 옷과 피부의 경계선을 미리 제외한다.
    """
    import mediapipe as mp

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment(mp_image)
    cat_mask = result.category_mask.numpy_view()
    skin = np.isin(cat_mask, list(SKIN_CLASS_IDS)).astype(np.uint8) * 255
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        skin = cv2.erode(skin, kernel)
    return skin

