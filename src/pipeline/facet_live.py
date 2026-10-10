"""Real-time Facet42-compatible feature extraction from a webcam frame.

Uses py-feat (open-source FACET-equivalent) to detect the face and estimate
the 20 Action Units and head pose, then maps them to the 35-dimensional
Facet 4.2 feature vector the trained video model expects:

  Index  Name         Source
  -----  ----         ------
  0      Anger        zero  (requires commercial Facet SDK)
  1      Contempt     zero
  2      Disgust      zero
  3      Joy          zero
  4      Fear         zero
  5      Baseline     zero
  6      Sadness      zero
  7      Surprise     zero
  8      Confusion    zero
  9      Frustration  zero
  10     AU1          inner brow raise   (py-feat AU01, probit-mapped)
  11     AU2          outer brow raise   (py-feat AU02)
  12     AU4          brow lowerer       (py-feat AU04)
  13     AU5          upper lid raiser   (py-feat AU05)
  14     AU6          cheek raiser       (py-feat AU06)
  15     AU7          lid tightener      (py-feat AU07)
  16     AU9          nose wrinkler      (py-feat AU09)
  17     AU10         upper lip raiser   (py-feat AU10)
  18     AU12         lip corner puller  (py-feat AU12)
  19     AU14         dimpler            (py-feat AU14)
  20     AU15         lip corner depressor (py-feat AU15)
  21     AU17         chin raiser        (py-feat AU17)
  22     AU18         lip puckerer       zero (py-feat has no AU18 model)
  23     AU20         lip stretcher      (py-feat AU20)
  24     AU23         lip tightener      (py-feat AU23)
  25     AU24         lip pressor        (py-feat AU24)
  26     AU25         lips part          (py-feat AU25)
  27     AU26         jaw drop           (py-feat AU26)
  28     AU28         lip suck           (py-feat AU28)
  29     AU43         eyes closed        (py-feat AU43)
  30     Has_Glasses  zero (not detectable)
  31     Is_Male      zero
  32     Pitch        head pitch (degrees, py-feat)
  33     Yaw          head yaw
  34     Roll         head roll

Why py-feat: the commercial FACET/iMotions SDK cannot be installed or
distributed, and the MediaPipe geometric-ratio approximation it replaced is
not calibrated to FACS intensity. py-feat's xgb AU model is trained on FACS-
coded data (BP4D / DISFA / UNBC) and outputs AU probabilities in [0,1].
The raw FACET 4.2 training features are approximately standard-normal
evidence scores (~50% negative, per-dim std ~1), so probabilities are sent
through the probit (inverse normal CDF) to land in the same distributional
family before the training-split normalisation is applied.

The Facet SDK emotion channels (cols 0-9) and binary metadata (30-31) cannot
be replicated without the commercial SDK, so they remain zero and the model
relies on the AU and head-pose columns (10-34).

First use downloads py-feat model weights (retinaface, mobilefacenet, xgb
AU) from HuggingFace — a one-time ~50 s cost, cached afterwards.

Usage
-----
    extractor = FacetLiveExtractor()
    frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    feat_35 = extractor.extract(frame_rgb)  # np.ndarray (35,) or None
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_META_PATH   = PROJECT_ROOT / "data" / "processed" / "video" / "video_fer" / "metadata.json"

FACET42_DIMS = [
    "Anger","Contempt","Disgust","Joy","Fear","Baseline","Sadness","Surprise",
    "Confusion","Frustration",
    "AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU12","AU14",
    "AU15","AU17","AU18","AU20","AU23","AU24","AU25","AU26","AU28","AU43",
    "Has_Glasses","Is_Male","Pitch","Yaw","Roll",
]
assert len(FACET42_DIMS) == 35

# FACET AU column (10-29) -> py-feat output column. AU18 (col 22) has no
# py-feat model and stays zero.
_AU_MAP = [
    (10, "AU01"), (11, "AU02"), (12, "AU04"), (13, "AU05"), (14, "AU06"),
    (15, "AU07"), (16, "AU09"), (17, "AU10"), (18, "AU12"), (19, "AU14"),
    (20, "AU15"), (21, "AU17"),               (23, "AU20"), (24, "AU23"),
    (25, "AU24"), (26, "AU25"), (27, "AU26"), (28, "AU28"), (29, "AU43"),
]

_MAX_FRAME_WIDTH = 480   # ~0.7 s/frame at this size on CPU
_FACE_SCORE_MIN  = 0.5


def _stub_torchcodec() -> None:
    """py-feat 2.x imports torchcodec unconditionally, but its wheels require
    a newer torch than the pinned 2.5.1 and fail to load (WinError 127).
    Webcam frames go through detect(data_type='tensor'), which never touches
    the video decoder, so a stub keeps image detection working."""
    import sys
    import types

    pkg = types.ModuleType("torchcodec")
    pkg.__path__ = []
    decoders = types.ModuleType("torchcodec.decoders")

    class VideoDecoder:  # pragma: no cover - never instantiated on image path
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torchcodec video decoding is unavailable")

    decoders.VideoDecoder = VideoDecoder
    pkg.decoders = decoders
    sys.modules.setdefault("torchcodec", pkg)
    sys.modules.setdefault("torchcodec.decoders", decoders)


def _probit(p: np.ndarray) -> np.ndarray:
    """Map probabilities in [0,1] to standard-normal z-scores."""
    from scipy.stats import norm

    p = np.clip(p.astype(np.float64), 1e-4, 1.0 - 1e-4)
    return norm.ppf(p).astype(np.float32)


class FacetLiveExtractor:
    """
    Extract a (35,) Facet42-compatible feature vector from a single RGB frame
    using py-feat (retinaface face detection, mobilefacenet landmarks,
    xgb AU model).

    Parameters
    ----------
    apply_norm : bool
        If True (default), apply the training-split normalisation so the
        output is in the same scale the model was trained on.
    """

    def __init__(self, apply_norm: bool = True):
        _stub_torchcodec()
        from feat.detector import Detectorv1

        self._detector = Detectorv1(
            au_model="xgb",
            emotion_model=None,
            face_model="retinaface",
            landmark_model="mobilefacenet",
            device="cpu",
        )

        self._norm_mean: np.ndarray | None = None
        self._norm_std:  np.ndarray | None = None
        if apply_norm and _META_PATH.exists():
            meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
            norm = meta.get("normalization", {})
            if "mean" in norm and "std" in norm:
                self._norm_mean = np.asarray(norm["mean"], dtype=np.float32)
                self._norm_std  = np.asarray(norm["std"],  dtype=np.float32)
                self._norm_std  = np.where(self._norm_std > 0, self._norm_std, 1.0)

    def extract(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from one RGB frame (H, W, 3).
        Returns np.ndarray (35,) float32 or None if no confident face is found.
        """
        import torch

        if frame_rgb is None:
            return None

        if frame_rgb.dtype != np.uint8:
            frame_rgb = (frame_rgb * 255).clip(0, 255).astype(np.uint8)

        h, w = frame_rgb.shape[:2]
        if w > _MAX_FRAME_WIDTH:
            import cv2  # local import to keep module importable without cv2
            scale = _MAX_FRAME_WIDTH / w
            frame_rgb = cv2.resize(
                frame_rgb, (_MAX_FRAME_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA
            )

        t = torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1).unsqueeze(0)

        # py-feat 2.1.5 draws tqdm bars even with progress_bar=False
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            out = self._detector.detect(t, data_type="tensor", progress_bar=False)
        if out is None or len(out) == 0:
            return None

        row = out.iloc[int(out["FaceScore"].argmax())]
        if float(row["FaceScore"]) < _FACE_SCORE_MIN:
            return None

        feat = np.zeros(35, dtype=np.float32)

        p = np.array([float(row[c]) for _, c in _AU_MAP], dtype=np.float32)
        z = _probit(p)
        for (idx, _), val in zip(_AU_MAP, z):
            feat[idx] = val

        feat[32] = float(row["Pitch"])   # degrees, as in FACET training data
        feat[33] = float(row["Yaw"])
        feat[34] = float(row["Roll"])

        if self._norm_mean is not None:
            feat = (feat - self._norm_mean) / self._norm_std
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat.astype(np.float32)

    def close(self):
        self._detector = None

    def __del__(self):
        try:
            self._detector = None
        except Exception:
            pass
