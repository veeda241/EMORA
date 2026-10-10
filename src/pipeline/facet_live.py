"""Real-time Facet42-compatible feature extraction from a webcam frame.

Uses MediaPipe FaceMesh to extract 468 3D facial landmarks, then maps them
to the 35-dimensional Facet 4.2 feature vector the trained video model
expects:

  Index  Name         Source
  -----  ----         ------
  0      Anger        mean  (requires commercial Facet SDK)
  1      Contempt     mean
  2      Disgust      mean
  3      Joy          mean
  4      Fear         mean
  5      Baseline     mean
  6      Sadness      mean
  7      Surprise     mean
  8      Confusion    mean
  9      Frustration  mean
  10     AU1          inner brow raise   (landmark geometry)
  11     AU2          outer brow raise
  12     AU4          brow lowerer
  13     AU5          upper lid raiser
  14     AU6          cheek raiser
  15     AU7          lid tightener
  16     AU9          nose wrinkler
  17     AU10         upper lip raiser
  18     AU12         lip corner puller  (smile)
  19     AU14         dimpler
  20     AU15         lip corner depressor
  21     AU17         chin raiser
  22     AU18         lip puckerer
  23     AU20         lip stretcher
  24     AU23         lip tightener
  25     AU24         lip pressor
  26     AU25         lips part
  27     AU26         jaw drop
  28     AU28         lip suck        (approx)
  29     AU43         eyes closed     (blink)
  30     Has_Glasses  mean (not detectable from landmarks)
  31     Is_Male      mean
  32     Pitch        head pitch (radians from solvePnP)
  33     Yaw          head yaw
  34     Roll         head roll

The Facet SDK emotion channels (cols 0–9) and binary metadata (30–31) cannot
be replicated without the commercial SDK, so they are imputed at the
training-split mean (normalising to 0) and the model relies on the AU and
head-pose columns (10–34).

The normalisation from the TRAINING split is applied here so the model
receives the same distribution it was trained on.

Usage
-----
    extractor = FacetLiveExtractor()
    frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    feat_35 = extractor.extract(frame_rgb)  # np.ndarray (35,) or None
"""

from __future__ import annotations

import json
import math
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

# Columns the landmark pipeline cannot produce: the 10 commercial FACET
# emotion channels (0-9) and the binary flags (30-31). They are imputed at
# the training-split mean so they normalise to 0 — raw zeros would normalise
# to ~+0.3..+1.2 (the emotion-channel training means are ~-1..-2), telling
# the model every emotion is elevated and biasing it towards the majority
# class (happy).
_UNAVAILABLE_IDX = tuple(range(0, 10)) + (30, 31)

# MediaPipe landmark indices for AU approximation.
# References:
#   MediaPipe canonical face mesh: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
#   AU landmark mappings based on: Ekman (1978), iMotions AU reference
_LM = {
    # brows
    "inner_brow_L": 336, "inner_brow_R": 107,
    "outer_brow_L": 293, "outer_brow_R": 70,
    "brow_mid_L":   285, "brow_mid_R":   55,
    # eyes
    "eye_upper_L":  386, "eye_lower_L":  374,
    "eye_upper_R":  159, "eye_lower_R":  145,
    # nose
    "nose_tip":     4,   "nose_root":    168,
    "nostril_L":    279, "nostril_R":    49,
    # upper lip
    "upper_lip_mid": 13, "upper_lip_L":  62, "upper_lip_R":  292,
    # lower lip
    "lower_lip_mid": 14, "lower_lip_L":  78, "lower_lip_R":  308,
    # lip corners
    "lip_corner_L": 291, "lip_corner_R": 61,
    # cheeks
    "cheek_L":      427, "cheek_R":      205,
    # chin
    "chin":         152, "chin_low":     18,
    # jaw
    "jaw_L":        172, "jaw_R":        397,
    # forehead
    "forehead":     10,
}

# 3D model points for solvePnP (standard face model, mm scale)
_MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip
    [0.0,   -63.6, -12.5  ],   # Chin
    [-43.3,  32.7, -26.0  ],   # Left eye corner (from camera)
    [43.3,   32.7, -26.0  ],   # Right eye corner
    [-28.9, -28.9, -24.1  ],   # Left mouth corner
    [28.9,  -28.9, -24.1  ],   # Right mouth corner
], dtype=np.float64)

# Landmark indices for the 6 points above
_PNP_IDX = [4, 152, 263, 33, 287, 57]


def _lm_xy(lms, idx: int) -> tuple[float, float]:
    p = lms.landmark[idx]
    return p.x, p.y


def _lm_xyz(lms, idx: int) -> tuple[float, float, float]:
    p = lms.landmark[idx]
    return p.x, p.y, p.z


def _dist(lms, a: int, b: int) -> float:
    ax, ay = _lm_xy(lms, a)
    bx, by = _lm_xy(lms, b)
    return math.hypot(ax - bx, ay - by)


def _norm_dist(lms, a: int, b: int, ref: float) -> float:
    """Euclidean distance between landmarks normalised by a reference length."""
    return _dist(lms, a, b) / (ref + 1e-8)


def _eye_openness(lms, upper: int, lower: int, ref: float) -> float:
    """Vertical eye opening normalised by face height."""
    ux, uy = _lm_xy(lms, upper)
    lx, ly = _lm_xy(lms, lower)
    return abs(uy - ly) / (ref + 1e-8)


def _extract_aus(lms, fw: float, fh: float) -> np.ndarray:
    """
    Approximate 20 Action Units from MediaPipe landmarks.
    Returns a (20,) float32 array in the same order as FACET42_DIMS[10:30].
    All values are normalised geometry ratios (not calibrated to FACS intensity).
    """
    aus = np.zeros(20, dtype=np.float32)
    ref = fh  # face height as normalisation reference

    # AU1 — inner brow raise: inner brow vs forehead distance
    brow_inner_avg = 0.5 * (_lm_xy(lms, _LM["inner_brow_L"])[1] +
                             _lm_xy(lms, _LM["inner_brow_R"])[1])
    forehead_y = _lm_xy(lms, _LM["forehead"])[1]
    aus[0] = max(0.0, (brow_inner_avg - forehead_y) / (ref + 1e-8))

    # AU2 — outer brow raise: outer brow vs forehead
    brow_outer_avg = 0.5 * (_lm_xy(lms, _LM["outer_brow_L"])[1] +
                             _lm_xy(lms, _LM["outer_brow_R"])[1])
    aus[1] = max(0.0, (brow_outer_avg - forehead_y) / (ref + 1e-8))

    # AU4 — brow lowerer: inner brows moving toward each other
    inner_brow_dist = _dist(lms, _LM["inner_brow_L"], _LM["inner_brow_R"])
    outer_brow_dist = _dist(lms, _LM["outer_brow_L"], _LM["outer_brow_R"])
    aus[2] = max(0.0, 1.0 - (inner_brow_dist / (outer_brow_dist + 1e-8)))

    # AU5 — upper lid raiser: eye openness increase
    eye_L = _eye_openness(lms, _LM["eye_upper_L"], _LM["eye_lower_L"], ref)
    eye_R = _eye_openness(lms, _LM["eye_upper_R"], _LM["eye_lower_R"], ref)
    aus[3] = (eye_L + eye_R) / 2.0

    # AU6 — cheek raiser: cheek–lip corner distance
    cheek_lip_L = _dist(lms, _LM["cheek_L"], _LM["lip_corner_L"])
    cheek_lip_R = _dist(lms, _LM["cheek_R"], _LM["lip_corner_R"])
    aus[4] = ((cheek_lip_L + cheek_lip_R) / 2.0) / (ref + 1e-8)

    # AU7 — lid tightener: eye narrowing (inverse of AU5)
    aus[5] = max(0.0, 0.05 - aus[3])

    # AU9 — nose wrinkler: nostril vs nose tip distance
    nose_nostril_L = _dist(lms, _LM["nose_tip"], _LM["nostril_L"])
    nose_nostril_R = _dist(lms, _LM["nose_tip"], _LM["nostril_R"])
    aus[6] = ((nose_nostril_L + nose_nostril_R) / 2.0) / (ref + 1e-8)

    # AU10 — upper lip raiser: upper lip vs nose distance
    aus[7] = _dist(lms, _LM["upper_lip_mid"], _LM["nose_tip"]) / (ref + 1e-8)

    # AU12 — lip corner puller (smile): horizontal lip stretch
    lip_width = _dist(lms, _LM["lip_corner_L"], _LM["lip_corner_R"])
    aus[8] = lip_width / (ref + 1e-8)

    # AU14 — dimpler: lip corner horizontal spread vs face width
    fw_val = _dist(lms, _LM["outer_brow_L"], _LM["outer_brow_R"])
    aus[9] = lip_width / (fw_val + 1e-8)

    # AU15 — lip corner depressor: corner y vs lower lip y
    corner_y_avg = 0.5 * (_lm_xy(lms, _LM["lip_corner_L"])[1] +
                           _lm_xy(lms, _LM["lip_corner_R"])[1])
    lower_lip_y  = _lm_xy(lms, _LM["lower_lip_mid"])[1]
    aus[10] = max(0.0, (lower_lip_y - corner_y_avg) / (ref + 1e-8))

    # AU17 — chin raiser: chin vs lower lip distance
    aus[11] = _dist(lms, _LM["chin"], _LM["lower_lip_mid"]) / (ref + 1e-8)

    # AU18 — lip puckerer: lip width narrowing
    aus[12] = max(0.0, 1.0 - lip_width / (fw_val + 1e-8))

    # AU20 — lip stretcher: corners pulled back (wide open mouth feel)
    upper_y = _lm_xy(lms, _LM["upper_lip_mid"])[1]
    lower_y = _lm_xy(lms, _LM["lower_lip_mid"])[1]
    aus[13] = abs(upper_y - lower_y) / (ref + 1e-8)

    # AU23 — lip tightener: upper–lower lip proximity
    aus[14] = max(0.0, 0.05 - aus[13])

    # AU24 — lip pressor: same proxy
    aus[15] = aus[14]

    # AU25 — lips part: vertical gap between upper and lower lip
    aus[16] = aus[13]  # same as AU20 proxy

    # AU26 — jaw drop: jaw y vs chin y
    jaw_y_avg = 0.5 * (_lm_xy(lms, _LM["jaw_L"])[1] +
                        _lm_xy(lms, _LM["jaw_R"])[1])
    aus[17] = max(0.0, (_lm_xy(lms, _LM["chin_low"])[1] - jaw_y_avg) / (ref + 1e-8))

    # AU28 — lip suck: lip corners pulled inward
    aus[18] = aus[12]  # same proxy as puckerer

    # AU43 — eyes closed: inverse of eye openness
    aus[19] = max(0.0, 1.0 - (aus[3] * 20.0))  # clamp to ~[0,1]

    return aus


def _extract_head_pose(
    lms, h: int, w: int, fov_deg: float = 60.0
) -> tuple[float, float, float]:
    """
    Estimate head pose (pitch, yaw, roll) using solvePnP on 6 landmarks.
    Returns angles in degrees.
    """
    import cv2  # local import to keep module importable without cv2

    focal = w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    cam_matrix = np.array([
        [focal, 0,     w / 2.0],
        [0,     focal, h / 2.0],
        [0,     0,     1.0    ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_pts = np.array(
        [[lms.landmark[i].x * w, lms.landmark[i].y * h] for i in _PNP_IDX],
        dtype=np.float64,
    )

    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS, image_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    # Decompose rotation matrix to Euler angles (pitch, yaw, roll)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2( rmat[2, 1], rmat[2, 2]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll  = math.degrees(math.atan2( rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll  = 0.0
    return float(pitch), float(yaw), float(roll)


class FacetLiveExtractor:
    """
    Extract a (35,) Facet42-compatible feature vector from a single RGB frame.

    Uses MediaPipe FaceLandmarker (Tasks API, mediapipe >= 0.10 / 1.0).

    Parameters
    ----------
    apply_norm : bool
        If True (default), apply the training-split normalisation so the
        output is in the same scale the model was trained on.
    """

    def __init__(self, apply_norm: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, pathlib, tempfile

        # Download the FaceLandmarker model bundle if not already cached
        _model_cache = pathlib.Path(tempfile.gettempdir()) / "face_landmarker.task"
        if not _model_cache.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            print(f"Downloading MediaPipe FaceLandmarker model (~5 MB)... ", end="", flush=True)
            urllib.request.urlretrieve(url, _model_cache)
            print("done")

        base_opts = mp_python.BaseOptions(model_asset_path=str(_model_cache))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp

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
        Returns np.ndarray (35,) float32 or None if no face detected.
        """
        if frame_rgb is None:
            return None

        if frame_rgb.dtype != np.uint8:
            frame_rgb = (frame_rgb * 255).clip(0, 255).astype(np.uint8)

        h, w = frame_rgb.shape[:2]

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        detection = self._landmarker.detect(mp_image)

        if not detection.face_landmarks:
            return None

        lms = detection.face_landmarks[0]  # list of NormalizedLandmark

        # Adapt landmark access to new API (NormalizedLandmark objects with .x .y .z)
        class _LmsWrapper:
            """Thin wrapper so the AU/pose functions work unchanged."""
            def __init__(self, landmark_list):
                self.landmark = landmark_list

        lms_w = _LmsWrapper(lms)

        ys = [lm.y for lm in lms]
        xs = [lm.x for lm in lms]
        fh = max(ys) - min(ys)
        fw = max(xs) - min(xs)

        feat = np.zeros(35, dtype=np.float32)
        feat[10:30] = _extract_aus(lms_w, fw, fh)

        # Head pose from facial transformation matrix if available
        if (detection.facial_transformation_matrixes and
                len(detection.facial_transformation_matrixes) > 0):
            mat = np.array(detection.facial_transformation_matrixes[0])  # (4,4)
            if mat.shape == (4, 4):
                r = mat[:3, :3]
                sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
                if sy > 1e-6:
                    pitch = math.degrees(math.atan2( r[2, 1], r[2, 2]))
                    yaw   = math.degrees(math.atan2(-r[2, 0], sy))
                    roll  = math.degrees(math.atan2( r[1, 0], r[0, 0]))
                else:
                    pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
                    yaw   = math.degrees(math.atan2(-r[2, 0], sy))
                    roll  = 0.0
                feat[32] = float(pitch)
                feat[33] = float(yaw)
                feat[34] = float(roll)
        else:
            # Fallback: solvePnP
            pitch, yaw, roll = _extract_head_pose(lms_w, h, w)
            feat[32] = float(pitch)
            feat[33] = float(yaw)
            feat[34] = float(roll)

        if self._norm_mean is not None:
            for idx in _UNAVAILABLE_IDX:
                feat[idx] = self._norm_mean[idx]
            feat = (feat - self._norm_mean) / self._norm_std
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat.astype(np.float32)

    def close(self):
        self._landmarker.close()

    def __del__(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
