"""Image preprocessing shared by cache building and closed-loop inference (must be identical)."""
from __future__ import annotations

import numpy as np

BASE_ROWS = (40, 200)          # the fruit row lives here in the 256x256 third-person image
BASE_OUT = (100, 160)          # (H, W) after downscaling the 160x256 crop
HAND_OUT = (96, 96)


def preprocess(base_rgb: np.ndarray, hand_rgb: np.ndarray):
    import cv2
    b = base_rgb[BASE_ROWS[0]:BASE_ROWS[1]]
    b = cv2.resize(b, (BASE_OUT[1], BASE_OUT[0]), interpolation=cv2.INTER_AREA)
    h = cv2.resize(hand_rgb, (HAND_OUT[1], HAND_OUT[0]), interpolation=cv2.INTER_AREA)
    return b, h
