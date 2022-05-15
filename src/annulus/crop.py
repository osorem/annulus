import cv2
import numpy as np
from . import settings as st

def imcrop(img: np.array, cd: st.Coords) -> np.array:
    """
    Crops image
    """
    x1, y1, x2, y2 = cd.x1, cd.y1, cd.x2, cd.y2
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img: np.array, x1: int, x2: int, y1: int, y2: int):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                            -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2