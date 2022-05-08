from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel


class ClassifierType(str, Enum):
    ORB = 'ORB'
    SSID = 'SSID'


class ClassiferAggMode(str, Enum):
    MEAN = "MEAN"
    MIN = "MIN"
    MAX = "MAX"
    MEDIAN = "MEDIAN"
    AVG = "AVG"
    VAR = "VAR"


class ClassifierPostOp(str, Enum):
    MIN = "MIN"
    MAX = "MAX"


class ClassifierThreshComparator(str, Enum):
    LARGER_THAN = "LARGER_THAN"
    SMALLER_THAN = "SMALLER_THAN"
    LARGER_THAN_EQ = "LARGER_THAN_EQ"
    SMALLER_THAN_EQ = "SMALLER_THAN_EQ"


class MatchNorm(str, Enum):
    HAMMING = 'HAMMING'
    HAMMING2 = 'HAMMING2'
    L1 = 'L1'
    L2 = 'L2'
    L2SQR = 'L2SQR'
    MINMAX = 'MINMAX'
    INF = 'INF'


class Settings(BaseModel):
    """
    This object includes the settings for detection.

    Attributes
    ----------
    color_low: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
        The low color threshold in HSV
    color_high: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
        The high color threshold in HSV
    red_thresh: Optional[int]
        The supposed threshold to filter color after reds are filtered
    dp: Optional[float]
        Circle detection; inverse ratio of the accumulator resolution to the image resolution
    min_dist_circle: Optional[int]
        Circle detection; minimum distance of circles from one another
    min_radius: Optional[int]
        Circle detection; minimum radius of the supposed circles
    max_radius: Optional[int]
        Circle detection; maximum radius of the supposed circles
    param_1: Optional[int]
        The first parameter for HOUGH_CIRCLES_GRADIENT method; the higher threshold of the two passed to the Canny edge detector
    param_2: Optional[int]
        The second parameter for HOUGH_CIRCLES_GRADIENT method; he accumulator threshold for the circle centers at the detection stage
    thresh_temp: Optional[float] 
        The SSIM threshold with wihch the detected circles should be to the target signs
    do_op: Optional[bool]
        Whether to do opening, closing, blurring etc operations on image upon color isolation (prefrably on)
    """
    color_low: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = (
        (120, 50, 50), (130, 255, 255))
    color_high: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = (
        (150, 60, 50), (180, 255, 255))
    red_thresh: Optional[int] = 120
    dp: Optional[float] = 2.8
    min_dist_circle: Optional[int] = 400
    min_radius: Optional[int] = 2
    max_radius: Optional[int] = 250
    param_1: Optional[int] = 400
    param_2: Optional[int] = 120
    do_op: Optional[bool] = True
    do_op_hsv: Optional[bool] = True
    do_op_circle: Optional[bool] = True
    add_hue: Optional[int] = 20
    do_classify: Optional[bool] = True
    classifier: Optional[ClassifierType] = ClassifierType.ORB
    classifier_norm: Optional[MatchNorm] = MatchNorm.HAMMING2
    classifier_aggmode: Optional[ClassiferAggMode] = ClassiferAggMode.MEAN
    classifier_threshold: Optional[float] = 50
    classifer_postop: Optional[ClassifierPostOp] = ClassifierPostOp.MIN
    classifier_thresh_comp: Optional[ClassifierThreshComparator] = ClassifierThreshComparator.SMALLER_THAN_EQ

    class Config:
        use_enum_values = True


class Coords(BaseModel):
    y1: int
    y2: int
    x1: int
    x2: int
