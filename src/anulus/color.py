from fileinput import close
from typing import List

import cv2
import numpy as np

from . import auto_brighten


def enclose_red(img: np.array,
                lower_thrershold=((120, 50, 50), (150, 255, 255)),
                upper_thrershold=((175, 60, 50), (180, 255, 255)),
                red_thresh=125,
                op_brighten=False,
                op_brighten_hsv=True,
                add_hue=40) -> np.array:
    """
    This function takes four arguments and isolates the red color. The red
    image is later given to HoughCircles to detect circles so it's necessary that image is clean
    and pristine and has no other objects but the red objects on the picture.

    Param
    -----
    img: np.arrray:
        The Numpy array containing the image signals.
    lower_threshold: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        The lowermost threshold in HSV values where H is from 0 to 180 and
        S and V are from 0 to 255 (unlike normal HSV where it's double)
        This value is added to the higher threshold upon masking.
    higher_threshold: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        The highermost threshold in HSV values where H is from 0 to 180 and
        S and V are from 0 to 255 (unlike normal HSV where it's double)
        This value is added to the lower threshold upon masking.
    red_thresh: int
        Basically after the image is AND'd with itself asnd the mask of the higher
        and lower threshld it is then filtered through RGB values to get the most
        saturated spots in the image. Pass a value that you think would represent
        the R, the G and the B of the sign and this single value acts for al l3.
    op: bool
        Do operations on the HSV image or not

    Returns
        final_image: np.array
            The filtered and thresholeded image (if chosen)

    """
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_ELLIPSE, ksize=(9, 9))

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    copy_img = img.copy()

    if op_brighten:
        img, _, _ = auto_brighten.automatic_brightness_and_contrast(img)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
    if op_brighten_hsv:
        hsv_img[:, :, 0] = np.where(
            (hsv_img[:, :, 0] > 100 - add_hue) &
            (hsv_img[:, :, 0] < 100 + (add_hue * 2)),
            hsv_img[:, :, 0] + add_hue, hsv_img[:, :, 0])

        hsv_img[:, :, 0] = np.clip(hsv_img[:, :, 0], 0, 180)

        normed = cv2.normalize(hsv_img, None, 0, 180,
                               cv2.NORM_MINMAX, cv2.CV_8UC1)

        close = cv2.morphologyEx(normed, cv2.MORPH_CLOSE, kernel)        

        hsv_img = cv2.GaussianBlur(normed, (5, 5), 0)

    lower_mask = cv2.inRange(hsv_img, lower_thrershold[0],
                             lower_thrershold[1])
    upper_mask = cv2.inRange(hsv_img, upper_thrershold[0],
                             upper_thrershold[1])

    isolated = cv2.bitwise_and(
        copy_img, copy_img, mask=lower_mask + upper_mask)
    close = cv2.morphologyEx(isolated, cv2.MORPH_CLOSE, kernel)
    copy_img = cv2.GaussianBlur(close, (5, 5), 0)

    return np.where(copy_img > red_thresh, copy_img, 0)
