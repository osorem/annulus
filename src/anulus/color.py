from fileinput import close
from typing import List

import cv2
import numpy as np

from . import settings as st

from . import auto_brighten

KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])


def enclose_red(img: np.array,
                lower_thrershold=((120, 50, 50), (150, 255, 255)),
                upper_thrershold=((175, 60, 50), (180, 255, 255)),
                red_thresh=125,
                op_brighten=False,
                op_brighten_hsv=True,
                op_sharpen=False,
                post_ops=[st.ColorPostOps.OP_CLOSE],
                add_red=20,
                add_hue=40,
                add_val=20,
                add_sat=20,
                convert_hsv=False) -> np.array:
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


    copy_img = img.copy()

    if add_red:
        copy_img[:, :, 2] += add_red
        copy_img[:, :, 2] = np.clip(copy_img[:, :, 2], 0, 255)

    if op_sharpen:
        copy_img = cv2.filter2D(copy_img, -1, KERNEL)
        copy_img = cv2.detailEnhance(copy_img)

    if op_brighten:
        copy_img, _, _ = auto_brighten.automatic_brightness_and_contrast(copy_img)

    
    hsv_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2HSV)
        
    if op_brighten_hsv:
        hsv_img[:, :, 0] = np.where(
            (hsv_img[:, :, 0] > 100 - add_hue) &
            (hsv_img[:, :, 0] < 100 + (add_hue)),
            hsv_img[:, :, 0] + add_hue, hsv_img[:, :, 0])

        hsv_img[:, :, 0] = np.clip(hsv_img[:, :, 0], 0, 180)

        hsv_img[:, :, 1] = np.where(
            (hsv_img[:, :, 1] > 150 - add_sat) &
            (hsv_img[:, :, 1] < 150 + (add_sat)),
            hsv_img[:, :, 1] + add_sat, hsv_img[:, :, 1])

        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 1, 255)

        hsv_img[:, :, 2] = np.where(
            (hsv_img[:, :, 2] > 150 - add_val) &
            (hsv_img[:, :, 2] < 150 + (add_val)),
            hsv_img[:, :, 2] + add_val, hsv_img[:, :, 2])

        hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)

    lower_mask = cv2.inRange(hsv_img, lower_thrershold[0],
                             lower_thrershold[1])
    upper_mask = cv2.inRange(hsv_img, upper_thrershold[0],
                             upper_thrershold[1])

    if convert_hsv:
        copy_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    isolated = cv2.bitwise_and(
        copy_img, copy_img, mask=lower_mask + upper_mask)
    close = cv2.morphologyEx(isolated, cv2.MORPH_CLOSE, kernel)
    copy_img = cv2.GaussianBlur(close, (5, 5), 0)

    img_copy = np.where(copy_img > red_thresh, copy_img, 0)

    for op in post_ops:
        if op == st.ColorPostOps.OP_BLUR:
            kernel_size = 5
            img_copy = cv2.GaussianBlur(img_copy,       
                         (kernel_size, kernel_size), 0)
        if op == st.ColorPostOps.OP_SHARPEN:
            img_copy = cv2.filter2D(img_copy, -1, KERNEL)
            img_copy = cv2.detailEnhance(img_copy)

        if op == st.ColorPostOps.OP_CLOSE:
            kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE, ksize=(8, 8))
            img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)

        if op == st.ColorPostOps.OP_THRESHOLD:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            img_copy = cv2.adaptiveThreshold(img_copy.astype(np.uint8), 255, 1, 1, 11, 2)

        if op == st.ColorPostOps.OP_NORMALIZE:
            img_copy = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    return img_copy
