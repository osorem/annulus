from typing import List

import cv2
import numpy as np

from . import settings as st

KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

def detect_circle(
        img: np.array,
        dp=3.3,
        min_dist=500,
        min_radius=5,
        max_radius=10,
        param_1=200,
        param_2=20,
        algo=st.CircleAlgo.GRADIENT,
        op_list=[st.CircleOps.OP_CLOSE]) -> np.array:
    """
    Detect circle is the only shape detect function that is used in the final detect
    function so it's the only one worth documenting.

    And as far as documentation goes it's exactly what it says on the tin.
    This function uses Hough Transform to find circles through cv2.HoughCircles.

    HoughCircles has two modes, `GRADIENT` and `GRADIENT_ALT`.
    We use the former because the lattr is useless in our case.

    This function needs a lot of tweaking to work.

    Params
    ------
    img: np.array
        The numpy array containing the image signals.
    dp: float
        inverse ratio of the accumulator resolution to the image resolution
    min_dist: int
        Minimum distance between detected circles.
    min_radius: int
        Minimum radius to be dtcected
    max_radius: int
        maximum radius to be detected
    param_1: int
        the higher threshold of the two passed to the Canny edge detector


    Returns
    -------
        Circle locations: np.array([x, y, r])

    """

    img_copy = img.copy()

    for op in op_list:
        if op == st.CircleOps.OP_BLUR:
            kernel_size = 5
            img_copy = cv2.GaussianBlur(img_copy,
                         (kernel_size, kernel_size), 0)
        if op == st.CircleOps.OP_SHARPEN:
            if len(img_copy.shape) == 2:
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            img_copy = cv2.filter2D(img_copy, -1, KERNEL)
            img_copy = cv2.detailEnhance(img_copy)

        if op == st.CircleOps.OP_CLOSE:
            kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_ELLIPSE, ksize=(8, 8))
            img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)

        if op == st.CircleOps.OP_THRESHOLD:
            if len(img_copy.shape) == 3:
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            img_copy = cv2.adaptiveThreshold(img_copy.astype(np.uint8), 255, 1, 1, 11, 2)

        if op == st.CircleOps.OP_NORMALIZE:
            if len(img_copy.shape) == 2:
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            img_copy = cv2.normalize(img_copy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    if len(img_copy.shape) == 2:
        gray = img_copy
    else:
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    algo_circle = cv2.HOUGH_GRADIENT

    if algo == st.CircleAlgo.GRADIENT:
        algo_circle = cv2.HOUGH_GRADIENT
    elif algo == st.CircleAlgo.GRADIENT_ALT:
        algo_circle = cv2.HOUGH_GRADIENT_ALT


    circles = cv2.HoughCircles(
        gray,
        algo_circle,
        dp,
        min_dist,
        minRadius=min_radius,
        maxRadius=max_radius,
        param1=param_1,
        param2=param_2)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(circles)
        return circles

    return []


def detect_line(img: np.array,
                edge_low_threshold=50,
                edge_high_threshold=150,
                rho=1,
                theta=45 * (np.pi / 180),
                threshold=15,
                min_line_length=20,
                max_line_gap=20) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    normed = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
    opened = cv2.morphologyEx(normed, cv2.MORPH_OPEN, kernel)
    kernel_size = 5
    blur_opened = cv2.GaussianBlur(opened, (kernel_size, kernel_size), 0)

    edges = cv2.Canny(blur_opened, edge_low_threshold, edge_high_threshold)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is not None:
        return True

    return False


def detect_rectangle(
    img, kernel_size=(
        3, 3), w_extrema=(
            20, 50), h_extrema=(
                10, 20)) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    normed = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_ELLIPSE, ksize=kernel_size)
    opened = cv2.morphologyEx(normed, cv2.MORPH_OPEN, kernel)
    kernel_size = 5
    blur_opened = cv2.GaussianBlur(opened, (kernel_size, kernel_size), 0)
    contours, _ = cv2.findContours(
        blur_opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        bbox = cv2.boundingRect(cnt)

        x, y, w, h = bbox

        if w_extrema[0] <= w < w_extrema[1] and h_extrema[0] <= h < h_extrema[1]:
            return True

    return False


def detect_triangle(img, eps=1.07) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, eps * cv2.arcLength(cnt, True), True)
        if len(approx.ravel()) == 3:
            return True
    return False
