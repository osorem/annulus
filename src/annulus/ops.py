import cv2
import numpy as np
from math import sqrt


def constrast_limit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized


def Laplacian_of_gaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image


def binarization(image):
    thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]
    return thresh


def preprocess_image(image):
    image = constrast_limit(image)
    image = Laplacian_of_gaussian(image)
    image = binarization(image)
    return image


def remove_line(img):
    gray = img.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 5
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength, maxLineGap)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return cv2.bitwise_and(img, img, mask=mask)


def remove_small_components(image, threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2


def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 128, 0])
    upper_blue = np.array([215, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0, 0, 128], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([170, 150, 50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)

    return mask


def find_contour(image):
    cnts, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    return cnts


def contour_is_sign(perimeter, centroid, threshold):
    result = []
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result]
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold:
        return True, max_value + 2
    else:
        return False, max_value + 2


def crop_contour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    return image[left:right, top:bottom]


def crop_sign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    return image[top:bottom, left:right]


def find_largests_sign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contour_is_sign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1, 2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinate = [(left-2, top-2), (right+3, bottom+1)]
            sign = crop_sign(image, coordinate)
    return sign, coordinate


def find_signs(image, contours, threshold, distance_theshold):
    signs = []
    coordinates = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, max_distance = contour_is_sign(c, [cX, cY], 1-threshold)
        if is_sign and max_distance > distance_theshold:
            sign = crop_contour(image, [cX, cY], max_distance)
            signs.append(sign)
            coordinate = np.reshape(c, [-1, 2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinates.append([(top-2, left-2), (right+1, bottom+1)])
    return signs, coordinates
