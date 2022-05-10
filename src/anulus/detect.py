import cv2
import numpy as np

from . import auto_brighten, color
from . import crop as crp
from . import matcher
from . import settings as st
from . import shape, template
from . import ops

KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])


def anulus_detect(img_path: str, stn: st.Settings, pyrd=True, resize=(820, 600)) -> np.array:
    """
    This is the main detect function.

    Params
    ------
        img_path: str
            Path to img, relative or absolute, does not matter
        settings: Settigngs
            The settings object you created.
        pyrd: bool
            Whethet to pyrDown the image or not (derease qualiy and size.

    Returns
        Final image annotated: np.array    
        Score dict: List 
            final SSIM score dict
    """
    print("Began anulus-detect-MAIN")

    p = st.Printer(verbose=stn.global_verbose)

    img = cv2.imread(img_path)

    if pyrd:
        p("Pyring down the image as pyrd=True...")
        img = cv2.pyrDown(img)
        img = cv2.resize(img, resize)

    p("Isolating the color red based on your settings...")
    color_isolated = color.enclose_red(
        img, stn.color_low,
        stn.color_high, stn.color_red_thresh,
        op_brighten=stn.color_auto_brighten,
        op_brighten_hsv=stn.color_op_hsv,
        op_sharpen=stn.color_sharpen,
        add_red=stn.color_add_red,
        add_hue=stn.color_add_hue,
        add_val=stn.color_add_value,
        add_sat=stn.color_add_saturation,
        post_ops=stn.color_post_ops,
        convert_hsv=stn.color_convert_hsv)

    circles = shape.detect_circle(
        color_isolated,
        stn.circle_dp,
        stn.circle_min_dist_from,
        stn.circle_min_radius,
        stn.circle_max_radius,
        stn.circle_param_1,
        stn.circle_param_2,
        op_list=stn.circle_op_list,
        algo=stn.circle_algo
    )

    output = img.copy()

    res = {}

    h, w, _ = img.shape

    p(f"Found {len(circles)} circles...")

    added = 0

    for i, circle in enumerate(circles):
        p(f"Operating on circle {i + 1}/{len(circles)}...")

        x, y, r = circle
        p("Getting the real colors...")
        if y >= r:
            y_left, y_right = y - r, y + r
        else:
            y_left, y_right = y, y + r

        if x >= r:
            x_left, x_right = x - r, x + r
        else:
            x_left, x_right = x, x + r

        isolated = color_isolated[y_left:y_right, x_left:x_right, :]

        if 0 in isolated.shape[:2]:
            p("Size of cropped image was 0, continuing...")
            continue

        cd_r = st.Coords(
            x1=x_left,
            y1=y_left,
            x2=x_right,
            y2=y_right
        )

        img_isolated_only = np.zeros((h, w, 3))
        img_isolated_only[y_left:y_right,
                          x_left:x_right] = np.where(
            crp.imcrop(color_isolated, cd_r) > 0, 1, 0)
        

        ys, xs, _ = np.where(img_isolated_only > 0)

        if len(ys) == 0 or len(xs) == 0:
            p("Image larger than zero empty, continuing...")
            continue

        
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        
        p("Cropping the image...")
        cd = st.Coords(
            x1=x_min - stn.classifier_add_bb,
            x2=x_max + stn.classifier_add_bb,
            y1=y_min - stn.classifier_add_bb,
            y2=y_max + stn.classifier_add_bb
        )

        img_cropped = crp.imcrop(img, cd)
        p("Getting the variance...")
        crp_var = np.var(img_cropped)

        if stn.detect_min_variance != 0:
            if crp_var < stn.detect_min_variance:
                p("Variance too small, continuing...")
                continue
        p("Operating on cropped image...")

        img_cropped = cv2.resize(img_cropped, (400, 400))

        img_cropped = cv2.GaussianBlur(img_cropped, (5, 5),
                                       cv2.BORDER_DEFAULT)
        img_cropped = cv2.filter2D(img_cropped, -1, KERNEL)
        img_cropped = cv2.detailEnhance(img_cropped)

        try:
            img_cropped, _, _ = auto_brighten.automatic_brightness_and_contrast(
                img_cropped)
        except:
            pass

        if stn.classifier == st.ClassifierType.ORB:
            temp, dct, agg_score = matcher.orb_matcher(img_cropped,
                                                       stn.classifier_threshold,
                                                       stn.classifier_norm,
                                                       stn.classifier_aggmode,
                                                       stn.classifer_postop,
                                                       stn.classifier_thresh_comp)
            p(f"Got an aggscore of {agg_score}")
        else:
            temp, dct = template.get_max_sim(img_cropped, stn.thresh_temp)

        if temp == -1:
            p("Could not detect any of the sign shapes based on given templates...")
            continue

        p("Shape detected, adding to list...")
        cv2.rectangle(output, (cd.x1, cd.y1), (cd.x2, cd.y2),
                      (0, 255, 0), thickness=2)
        cv2.circle(output, (x, y), r, (0, 180, 0), 2)

        if stn.do_classify:
            p("DoClassify enabled, marking classification...")
            cv2.putText(output, temp,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,
                        color=(0, 210, 0), thickness=2, org=(x, y))

        temp = f"{i + 1} - {temp}"

        res[temp] = {}
        res[temp]['agg_score'] = agg_score
        res[temp]['coords'] = {"real": (x, y, r), "adjusted": cd}
        res[temp]['scores'] = dct
        res[temp]['cropped'] = img_cropped
        res[temp]['variance'] = crp_var

        added += 1

    if added == 0:
        p("Warning: No signs detected, output image won't have any marks...")

    p("Done! Returning the output image, scores, sign coordinates and isolated color.")

    return output, color_isolated, res


def anulus_detect_alt(img_path: str, stn: st.Settings, pyrd=True, resize=(820, 600)) -> np.array:
    print("Began anulus-detect-ALT")

    p = st.Printer(verbose=stn.global_verbose)

    img = cv2.imread(img_path)

    if pyrd:
        p("Pyring down the image as pyrd=True...")
        img = cv2.pyrDown(img)
        img = cv2.resize(img, resize)

    p("Preprocessing the image...")
    img_preprocessed = ops.preprocess_image(img)

    if stn.preprocess_cleanup:
        img_preprocessed = ops.remove_small_components(
            img_preprocessed,
            threshold=stn.preprocess_threshold
        )

    if stn.preprocess_remlines:
        img_preprocessed = ops.remove_line(
            img_preprocessed
        )


    p("Detecting signs...")

    cnts = ops.find_contour(img_preprocessed)
    p(f"Found {len(cnts)} contours")

    signs, coords = ops.find_signs(img_preprocessed, cnts, stn.detect_threshold,    
                         stn.detect_distance_threshold)
    
    output = img.copy()

    if len(coords) == 0:
        p("Did not find any signs.")
        return output, signs, coords, img_preprocessed


    for i, c in enumerate(coords):
        p(f"Operating on detected sign {i + 1} / {len(coords)}")

        ul_c, br_c = c

        x1, y1 = ul_c
        x2, y2 = br_c

        p("Cropping the image...")
        cd = st.Coords(
            x1=x1 - stn.classifier_add_bb,
            x2=x2 + stn.classifier_add_bb,
            y1=y1 - stn.classifier_add_bb,
            y2=y2 + stn.classifier_add_bb
        )

        img_cropped = crp.imcrop(img, cd)
        p("Getting the variance...")
        crp_var = np.var(img_cropped)

        if stn.detect_min_variance != 0:
            if crp_var < stn.detect_min_variance:
                p("Variance too small, continuing...")
                continue
        p("Operating on cropped image...")

        img_cropped = cv2.resize(img_cropped, (400, 400))

        img_cropped = cv2.GaussianBlur(img_cropped, (5, 5),
                                       cv2.BORDER_DEFAULT)
        img_cropped = cv2.filter2D(img_cropped, -1, KERNEL)
        img_cropped = cv2.detailEnhance(img_cropped)

        try:
            img_cropped, _, _ = auto_brighten.automatic_brightness_and_contrast(
                img_cropped)
        except:
            pass

        temp, dct, agg_score = matcher.orb_matcher(img_cropped,
                                                       stn.classifier_threshold,
                                                       stn.classifier_norm,
                                                       stn.classifier_aggmode,
                                                       stn.classifer_postop,
                                                       stn.classifier_thresh_comp)
        p(f"Got an aggscore of {agg_score}")

        if temp == -1:
            p("Could not detect any of the sign shapes based on given templates...")
            continue

        p("Shape detected, adding to list...")
        cv2.rectangle(output, (cd.x1, cd.y1), (cd.x2, cd.y2),
                      (0, 255, 0), thickness=2)


    return output, signs, coords, img_preprocessed
