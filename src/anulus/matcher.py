import inspect
import os
from typing import Dict, Union

import cv2
import numpy as np

from . import settings as st

path = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))


path_matchers = os.path.join(path, 'matchers')

imgs_temp = [os.path.join(path_matchers, name) for name in os.listdir(
    path_matchers) if name.endswith(".png")]

print(f"Loaded {len(imgs_temp)} images to be used for matching")

classes = {
    "bike_no_access_old.png": "No Acces for Bikes (Old)",
    "bike_no_access.png": "No Access for Bikes",
    "give_way.png": "Give Way",
    "no_access_axel_load_4_8.png": "No Axel Load Larger 4.8t",
    "no_access_bicycles_old.png": "No Access for Bicycles (Old)",
    "no_access_bicycles.png": "No Access for Bicycles",
    "no_access_bus_haulers.png": "No Access for Buses and Haulders",
    "no_access_bus.png": "No Access for Buses",
    "no_access_cars_bikes_old.png": "No Access for Cars and Bikes (Old)",
    "no_access_cars_bikes.png": "No Access for Bikes and Cars",
    "no_access_haulers_old.png": "No Access for Haulers (Old)",
    "no_access_haulers.png": "No Access for Haulers",
    "no_access_hazardous.png": "No Access for Hazard Carriers",
    "no_access_millue_zone.png": "Millue Zone",
    "no_access_mopeds_bicycles_old.png": "No Access for Mopeds and Bicycles (Old)",
    "no_access_mopeds_bicycles.png": "No Access for Mopeds and Bicycles",
    "no_access_mopeds_old.png": "No Acccess for Mopeds (Old)",
    "no_access_mopeds.png": "No Access for Mopeds",
    "no_access_more_two_old.png": "No Access for Vehicles with More than 2 Wheels (Old)",
    "no_access_more_two.png": "No Access for Vehicles with More than 2 Wheels",
    "no_access_ped_old.png": "No Pedesterians (Old)",
    "no_access_ped.png": "No Pedesterians",
    "no_access_slow_bicycle_bike_invalid.png": "No Access for Two-Wheeled Slow Vehicles",
    "no_access_slow_old.png": "No Access for Slow Vehicles (Old)",
    "no_access_slow.png": "No Access for Slow Vehicles",
    "no_access_trailers_old.png": "No Access for Trailer Cars (Old)",
    "no_access_trailers.png": "No Access for Trailer Cars",
    "no_access_weight_5_4.png": "No Access for Weights Larger than 5.4t",
    "no_access_wide_load.png": "No Access for Wide Length (min 10m)",
    "no_access_width_2_3.png": "No Acces for Wide Width (min 2.3m)",
    "no_access_wide_height.png": "No Acces for Wide Height (min 3.1m)",
    "no_entry.png": "No Entry",
    "no_overtaking_trucks.png": "No Overtaking Haulers",
    "no_overtaking.png": "No Overtaking",
    "no_parking_bikes_mopeds.png": "No Parking Bikes and Mopeds",
    "no_parking.png": "No Parking",
    "no_stopping.png": "No Stopping",
    "no_u_turn.png": "No U Turn",
    "road_closed.png": "Road Closed",
    "speed_limit_50.png": "Speed Limit (min 50)",
    "stop.png": "Stop"
}


queries_imgs = {k.split("/")[-1]: cv2.imread(k, 0) for k in imgs_temp}
orb = cv2.ORB_create()
queries_descriptors = {k: orb.detectAndCompute(
    v, None)[1] for k, v in queries_imgs.items()}


def orb_matcher(img: np.array, threshold=60,
                norm=st.MatchNorm.HAMMING,
                mode=st.ClassiferAggMode.MEAN,
                post=st.ClassifierPostOp.MIN,
                comp=st.ClassifierThreshComparator.SMALLER_THAN_EQ) -> Union[str, Dict, int]:
    global orb
    global queries_descriptors

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if norm == st.MatchNorm.HAMMING:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif norm == st.MatchNorm.HAMMING2:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    elif norm == st.MatchNorm.L1:
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    elif norm == st.MatchNorm.L2:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif norm == st.MatchNorm.L2SQR:
        matcher = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)
    elif norm == st.MatchNorm.INF:
        matcher = cv2.BFMatcher(cv2.NORM_INF, crossCheck=True)
    elif norm == st.MatchNorm.MINMAX:
        matcher = cv2.BFMatcher(cv2.NORM_MINMAX, crossCheck=True)

    mode_func = np.mean

    if mode == st.ClassiferAggMode.MEAN:
        mode_func = np.mean
    elif mode == st.ClassiferAggMode.MAX:
        mode_func = np.max
    elif mode == st.ClassiferAggMode.MIN:
        mode_func = np.min
    elif mode == st.ClassiferAggMode.MEDIAN:
        mode_func = np.median
    elif mode == st.ClassiferAggMode.AVG:
        mode_func = np.average
    elif mode == st.ClassiferAggMode.VAR:
        mode_func = np.var

    _, img_descriptors = orb.detectAndCompute(img, None)

    scores_agg = {}

    for k, v in queries_descriptors.items():
        matches = matcher.match(img_descriptors, v)

        dists = [m.distance for m in matches]

        scores_agg[k] = mode_func(dists)

    post_func = min

    if post == st.ClassifierPostOp.MIN:
        post_func = min
    elif post == st.ClassifierPostOp.MAX:
        post_func = max

    post_ = post_func(scores_agg, key=scores_agg.get)

    cond = scores_agg[post_] < threshold

    if comp == st.ClassifierThreshComparator.SMALLER_THAN:
        cond = scores_agg[post_] < threshold
    elif comp == st.ClassifierThreshComparator.SMALLER_THAN_EQ:
        cond = scores_agg[post_] <= threshold
    elif comp == st.ClassifierThreshComparator.LARGER_THAN_EQ:
        cond = scores_agg[post_] >= threshold
    elif comp == st.ClassifierThreshComparator.LARGER_THAN:
        cond = scores_agg[post_] > threshold

    if cond:
        return -1, scores_agg, scores_agg[post_]

    return f"{classes[post_]} - { scores_agg[post_]}", scores_agg, scores_agg[post_]
