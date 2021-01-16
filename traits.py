import math
from typing import List

import cv2
import numpy as np

from options import VesselDetectorOptions
from results import VesselDetectorResult


def find_contours(
        threshold_image: np.ndarray,
        color_image: np.ndarray,
        options: VesselDetectorOptions) -> (np.ndarray, List[VesselDetectorResult]):
    contours, hierarchy = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = color_image.copy()
    min_area = math.pi * (options.min_radius ** 2)
    max_area = math.pi * ((options.min_radius * 100) ** 2)
    filtered_counters = []
    results = []
    i = 0

    for contour in contours:
        i += 1
        cnt = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
        bounding_rect = cv2.boundingRect(cnt)
        (x, y, w, h) = bounding_rect
        min_rect = cv2.minAreaRect(cnt)
        area = cv2.contourArea(contour)
        rect_area = w * h
        if max_area > area > min_area:
            filtered_counters.append(contour)

            # draw and label contours
            cv2.drawContours(contours_image, [contour], 0, (0, 255, 0), 3)
            cv2.putText(contours_image, str(i), (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # draw min bounding box
            # box = np.int0(cv2.boxPoints(min_rect))
            # cv2.drawContours(contours_image, [box], 0, (0, 0, 255), 2)

            # draw min bounding box
            # box = np.int0(cv2.boxPoints(bounding_rect))
            # cv2.drawContours(contours_image, [bounding_rect], 0, (0, 0, 255), 2)

            result = VesselDetectorResult(
                id=str(i),
                area=area,
                solidity=min(round(area / rect_area, 4), 1),
                max_height=h,
                max_width=w)
            results.append(result)

    print(f"Kept {len(filtered_counters)} of {len(contours)} total contours")

    return contours_image, results
