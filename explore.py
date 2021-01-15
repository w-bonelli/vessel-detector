from os.path import join

import cv2
from czifile import czifile
import numpy as np

from options import VesselDetectorOptions
from thresholding import simple_threshold, adaptive_threshold_gaussian, otsu_threshold
from traits import find_contours
from utils import write_results


def detect_edges(image: np.ndarray, stem: str):
    print(f"Finding edges in {stem}")
    edges = cv2.Canny(image.copy(), 100, 200)
    cv2.imwrite(f"{stem}.edges.png", edges)
    return edges


def detect_contours(
        binary: np.ndarray,
        color: np.ndarray,
        options: VesselDetectorOptions,
        stem: str) -> np.ndarray:
    print(f"Finding contours in {stem}")
    contours, results = find_contours(binary.copy(), color.copy(), options)
    cv2.imwrite(f"{stem}.contours.png", contours)
    write_results(results, options, f"{stem}.contours")
    return contours


def invert(image: np.ndarray, stem: str) -> np.ndarray:
    print(f"Inverting {stem}")
    inverted = cv2.bitwise_not(image.copy())
    cv2.imwrite(f"{stem}.inv.png", inverted)
    return inverted


def apply_simple_threshold(image: np.ndarray, stem: str, invert: bool = False) -> np.ndarray:
    print(f"Applying simple binary threshold to {stem}")
    thresh = simple_threshold(image, invert=invert)
    cv2.imwrite(f"{stem}.thresh.simple{'.inv.' if invert else '.'}png", thresh)
    return thresh


def apply_adaptive_threshold(image: np.ndarray, stem: str, invert: bool = False) -> np.ndarray:
    print(f"Applying adaptive threshold to {stem}")
    thresh = adaptive_threshold_gaussian(image, invert=invert)
    cv2.imwrite(f"{stem}.thresh.adaptive{'.inv.' if invert else '.'}png", thresh)
    return thresh


def apply_otsu_threshold(image: np.ndarray, stem: str, invert: bool = False) -> np.ndarray:
    print(f"Applying OTSU threshold to {stem}")
    thresh = otsu_threshold(image, invert=invert)
    cv2.imwrite(f"{stem}.thresh.otsu{'.inv.' if invert else '.'}png", thresh)
    return thresh


def edges_and_contours(
        grayscale: np.ndarray,
        color: np.ndarray,
        options: VesselDetectorOptions,
        stem: str,
        invert: bool):
    # apply thresholds
    thresh_simple = apply_simple_threshold(grayscale, stem, invert)
    thresh_adaptive = apply_adaptive_threshold(grayscale, stem, invert)
    thresh_otsu = apply_otsu_threshold(grayscale, stem, invert)

    # find edges for all 3 threshold types
    # (some may work better than others depending on the input image)
    edges_thresh_simple = detect_edges(thresh_simple, f"{stem}.thresh.simple")
    edges_thresh_adaptive = detect_edges(thresh_adaptive, f"{stem}.thresh.adaptive")
    edges_thresh_otsu = detect_edges(thresh_otsu, f"{stem}.thresh.otsu")

    # find contours for all 3 threshold types
    # (some may work better than others depending on the input image)
    contours_thresh_simple = detect_contours(thresh_simple, color, options, f"{stem}.thresh.simple")
    contours_thresh_adaptive = detect_contours(thresh_adaptive, color, options, f"{stem}.thresh.adaptive")
    contours_thresh_otsu = detect_contours(thresh_otsu, color, options, f"{stem}.thresh.otsu")


def explore1(options: VesselDetectorOptions):
    output_prefix = join(options.output_directory, options.input_stem)
    print(f"Extracting traits from {output_prefix}'")

    # read image in grayscale and color
    if options.input_file.endswith('.czi'):
        grayscale = czifile.imread(options.input_file)
        grayscale.shape = (grayscale.shape[2], grayscale.shape[3], grayscale.shape[4])  # drop first 2 columns
        color = None
    else:
        grayscale = cv2.imread(options.input_file, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(options.input_file)

    cv2.imwrite(f"{output_prefix}.orig.gray.png", grayscale)
    cv2.imwrite(f"{output_prefix}.orig.color.png", color)

    edges_and_contours(grayscale, color, options, f"{output_prefix}", invert=False)
    edges_and_contours(grayscale, color, options, f"{output_prefix}.inv", invert=True)

    # invert grayscale image
    # inv_grayscale = invert(grayscale, output_prefix)
    # edges_and_contours(inv_grayscale, color, options, f"{output_prefix}.inv", invert=True)
