import math
from os.path import join

import cv2
import skimage
from czifile import czifile
import numpy as np
from sklearn.cluster import KMeans

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


def detect_circles(grayscale: np.ndarray, color: np.ndarray, options: VesselDetectorOptions, stem: str) -> np.ndarray:
    print(f"Detecting circles in {stem}")
    image_copy = color.copy()
    circles = cv2.HoughCircles(grayscale,
                               cv2.HOUGH_GRADIENT, dp=1, minDist=options.min_radius * 2, param1=50,
                               param2=30, minRadius=options.min_radius, maxRadius=options.min_radius * 10)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for pt in circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(image_copy, (a, b), r, (0, 255, 0), 2)
            cv2.circle(image_copy, (a, b), 1, (0, 0, 255), 3)
    print(f"Found {len(circles)} circles in {stem}")
    cv2.imwrite(f"{stem}.circles.png", image_copy)
    return image_copy


def sobel_edges(image: np.ndarray, stem: str) -> np.ndarray:
    print(f"Finding Sobel edges in {stem}")
    sobel = image.copy()
    sobel_x = cv2.Sobel(sobel, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(sobel, cv2.CV_64F, 0, 1, ksize=3)
    sobel_xy = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
    cv2.imwrite(f"{stem}.sobel.png", sobel_xy)
    return sobel_xy


def kmeans(preprocessed: np.ndarray, color: np.ndarray, options: VesselDetectorOptions, stem: str) -> np.ndarray:
    print(f"K-means clustering {stem}")
    preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
    Z = preprocessed.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, labels, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    print(f"Found {len(labels)} features")
    center = np.uint8(center)
    res = center[labels.flatten()]
    res2 = res.reshape((preprocessed.shape))

    kmeansImage = np.zeros(color.shape[:2], dtype=np.uint8)
    clustering = np.reshape(np.array(labels, dtype=np.uint8),
                            (color.shape[0], color.shape[1]))
    for i, label in enumerate(labels):
        kmeansImage[clustering == label] = int((255) / (10 - 1)) * i

    concatImage = np.concatenate((color,
                                  193 * np.ones((color.shape[0], int(0.0625 * color.shape[1]), 3), dtype=np.uint8),
                                  cv2.cvtColor(kmeansImage, cv2.COLOR_GRAY2BGR)), axis=1)

    # label_hue = np.uint8(179 * labels / np.max(labels))
    # blank_ch = 255 * np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # labeled_img[label_hue == 0] = 0

    cv2.imwrite(f"{stem}.kmeans.overlay.png", concatImage)

    return res2


def kmeans2(image: np.ndarray, stem: str) -> np.ndarray:
    print(f"K-means clustering (version 2) {stem}")
    (width, height, n_channel) = image.shape
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    numClusters = max(2, 200)
    kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)
    pred_label = kmeans.labels_
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))
    sortedLabels = sorted([n for n in range(numClusters)], key=lambda x: -np.sum(clustering == x))
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i

    ret, thresh = cv2.threshold(kmeansImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 150
    img_thresh = np.zeros([width, height], dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img_thresh[output == i + 1] = 255
    cv2.imwrite(f"{stem}.kmeans2.png", img_thresh)
    return img_thresh


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

    # blur image
    # kmeans(cv2.GaussianBlur(thresh_simple.copy(), (5, 5), cv2.BORDER_DEFAULT), color.copy(), options, f"{stem}.thresh.simple")
    # kmeans(cv2.GaussianBlur(thresh_simple.copy(), (5, 5), cv2.BORDER_DEFAULT), color.copy(), options, f"{stem}.thresh.adaptive")
    # kmeans(cv2.GaussianBlur(thresh_simple.copy(), (5, 5), cv2.BORDER_DEFAULT), color.copy(), options, f"{stem}.thresh.otsu")

    # find edges for all 3 threshold types
    edges_thresh_simple = detect_edges(thresh_simple, f"{stem}.thresh.simple")
    edges_thresh_adaptive = detect_edges(thresh_adaptive, f"{stem}.thresh.adaptive")
    edges_thresh_otsu = detect_edges(thresh_otsu, f"{stem}.thresh.otsu")

    # Sobel edge detection
    sobel_thresh_simple = sobel_edges(thresh_simple, f"{stem}.thresh.simple")
    sobel_thresh_adaptive = sobel_edges(thresh_adaptive, f"{stem}.thresh.adaptive")
    sobel_thresh_otsu = sobel_edges(thresh_otsu, f"{stem}.thresh.otsu")

    # find contours for all 3 threshold types
    contours_thresh_simple = detect_contours(thresh_simple, color, options, f"{stem}.thresh.simple")
    contours_thresh_adaptive = detect_contours(thresh_adaptive, color, options, f"{stem}.thresh.adaptive")
    contours_thresh_otsu = detect_contours(thresh_otsu, color, options, f"{stem}.thresh.otsu")

    # find contours for all 3 threshold types
    contours_thresh_simple = detect_contours(sobel_thresh_simple, color, options, f"{stem}.thresh.simple.sobel")
    contours_thresh_adaptive = detect_contours(sobel_thresh_adaptive, color, options, f"{stem}.thresh.adaptive.sobel")
    contours_thresh_otsu = detect_contours(sobel_thresh_otsu, color, options, f"{stem}.thresh.otsu.sobel")


def edges_and_contours_czi(
        grayscale: np.ndarray,
        color: np.ndarray,
        options: VesselDetectorOptions,
        stem: str,
        invert: bool):
    # apply thresholds
    thresh_simple = apply_simple_threshold(grayscale, stem, invert)
    thresh_otsu = apply_otsu_threshold(grayscale, stem, invert)

    # blur image
    # kmeans(cv2.GaussianBlur(thresh_simple.copy(), (5, 5), cv2.BORDER_DEFAULT), color.copy(), options, f"{stem}.thresh.simple")
    # kmeans(cv2.GaussianBlur(thresh_simple.copy(), (5, 5), cv2.BORDER_DEFAULT), color.copy(), options, f"{stem}.thresh.adaptive")
    # kmeans(cv2.GaussianBlur(thresh_simple.copy(), (5, 5), cv2.BORDER_DEFAULT), color.copy(), options, f"{stem}.thresh.otsu")

    # find edges for all 3 threshold types
    edges_thresh_simple = detect_edges(thresh_simple, f"{stem}.thresh.simple")
    edges_thresh_otsu = detect_edges(thresh_otsu, f"{stem}.thresh.otsu")

    # Sobel edge detection
    sobel_thresh_simple = sobel_edges(thresh_simple, f"{stem}.thresh.simple")
    sobel_thresh_otsu = sobel_edges(thresh_otsu, f"{stem}.thresh.otsu")

    # find contours for all 3 threshold types
    contours_thresh_simple = detect_contours(thresh_simple, color, options, f"{stem}.thresh.simple")
    contours_thresh_otsu = detect_contours(thresh_otsu, color, options, f"{stem}.thresh.otsu")

    # find contours for all 3 threshold types
    contours_thresh_simple = detect_contours(sobel_thresh_simple, color, options, f"{stem}.thresh.simple.sobel")
    contours_thresh_otsu = detect_contours(sobel_thresh_otsu, color, options, f"{stem}.thresh.otsu.sobel")


def explore1(options: VesselDetectorOptions):
    output_prefix = join(options.output_directory, options.input_stem)
    output_ext = 'png'
    czi = False
    print(f"Extracting traits from {output_prefix}'")

    # read the image
    if options.input_file.endswith('.czi'):
        czi = True
        image = czifile.imread(options.input_file)
        image.shape = (image.shape[2], image.shape[3], image.shape[4])  # drop first 2 columns
        image_copy = image.copy()
        grayscale = image_copy
        color = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
        output_ext = 'jpg'
        cv2.imwrite(f"{output_prefix}.orig.{output_ext}", image_copy)
    else:
        image = cv2.imread(options.input_file)
        image_copy = image.copy()
        grayscale = image_copy
        cv2.imwrite(f"{output_prefix}.orig.{output_ext}", image_copy)

    cv2.imwrite(f"{output_prefix}.exp.orig.{output_ext}", grayscale)
    cv2.imwrite(f"{output_prefix}.exp.orig.color.{output_ext}", color)

    # edges and contours
    if czi:
        grayscale = cv2.imread(cv2.imwrite())
        edges_and_contours_czi(grayscale, color, options, f"{output_prefix}.exp", invert=False)
        edges_and_contours_czi(grayscale, color, options, f"{output_prefix}.exp.inv", invert=True)
    else:
        edges_and_contours(grayscale, color, options, f"{output_prefix}.exp", invert=False)
        edges_and_contours(grayscale, color, options, f"{output_prefix}.exp.inv", invert=True)

    # circle detection
    # circles_edges_thresh_simple = detect_circles(grayscale, color, options, output_prefix)
    # circles_edges_thresh_adaptive = detect_circles(grayscale, color, options, output_prefix)
    # circles_edges_thresh_otsu = detect_circles(grayscale, color, options, output_prefix)

    # invert grayscale image
    # inv_grayscale = invert(grayscale, output_prefix)
    # edges_and_contours(inv_grayscale, color, options, f"{output_prefix}.inv", invert=True)