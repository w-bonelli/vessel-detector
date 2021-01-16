import math
import warnings
from os.path import join
from typing import List

import cv2
import czifile
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from sklearn.cluster import KMeans

from options import VesselDetectorOptions
from results import VesselDetectorResult
from thresholding import otsu_threshold, adaptive_threshold_gaussian, simple_threshold
from utils import write_results

warnings.filterwarnings("ignore")

MBFACTOR = float(1 << 20)


def grayscale_cluster(image, args_num_clusters, min_cluster_size=500):

    image = cv2.filter2D(image, -1, np.ones((5, 5), np.float32) / 25)
    _, image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)

    (width, height) = image.shape

    # Flatten the 2D image array into an MxN feature vector, where M is the number of pixels and N is the dimension (number of channels).
    reshaped = image.reshape(image.shape[0] * image.shape[1], 1)

    # Perform K-means clustering.
    if args_num_clusters < 2:
        print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')

    # define number of cluster
    numClusters = max(2, args_num_clusters)

    # clustering method
    kmeans = KMeans(n_clusters=numClusters, n_init=40, max_iter=500).fit(reshaped)

    # get lables
    pred_label = kmeans.labels_

    # Reshape result back into a 2D array, where each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # Sort the cluster labels in order of the frequency with which they occur.
    sortedLabels = sorted([n for n in range(numClusters)], key=lambda x: -np.sum(clustering == x))

    # Initialize K-means grayscale image; set pixel colors based on clustering.
    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(sortedLabels):
        kmeansImage[clustering == label] = int(255 / (numClusters - 1)) * i

    ret, thresh = cv2.threshold(kmeansImage, 140, 255, cv2.THRESH_BINARY)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    print(f"Found {nb_components} clusters")

    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img_thresh = np.zeros([width, height], dtype=np.uint8)

    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_cluster_size:
            img_thresh[output == i + 1] = 255

    return img_thresh


def compute_watershed(orig, thresh, min_distance_value):
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)

    localMax = peak_local_max(D, indices=False, min_distance=min_distance_value, labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print(f"{len(np.unique(labels)) - 1} unique segments found")

    return labels


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


def extract_traits_internal(
        grayscale_image: np.ndarray,
        color_image: np.ndarray,
        options: VesselDetectorOptions,
        output_prefix: str):

    # enhance contrast
    print("Enhancing contrast")
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    enhanced = clahe.apply(grayscale_image)  # apply CLAHE to the L-channel
    enhanced = cv2.equalizeHist(enhanced)
    enhanced = cv2.medianBlur(enhanced, 3)
    enhanced = cv2.filter2D(enhanced, -1, np.ones((5, 5), np.float32) / 25)
    cv2.imwrite(f"{output_prefix}.enhanced.png", enhanced)

    # simple threshold
    print("Applying simple binary threshold")
    threshold_simple = simple_threshold(grayscale_image)
    cv2.imwrite(f"{output_prefix}.threshold.simple.png", threshold_simple)

    # simple threshold edge detection
    print(f"Finding edges in simple binary mask")
    edges_simple = cv2.Canny(threshold_simple.copy(), 100, 200)
    cv2.imwrite(f"{output_prefix}.threshold.simple.edges.png", edges_simple)

    # simple threshold contour detection
    print(f"Finding contours in simple binary mask")
    contoured_simple, results = find_contours(threshold_simple.copy(), color_image.copy(), options)
    cv2.imwrite(f"{output_prefix}.threshold.simple.contours.png", contoured_simple)
    write_results(results, options, f"{output_prefix}.threshold.simple.contours")

    # adaptive threshold
    print("Applying adaptive threshold")
    threshold_adaptive = adaptive_threshold_gaussian(grayscale_image)
    cv2.imwrite(f"{output_prefix}.threshold.adaptive.png", threshold_adaptive)

    # adaptive threshold contour detection
    print(f"Finding contours in adaptive mask")
    contoured_adaptive, results = find_contours(threshold_adaptive.copy(), color_image.copy(), options)
    cv2.imwrite(f"{output_prefix}.threshold.adaptive.contours.png", contoured_adaptive)
    write_results(results, options, f"{output_prefix}.threshold.adaptive.contours")

    # dilation/erosion/closing
    print(f"Dilating, eroding, and closing image")
    kernel = np.ones((7, 7), np.uint8)
    dilated_image = cv2.dilate(threshold_adaptive.copy(), kernel, iterations=1)
    eroded_image = cv2.erode(threshold_adaptive.copy(), kernel, iterations=1)
    closed_image = cv2.morphologyEx(dilated_image.copy(), cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{output_prefix}.threshold.adaptive.dilated.png", dilated_image)
    cv2.imwrite(f"{output_prefix}.threshold.adaptive.eroded.png", eroded_image)
    cv2.imwrite(f"{output_prefix}.threshold.adaptive.closed.png", closed_image)

    # closed adaptive threshold contour detection
    print(f"Finding contours in closed adaptive mask")
    contoured_adaptive_closed, results = find_contours(closed_image.copy(), color_image.copy(), options)
    cv2.imwrite(f"{output_prefix}.threshold.adaptive.closed.contours.png", contoured_adaptive_closed)
    write_results(results, options, f"{output_prefix}.threshold.adaptive.closed.contours")

    # OTSU threshold
    print("Applying OTSU threshold")
    threshold_otsu = otsu_threshold(grayscale_image)
    cv2.imwrite(f"{output_prefix}.threshold.otsu.png", threshold_otsu)

    # OTSU threshold contour detection
    print(f"Finding contours in OTSU mask")
    contoured_otsu, results = find_contours(threshold_otsu.copy(), color_image.copy(), options)
    write_results(results, options, f"{output_prefix}.threshold.otsu.contours")
    cv2.imwrite(f"{output_prefix}.threshold.otsu.contours.png", contoured_otsu)


def extract_traits(options: VesselDetectorOptions):
    output_prefix = join(options.output_directory, options.input_stem)
    print(f"Extracting traits from image '{options.input_file}'")

    # read image
    if options.input_file.endswith('.czi'):
        image_grayscale = czifile.imread(options.input_file)
        image_grayscale.shape = (image_grayscale.shape[2], image_grayscale.shape[3], image_grayscale.shape[4])  # drop first 2 columns
        image_grayscale = cv2.filter2D(image_grayscale, -1, np.ones((5, 5), np.float32) / 25)
        image_color = None
    else:
        image_grayscale = cv2.imread(options.input_file, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(options.input_file)

    # invert images
    inverted_grayscale = cv2.bitwise_not(image_grayscale.copy())
    inverted_color = cv2.bitwise_not(image_color.copy())
    cv2.imwrite(f"{output_prefix}.inverted.png", inverted_grayscale)
    cv2.imwrite(f"{output_prefix}.inverted.color.png", inverted_color)

    # extract traits from original
    extract_traits_internal(
        grayscale_image=image_grayscale.copy(),
        color_image=image_color.copy(),
        options=options,
        output_prefix=join(options.output_directory, options.input_stem))

    # extract traits from inverted
    extract_traits_internal(
        grayscale_image=inverted_grayscale.copy(),
        color_image=inverted_color.copy(),
        options=options,
        output_prefix=join(options.output_directory, f"{options.input_stem}.inverted"))

