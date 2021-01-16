from os.path import join

import cv2
from czifile import czifile

from explore import edges_and_contours_czi, edges_and_contours
from options import VesselDetectorOptions
from segmentation import clustering_grayscale, clustering_color, apply_watershed, find_vessels
from utils import write_results


def suxing(options: VesselDetectorOptions):
    output_prefix = join(options.output_directory, f"{options.input_stem}.suxing")
    output_ext = 'png'

    # read the image
    if options.input_file.endswith('.czi'):
        image = czifile.imread(options.input_file)
        image.shape = (image.shape[2], image.shape[3], image.shape[4])  # drop first 2 columns
        image_copy = image.copy()
        output_ext = 'jpg'
        cv2.imwrite(f"{output_prefix}.orig.{output_ext}", image_copy)
    else:
        image = cv2.imread(options.input_file)
        image_copy = image.copy()
        cv2.imwrite(f"{output_prefix}.orig.{output_ext}", image_copy)

    # make backup image
    image_copy = image.copy()
    cv2.imwrite(f"{output_prefix}.orig.{output_ext}", image_copy)

    # threshold and clustering
    colorspace = 'lab'
    channels = 'all'
    clusters = 2
    thresh = clustering_grayscale(image_copy, colorspace, channels, clusters) if image.shape[2] == 1 \
        else clustering_color(image_copy, colorspace, channels, clusters)
    cv2.imwrite(f"{output_prefix}.seg.{output_ext}", thresh)

    # invert segmented image
    print(f"Inverting full image")
    inv_orig = cv2.bitwise_not(image.copy())
    cv2.imwrite(f"{output_prefix}.orig.inv.{output_ext}", inv_orig)

    print(f"Inverting segmented image")
    inv_thresh = cv2.bitwise_not(thresh.copy())
    cv2.imwrite(f"{output_prefix}.seg.inv.{output_ext}", inv_thresh)

    ## standard vessel-finding

    # watershed segmentation
    min_distance_value = 5
    labels = apply_watershed(image_copy, thresh, min_distance_value)

    # find vessel contours
    if options.min_radius is not None:
        (avg_curv, label_trait, results) = find_vessels(image_copy, labels, image.shape[2] == 1, options.min_radius)
        write_results(results, options, f"{output_prefix}")
    else:
        (avg_curv, label_trait, results) = find_vessels(image_copy, labels, image.shape[2] == 1)
        write_results(results, options, f"{output_prefix}")
    if label_trait is not None:
        cv2.imwrite(f"{output_prefix}.curv.{output_ext}", label_trait)

    ## inverted vessel-finding

    # watershed segmentation
    min_distance_value = 5
    inv_labels = apply_watershed(inv_orig, inv_thresh, min_distance_value)

    # find vessel contours
    if options.min_radius is not None:
        (avg_curv, label_trait, results) = find_vessels(inv_orig, inv_labels, image.shape[2] == 1, options.min_radius)
        write_results(results, options, f"{output_prefix}.inv")
    else:
        (avg_curv, label_trait, results) = find_vessels(inv_orig, inv_labels, image.shape[2] == 1)
        write_results(results, options, f"{output_prefix}.inv")
    if label_trait is not None:
        cv2.imwrite(f"{output_prefix}.inv.curv.{output_ext}", label_trait)

    return options.input_stem, None, None, None, None, avg_curv


def alt1(options: VesselDetectorOptions):
    output_prefix = join(options.output_directory, f"{options.input_stem}.alt1")
    output_ext = 'png'
    czi = False

    # read the image
    if options.input_file.endswith('.czi'):
        czi = True
        color = czifile.imread(options.input_file)
        color.shape = (color.shape[2], color.shape[3], color.shape[4])  # drop first 2 columns
        image_copy = color.copy()
        grayscale = image_copy
        color = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        output_ext = 'jpg'
        cv2.imwrite(f"{output_prefix}.orig.{output_ext}", image_copy)
    else:
        # color = cv2.imread(options.input_file)
        # grayscale = image_copy
        grayscale = cv2.imread(options.input_file, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(options.input_file)
        image_copy = color.copy()
        cv2.imwrite(f"{output_prefix}.orig.{output_ext}", image_copy)

    cv2.imwrite(f"{output_prefix}.exp.orig.{output_ext}", grayscale)
    cv2.imwrite(f"{output_prefix}.exp.orig.color.{output_ext}", color)

    # edges and contours
    if czi:
        temp_copy = grayscale.copy()
        temp_name = f"{output_prefix}.temp.jpg"
        cv2.imwrite(temp_name, temp_copy)
        grayscale = cv2.imread(temp_name, cv2.IMREAD_GRAYSCALE)
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