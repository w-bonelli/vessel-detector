from os.path import join
from pathlib import Path

import cv2
import imageio
import numpy as np
import pytest

from options import VesselDetectorOptions

from thresholding import otsu_threshold, adaptive_threshold_gaussian, simple_threshold

test_dir = Path(__file__).parent
output_dir = join(test_dir, 'output')
Path(output_dir).mkdir(parents=True, exist_ok=True)
test_images = [
    'ploc-10-2-3.jpg',
    'ploc-45-2-1.jpg'
]


@pytest.mark.parametrize("test_image", test_images)
def test_simple_threshold(test_image):
    options = VesselDetectorOptions(
        input_file=join(test_dir, 'data', test_image),
        output_directory=join(test_dir, 'output'))
    image = imageio.imread(options.input_file, as_gray=True)

    # apply threshold
    thresholded = simple_threshold(image, 90)

    # make sure image sizes are equal
    assert np.shape(image)[0] == np.shape(thresholded)[0]
    assert np.shape(image)[1] == np.shape(thresholded)[1]

    imageio.imwrite(join(test_dir, 'output', f"{options.input_stem}.threshold.simple.test.png"), thresholded)


@pytest.mark.parametrize("test_image", test_images)
def test_gaussian_adaptive_threshold(test_image):
    options = VesselDetectorOptions(
        input_file=join(test_dir, 'data', test_image),
        output_directory=join(test_dir, 'output'))
    image = cv2.imread(options.input_file, cv2.IMREAD_GRAYSCALE)

    # apply threshold
    thresholded = adaptive_threshold_gaussian(image)

    # make sure image sizes are equal
    assert np.shape(image)[0] == np.shape(thresholded)[0]
    assert np.shape(image)[1] == np.shape(thresholded)[1]

    imageio.imwrite(join(test_dir, 'output', f"{options.input_stem}.threshold.adaptive.test.png"), thresholded)


@pytest.mark.parametrize("test_image", test_images)
def test_otsu_threshold(test_image):
    options = VesselDetectorOptions(
        input_file=join(test_dir, 'data', test_image),
        output_directory=join(test_dir, 'output'))
    image = cv2.imread(options.input_file, cv2.IMREAD_GRAYSCALE)

    # apply threshold
    thresholded = otsu_threshold(image)

    # make sure image sizes are equal
    assert np.shape(image)[0] == np.shape(thresholded)[0]
    assert np.shape(image)[1] == np.shape(thresholded)[1]

    imageio.imwrite(join(test_dir, 'output', f"{options.input_stem}.threshold.otsu.test.png"), thresholded)
