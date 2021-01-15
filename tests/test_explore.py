from glob import glob
from os.path import join
from pathlib import Path

import cv2
import pytest

from explore import explore1, detect_circles
from options import VesselDetectorOptions

test_dir = Path(__file__).parent
output_dir = join(test_dir, 'output')
data_dir = join(test_dir, 'data')
Path(output_dir).mkdir(parents=True, exist_ok=True)
test_images = glob(join(data_dir, '*.jpg'))


@pytest.mark.parametrize("image_file", test_images)
def test_explore1(image_file):
    options = VesselDetectorOptions(
        input_file=join(data_dir, image_file),
        output_directory=output_dir)

    explore1(options)


@pytest.mark.parametrize("image_file", test_images)
@pytest.mark.skip(reason='bad')
def test_detect_circles(image_file):
    options = VesselDetectorOptions(
        input_file=join(data_dir, image_file),
        output_directory=output_dir)
    output_prefix = join(options.output_directory, options.input_stem)
    grayscale = cv2.imread(options.input_file, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(options.input_file)

    circles = detect_circles(grayscale, color, options, output_prefix)


@pytest.mark.parametrize("image_file", test_images)
@pytest.mark.skip(reason='bad')
def test_detect_circles_adaptive_threshold(image_file):
    options = VesselDetectorOptions(
        input_file=join(data_dir, image_file),
        output_directory=output_dir)
    output_prefix = join(options.output_directory, options.input_stem)
    grayscale = cv2.imread(options.input_file, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(options.input_file)

    circles = detect_circles(grayscale, color, options, output_prefix)
