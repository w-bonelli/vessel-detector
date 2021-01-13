from os.path import join
from pathlib import Path

import pytest

from options import VesselDetectorOptions
from traits import extract_traits

test_dir = Path(__file__).parent
output_dir = join(test_dir, 'output')
Path(output_dir).mkdir(parents=True, exist_ok=True)
test_images = [
    'ploc-10-1-1.jpg',
    # 'ploc-10-1-1.czi'
]


@pytest.mark.parametrize("image_file", test_images)
def test_grayscale_region(image_file):
    options = VesselDetectorOptions(
        input_file=join(test_dir, 'data', image_file),
        output_directory=join(test_dir, 'output'))
    output_prefix = join(options.output_directory, options.input_stem)
    pass


def test_color_region():
    pass


def test_find_contours():
    pass


def test_color_cluster():
    pass


def test_grayscale_cluster():
    pass


def test_watershed():
    pass


def test_compute_curvature():
    pass


@pytest.mark.parametrize("image_file", test_images)
def test_extract_traits(image_file):
    options = VesselDetectorOptions(
        input_file=join(test_dir, 'data', image_file),
        output_directory=join(test_dir, 'output'))
    output_prefix = join(options.output_directory, options.input_stem)
    results = extract_traits(options)