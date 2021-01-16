from glob import glob
from os.path import join
from pathlib import Path

import pytest

from options import VesselDetectorOptions

test_dir = Path(__file__).parent
output_dir = join(test_dir, 'output')
data_dir = join(test_dir, 'data')
Path(output_dir).mkdir(parents=True, exist_ok=True)
test_images = glob(join(data_dir, '*.jpg')) # .extend(glob('*.czi'))


@pytest.mark.parametrize("image_file", test_images)
def test_grayscale_region(image_file):
    options = VesselDetectorOptions(
        input_file=join(data_dir, image_file),
        output_directory=output_dir)
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