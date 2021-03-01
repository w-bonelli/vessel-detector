from glob import glob
from os.path import join
from pathlib import Path

import pytest

from methods import alt1, original
from options import VesselDetectorOptions

test_dir = Path(__file__).parent
output_dir = join(test_dir, 'output')
data_dir = join(test_dir, 'data')
Path(output_dir).mkdir(parents=True, exist_ok=True)
jpg_images = glob(join(data_dir, '*.jpg'))
czi_images = glob(join(data_dir, '*.czi'))


@pytest.mark.parametrize("image_file", jpg_images)
def test_original_jpg(image_file):
    options = VesselDetectorOptions(
        input_file=join(data_dir, image_file),
        output_directory=output_dir)

    original(options)


@pytest.mark.parametrize("image_file", czi_images)
def test_original_czi(image_file):
    options = VesselDetectorOptions(
        input_file=join(data_dir, image_file),
        output_directory=output_dir)

    original(options)


@pytest.mark.parametrize("image_file", jpg_images)
def test_alt1_jpg(image_file):
    options = VesselDetectorOptions(
        input_file=join(data_dir, image_file),
        output_directory=output_dir)

    alt1(options)


@pytest.mark.parametrize("image_file", czi_images)
def test_alt1_czi(image_file):
    options = VesselDetectorOptions(
        input_file=join(data_dir, image_file),
        output_directory=output_dir)

    alt1(options)