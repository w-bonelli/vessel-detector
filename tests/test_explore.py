from glob import glob
from os.path import join
from pathlib import Path

import pytest

from explore import explore1
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
