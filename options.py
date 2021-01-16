from pathlib import Path


class VesselDetectorOptions:
    def __init__(self, input_file, output_directory, min_radius=25):
        self.input_file = input_file
        self.input_name = Path(input_file).name
        self.input_stem = Path(input_file).stem
        self.output_directory = output_directory
        self.min_radius = min_radius
