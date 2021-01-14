from pathlib import Path

import click

from options import VesselDetectorOptions
from traits import extract_traits


@click.command()
@click.argument('input_file')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-mr', '--min_radius', required=False, type=int, default=15)
@click.option('-ms', '--min_size', required=False, type=int, default=500)
def cli(input_file, output_directory, min_radius, min_size):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    options = VesselDetectorOptions(
        input_file=input_file,
        output_directory=output_directory,
        min_radius=min_radius,
        min_cluster_size=min_size)

    extract_traits(options)


if __name__ == '__main__':
    cli()
