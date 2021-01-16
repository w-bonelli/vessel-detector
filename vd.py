from contextlib import closing
from glob import glob
from multiprocessing import cpu_count, Pool
from os.path import join
from pathlib import Path

import click

from options import VesselDetectorOptions
from segmentation import extract_traits


@click.group()
def cli():
    pass


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-mr', '--min_radius', required=False, type=int, default=15)
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,czi')
def detect(source, output_directory, min_radius, file_types):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    parsed_file_types = [ft.lower() for ft in file_types.split(',')]
    if 'jpg' in parsed_file_types:
        parsed_file_types.append('jpeg')
    if len(parsed_file_types) == 0:
        raise ValueError(f"You must specify file types!")

    if Path(source).is_file():  # if input is a file, just process it
        options = VesselDetectorOptions(
            input_file=source,
            output_directory=output_directory,
            min_radius=min_radius)
        print(f"Searching for vessels with minimum radius {min_radius}px in: {source}")
        extract_traits(source, options.min_radius)
    elif Path(source).is_dir():  # if input is a directory, use as many cores as the host can spare
        sources = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in parsed_file_types), [])
        sources_str = '\n'.join(sources)
        cores = cpu_count()
        print(f"Using {cores} cores to search {len(sources)} files for vessels:\n{sources_str}")
        with closing(Pool(processes=cores)) as pool:
            pool.starmap(extract_traits, [(source, min_radius) for source in sources])
            pool.terminate()


if __name__ == '__main__':
    cli()
