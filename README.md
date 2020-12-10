# Vessel Detector

Detects injection-filled and empty vessels in stem tissues.

Author: Suxing Liu (adapted by Wes Bonelli)

## Requirements

Either [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/) is required to run this project in a Unix environment. First, clone the project with `git clone https://github.com/w-bonelli/vessel-analysis.git`.

## Usage

To analyze files in a directory relative to the project root (e.g., `input/directory`):

### Docker

```bash
docker run -v "$(pwd)":/opt/vessel-analysis computationalplantscience/vessel-analysis python3 trait_extract_parallel.py -i input/directory -o output/directory -r 15 -c 500 -ft jpg
```

### Singularity

```bash
singularity exec docker://computationalplantscience/vessel-analysis python3 trait_extract_parallel.py -i /input/directory -o output/directory -r 15 -c 500 -ft jpg
```

The `.czi` file format is also supported.