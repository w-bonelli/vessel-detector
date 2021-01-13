# Vessel Detector

[![Coverage Status](https://coveralls.io/repos/github/w-bonelli/vessel-detector/badge.svg?branch=master)](https://coveralls.io/github/w-bonelli/vessel-detector?branch=master)

Detects injection-filled and empty vessels in stem tissues.

Author: Suxing Liu (adapted by Wes Bonelli)

## Requirements & Installation

Either [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/) is required to run this project in a Unix environment. First, clone the project with `git clone https://github.com/w-bonelli/vessel-analysis.git`.

## Usage

A good way to get started is to run the tests:

```shell
docker run -it -v "$PWD":/opt/dev -w /opt/dev wbonelli/vessel-detector python3 -m pytest -s
```

### Docker

To invoke `vessel-detector` with Docker, run something like:

```bash
docker run -it -v "$PWD":/opt/dev -w /opt/dev wbonelli/vessel-detector python3 vessels.py <input file> -o <output directory> -r 15 -c 500
```

### Singularity

To use Singularity:

```bash
singularity exec docker://wbonelli/vessel-detector python3 vessels.py <input file> -o <output directory> -r 15 -c 500
```

The `.czi` file format is also supported.