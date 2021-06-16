# Vessel Detector

![CI](https://github.com/w-bonelli/vessel-detector/workflows/CI/badge.svg) [![Coverage Status](https://coveralls.io/repos/github/w-bonelli/vessel-detector/badge.svg?branch=master)](https://coveralls.io/github/w-bonelli/vessel-detector?branch=master)

Detects injection-filled and empty vessels in stem tissues.

Author: Suxing Liu (adapted by Wes Bonelli)

## Requirements & Installation

The easiest way to run this project in a Unix environment is with [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/).

## Usage

To explore the `vessel-detector` image, open a shell inside it:

```shell
docker run -it -v "$(pwd)":/opt/vessel-detector -w /opt/vessel-detector wbonelli/vessel-detector bash
```

A good way to get started is to run the tests:

```shell
docker run -it -v "$(pwd)":/opt/dev -w /opt/dev wbonelli/vessel-detector python3 -m pytest -s
```

#### Docker

To run with Docker, use a command like:

```shell
docker run -it -v "$(pwd)":/opt/vessel-detector -w /opt/vessel-detector wbonelli/vessel-detector python3 vd.py detect <input file> -o <output directory> -mr <minimum vessel radius> -ft <filetypes, comma-separated>
```

#### Singularity

To use Singularity:

To use Singularity:

```bash
singularity exec docker://wbonelli/vessel-detector python3 vd.py detect <input file> -o <output directory> -mr <minimum vessel radius> -ft <filetypes, comma-separated>
```

### Supported filetypes

By default, JPG, PNG, and CZI files are supported. To limit the analysis to certain filetypes, use the `-ft` flag (a comma-separated for multiple), for instance: `-ft png,czi`.
