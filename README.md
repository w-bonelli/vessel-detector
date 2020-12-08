# Arabidopsis Rosette Analysis

Author: Suxing Liu

[![Build Status](https://travis-ci.com/Computational-Plant-Science/arabidopsis-rosette-analysis.svg?branch=master)](https://travis-ci.com/Computational-Plant-Science/arabidopsis-rosette-analysis)

Robust and parameter-free plant image segmentation and trait extraction.

1. Process with plant image top view, including whole tray plant image, this tool will segment it into individual images.
2. Robust segmentation based on parameter-free color clustering method.
3. Extract individual plant gemetrical traits, and write output into excel file.

## Requirements

Either [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/) is required to run this project in a Unix environment.

## Usage

### Docker

```bash
docker run computationalplantscience/arabidopsis-rosette-analysis python3 trait_extract_parallel.py -i /input/directory -o /output/directory -ft jpg
```

### Singularity

```bash
singularity exec docker://computationalplantscience/arabidopsis-rosette-analysis python3 trait_extract_parallel.py -i /input/directory -o /output/directory -ft jpg
```