FROM ubuntu:18.04

LABEL maintainer="Suxing Liu, Wes Bonelli"

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-setuptools \
    python3-pip \
    python3-numexpr \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade numpy && \
    pip3 install Pillow \
    scipy \
    scikit-build \
    scikit-image \
    scikit-learn \
    matplotlib \
    opencv-python \
    openpyxl \
    seaborn \
    imutils \
    czifile


RUN mkdir /lscratch /db /work /scratch
