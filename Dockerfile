FROM ubuntu:18.04

LABEL maintainer="Suxing Liu, Wes Bonelli"

COPY . /opt/vessel-detector

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    python3 \
    python3-setuptools \
    python3-numexpr \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip3 install --upgrade pip && \
    pip3 install -r /opt/vessel-detector/requirements.txt