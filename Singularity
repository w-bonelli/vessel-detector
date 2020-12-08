Bootstrap: docker
From: ubuntu:18.04

%labels
	Maintainer: Suxing Liu, Wes Bonelli

%post
	apt update && \
	apt install -y \
		build-essential \
		python3-setuptools \
		python3-pip \
		python3-numexpr \
		libgl1-mesa-glx \
		libsm6 \
		libxext6 \
		libfontconfig1 \
		libxrender1

	pip3 install --upgrade pip && \
	pip3 install numpy \
		Pillow \
		scipy \
		scikit-build \
		scikit-image \
		scikit-learn \
		matplotlib \
		opencv-python \
		openpyxl \
		seaborn \
		imutils && \
	pip3 install numpy --upgrade
	mkdir /lscratch /db /work /scratch
  
