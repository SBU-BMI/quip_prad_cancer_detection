FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN     apt-get -y update && \
        apt-get install --yes python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev && \
	pip install --upgrade pip && \
	conda update -n base -c defaults conda && \
	pip3 install setuptools==45 && \
	pip install cython && \
	conda install --yes pytorch=0.4.1 cuda90 -c pytorch && \
        conda install --yes scikit-learn && \
        pip install "Pillow<7" pymongo pandas && \
        pip install torchvision==0.2.1 && \
        conda install --yes -c conda-forge opencv

RUN 	pip3 install setuptools==45 && pip install openslide-python

COPY	. /root/quip_prad_cancer_detection/.

RUN	chmod 0755 /root/quip_prad_cancer_detection/scripts/*

ENV	BASE_DIR="/root/quip_prad_cancer_detection"
ENV	PATH="./":$PATH
WORKDIR	/root/quip_prad_cancer_detection/scripts

CMD ["/bin/bash"]

