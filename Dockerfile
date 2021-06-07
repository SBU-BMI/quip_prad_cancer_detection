FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN	apt-get -y update 
RUN 	apt-get install --yes python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev 
RUN 	pip install --upgrade pip 
RUN 	conda update -n base -c defaults conda 
RUN 	pip3 install setuptools==45 
RUN 	pip install cython 
RUN 	conda install --yes pytorch=0.4.1 cuda90 -c pytorch 
RUN 	conda install --yes scikit-learn 
RUN 	pip3 install "Pillow<7" pymongo pandas 
RUN 	pip3 install torchvision==0.2.1 
RUN 	conda install --yes -c conda-forge opencv

RUN 	pip install openslide-python

ENV	BASE_DIR="/quip_app/quip_prad_cancer_detection"
ENV	PATH="./":$PATH

COPY	. ${BASE_DIR}/. 

RUN	chmod 0755 ${BASE_DIR}/scripts/*

WORKDIR	${BASE_DIR}/scripts

CMD ["/bin/bash"]
