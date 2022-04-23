FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN	apt-get -y update && \
	apt-get install --yes python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev && \
	pip install --upgrade pip==21.0.1 && \
	pip3 install setuptools==45 && \
	pip3 install cython && \
	conda install --yes pytorch=0.4.1 cuda90 -c pytorch && \
	pip3 install scikit-learn && \
	pip3 install "Pillow<7" pymongo pandas && \
	pip3 install torchvision==0.2.1 && \
	pip3 install opencv-python && \
	pip3 install openslide-python

ENV	BASE_DIR="/quip_app/quip_prad_cancer_detection"
ENV	PATH="./":$PATH

ENV	MODEL_VER="v1.0"
ENV	MODEL_URL="https://github.com/SBU-BMI/quip_prad_cancer_detection/blob/master/models_cnn/RESNET_34_prostate_beatrice_john___1117_1038_0.9533516227597434_87.t7"

COPY	. ${BASE_DIR}/. 

RUN	chmod 0755 ${BASE_DIR}/scripts/*

WORKDIR	${BASE_DIR}/scripts

CMD ["/bin/bash"]
