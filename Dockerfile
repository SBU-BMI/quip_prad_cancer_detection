FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN	apt-get -y update && \
	apt-get install -y libsm6 \ 
		libxext6 \
		libxrender-dev \
		libglib2.0-0 \
		python3-openslide \
		wget \
		zip \
		libgl1-mesa-glx \
		libgl1-mesa-dev

RUN 	conda update -n base -c defaults conda && \
	conda install --yes scikit-learn && \
	pip install Pillow pymongo && \
	pip install openslide-python && \
	pip install opencv-python

RUN	conda install --yes pytorch=0.4.0 torchvision=0.2.0

COPY	. /root/quip_prad_cancer_detection/.

WORKDIR /root/quip_prad_cancer_detection/models_cnn
RUN	wget -v -O \
	RESNET_34_prostate_trueVal_hard_train__0530_0015_0.954882634484846_1919.t7 \
	-L https://stonybrookmedicine.box.com/shared/static/nv9xs53n9pus5kz9pzv21z97vrwvz4ek.t7 

WORKDIR /root/quip_prad_cancer_detection/scripts
RUN	chmod 0755 *

ENV	PATH="./":$PATH

CMD ["/bin/bash"]
