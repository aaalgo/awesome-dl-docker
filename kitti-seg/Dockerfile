from tensorflow/tensorflow:1.0.0-gpu
#ADD requirements.txt /requirements.txt
#RUN pip install -r /requirements.txt
RUN apt-get update && apt-get install -y vim protobuf-compiler libgtk2.0 git libboost-all-dev python-vtk python-qt4 python-qt4-gl python-setuptools python-numpy python-configobj python-opencv
RUN cd / && git clone --depth 1 --recurse-submodules https://github.com/MarvinTeichmann/KittiSeg

